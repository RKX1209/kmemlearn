/*
 * Memlearn - An image recognition approach for malware detection using memory access patterns
 *
 * Author:
 *    Ren Kimura <rkx1209dev@gmail.com>
 */

#include <linux/init.h>
#include <linux/device.h>
#include <linux/hashtable.h>
#include <linux/mm.h>
#include <linux/mmu_context.h>
#include <linux/types.h>
#include <linux/list.h>
#include <linux/rbtree.h>
#include <linux/spinlock.h>
#include <linux/eventfd.h>
#include <linux/uuid.h>
#include <linux/kvm_host.h>
#include <linux/vfio.h>
#include <linux/mdev.h>
#include <asm/page.h>

#include "memlearn.h"

#define hash_key_exist(name, key) \
	!hlist_empty(&name[hash_min(key, HASH_BITS(name))])

struct mlearn_info *g_info = NULL;

static u64 kernel_area_start = 0xffff880000000000UL;
static u64 guest_phys_base = 0; //XXX: Depend on guest environment
static size_t kernel_size = 0x100000000;
static u64 kernel_start_page, kernel_end_page;
static struct mem_access_packet g_packet;

static bool mlearn_in_kernel_text(u64 gpa) {
	return guest_phys_base <= gpa && gpa <= guest_phys_base + kernel_size;
}

static bool mlearn_in_kernel_text_page(u64 gfn) {
	return kernel_start_page <= gfn && gfn <= kernel_end_page;
}

static inline bool mlearn_initialized(void) {
	return (g_info != NULL);
}

static unsigned long guest_phys_to_virt(void *gpa) {
	unsigned long gva = (unsigned long)gpa;
	if (mlearn_in_kernel_text((u64)gpa)) {
		gva += kernel_area_start;
	}
	return gva;
}

/* XXX: Ugly duplicate functions... */
static struct mem_access_entry *
mlearn_access_table_find(struct mlearn_info *info, u64 gfn) {
	struct mem_access_entry *p, *res = NULL;
	hash_for_each_possible(info->access_log, p, link, gfn) {
		if (gfn == p->gfn) {
			res = p;
			break;
		}
	}
	return res;
}

static struct mem_hook_entry *
mlearn_hook_table_find(struct mlearn_info *info, u64 gfn) {
	struct mem_hook_entry *p, *res = NULL;
	hash_for_each_possible(info->hooks, p, link, gfn) {
		if (gfn == p->gfn) {
			res = p;
			break;
		}
	}
	return res;
}

static bool mlearn_gfn_is_accessed(struct mlearn_info *info, u64 gfn, enum kvm_page_track_mode mode) {
	struct mem_access_entry *p;
	p = mlearn_access_table_find (info, gfn);
	if (!p)
		return false;
	return p->rwx & mode;
}

/*
 * NOTE: kvm_slot_page_track_remove_page doesn't restore page permission:(
 * So I've added new API "kvm_slot_page_track_unprotect_page"(arch/x86/kvm/page_track.c)
 * to unprotect tracked page.
 */
static void __mlearn_page_track(struct kvm_vcpu *vcpu, gpa_t gpa,
		const u8 *val, int len,
		struct kvm_page_track_notifier_node *node, enum kvm_page_track_mode mode)
{
  u64 gfn = gpa_to_gfn(gpa);
	struct kvm *kvm = g_info->kvm;
	if (!mlearn_in_kernel_text_page(gfn)) {
		return;
	}
	// At first, we must restore page permission to previous one.
	/* See: arch/x86/kvm/mmu.c, page_track.c */
	kvm_slot_page_track_unprotect_page(kvm, gfn_to_memslot(kvm, gfn), gfn, mode);
	unsigned long gva = guest_phys_to_virt((void*) gpa);
	struct mem_access_entry *old = mlearn_access_table_find(g_info, gfn);
	if (!mlearn_gfn_is_accessed (g_info, gfn, mode)) {
		if (mode == KVM_PAGE_TRACK_READ)
			printk("[READ] guest addr 0x%lx(0x%016llx), gfn:%llu\n", gva, gpa, gfn);
		else if (mode == KVM_PAGE_TRACK_WRITE)
			printk("[WRITE] guest addr 0x%lx(0x%016llx), gfn:%llu\n", gva, gpa, gfn);
		else
			printk("[EXEC] guest addr 0x%lx(0x%016llx), gfn:%llu\n", gva, gpa, gfn);
		if (!old) {
			struct mem_access_entry *new = vzalloc(sizeof(struct mem_access_entry));
			new->gva = gva;
			new->gpa = gpa;
			new->gfn = gfn;
			new->rwx = mode;
			hash_add(g_info->access_log, &new->link, gfn);
		} else {
			old->rwx |= mode;
		}
	}
}

static void mlearn_page_track_read(struct kvm_vcpu *vcpu, gpa_t gpa,
		const u8 *val, int len,
		struct kvm_page_track_notifier_node *node)
{
	__mlearn_page_track(vcpu, gpa, val, len, node, KVM_PAGE_TRACK_READ);
}

static void mlearn_page_track_write(struct kvm_vcpu *vcpu, gpa_t gpa,
		const u8 *val, int len,
		struct kvm_page_track_notifier_node *node)
{
	__mlearn_page_track(vcpu, gpa, val, len, node, KVM_PAGE_TRACK_WRITE);
}

static void mlearn_page_track_flush_slot(struct kvm *kvm,
		struct kvm_memory_slot *slot,
		struct kvm_page_track_notifier_node *node)
{
	int i;
	u64 gfn;
	struct mlearn_info *info = container_of(node,
					struct mlearn_info, track_node);

	spin_lock(&kvm->mmu_lock);
	for (i = 0; i < slot->npages; i++) {
			gfn = slot->base_gfn + i;
			struct mem_hook_entry *entry = mlearn_hook_table_find(info, gfn);
			if (!entry)
				continue;
			if (entry->rwx & KVM_PAGE_TRACK_READ) {
				kvm_slot_page_track_unprotect_page(kvm, slot, gfn, KVM_PAGE_TRACK_READ);
				kvm_slot_page_track_remove_page(kvm, slot, gfn, KVM_PAGE_TRACK_READ);
			}
			if (entry->rwx & KVM_PAGE_TRACK_WRITE) {
				kvm_slot_page_track_unprotect_page(kvm, slot, gfn, KVM_PAGE_TRACK_WRITE);
				kvm_slot_page_track_remove_page(kvm, slot, gfn, KVM_PAGE_TRACK_WRITE);
			}
			if (entry->rwx & KVM_PAGE_TRACK_EXEC) {
				kvm_slot_page_track_unprotect_page(kvm, slot, gfn, KVM_PAGE_TRACK_EXEC);
				kvm_slot_page_track_remove_page(kvm, slot, gfn, KVM_PAGE_TRACK_EXEC);
			}
	}
	spin_unlock(&kvm->mmu_lock);
}

static struct mem_hook_entry *hook_entry_new(u64 gfn, int rwx) {
	struct mem_hook_entry *hent = vzalloc(sizeof(struct mem_hook_entry));
	if (!hent) {
		return NULL;
	}
	hent->gfn = gfn;
	hent->rwx = rwx;
	return hent;
}

static int mlearn_set_hook_page(struct mlearn_info *info, u64 gfn, enum kvm_page_track_mode mode) {
	struct kvm* kvm = info->kvm;
  struct kvm_memory_slot *slot;
	struct mem_hook_entry *hent;
	int idx;
	//idx = srcu_read_lock(&kvm->srcu);
	slot = gfn_to_memslot(kvm, gfn);
	if (!slot)
		goto failed;
	//spin_lock(&kvm->mmu_lock);

  //printk("[HOOK] guest page:%llu\n", gfn);
	hent = hook_entry_new (gfn, mode);
	if (!hent)
		goto failed;

	hash_add (info->hooks, &hent->link, gfn);
  kvm_slot_page_track_add_page(kvm, slot, gfn, mode);

	//spin_unlock(&kvm->mmu_lock);
	//srcu_read_unlock(&kvm->srcu, idx);
	return 0;
failed:
	printk("failed to hook %lld\n", gfn);
	return -EINVAL;
}

static int mlearn_set_hook_range(u64 gfn_s, u64 gfn_e, enum kvm_page_track_mode mode) {
	u64 gfn;
	for (gfn = gfn_s; gfn < gfn_e; gfn++)
	{
		mlearn_set_hook_page(g_info, gfn, mode);
	}
	return 0;
}

/* Explicit page hook function */
static int mlearn_set_hook_kernel_text(void) {
	/* XXX: When set hooks to entire kernel area, Guest OS crash. */
	return mlearn_set_hook_range (kernel_start_page, kernel_end_page , KVM_PAGE_TRACK_ALL);
}

/* Implicit page hook function for KVM page fault handler */
int mlearn_update_hook_all(struct kvm *kvm, u64 gfn) {
	struct mem_hook_entry *hent;
	if (mlearn_initialized() && mlearn_in_kernel_text_page(gfn)) {
		//printk("requested hooking for gfn: %lld\n", gfn);
		hent = hook_entry_new (gfn, KVM_PAGE_TRACK_ALL);
		if (!hent)
			return 0;
		list_add (&hent->qlink, &g_info->wqueue);
	}
	return 1;
}

int mlearn_add_packet_log(u64 gva) {
	if (g_packet.size >= PACKET_MAX_SIZE)
		return 0;
	g_packet.gvas[g_packet.size++] = gva;
	return 1;
}

int mlearn_init(struct kvm *kvm)
{
  g_info = vzalloc(sizeof(struct mlearn_info));
  if (!g_info)
    return -ENOMEM;

 	kernel_start_page = guest_phys_base >> PAGE_SHIFT;
	kernel_end_page = (guest_phys_base + kernel_size) >> PAGE_SHIFT;

	g_info->kvm = kvm;
	g_info->track_node.track_read = mlearn_page_track_read;
  g_info->track_node.track_write = mlearn_page_track_write;
	g_info->track_node.track_flush_slot = mlearn_page_track_flush_slot;

	hash_init(g_info->access_log);
	hash_init(g_info->hooks);
	INIT_LIST_HEAD(&g_info->wqueue);

  kvm_page_track_register_notifier(kvm, &g_info->track_node);
	mlearn_set_hook_kernel_text();
  return 0;
}

/* Flush: Reset all page hooks. */
int mlearn_flush(struct kvm *kvm)
{
	struct mem_access_entry *maccess = NULL;
	struct list_head *lptr;
	struct mem_hook_entry *hent;
	int bkt;

	/* Disable hook callback temporary for avoiding race condition */
  kvm_page_track_unregister_notifier(kvm, &g_info->track_node);

	/* Restore hook status of accessed pages. */
	hash_for_each(g_info->access_log, bkt, maccess, link) {
		u64 gfn = maccess->gfn;
		mlearn_set_hook_page (g_info, gfn, KVM_PAGE_TRACK_ALL);
		//kfree (maccess);
	}

	/* Set hook pages requested from page_fault_handler */
	list_for_each(lptr, &g_info->wqueue) {
		hent = list_entry(lptr, struct mem_hook_entry, qlink);
		//printk("add hook from wait queue gfn: %lld(%d)\n", hent->gfn, hent->rwx);
		mlearn_set_hook_page (g_info, hent->gfn, hent->rwx);
		//kfree(hent);
	}

	hash_init(g_info->access_log);
	hash_init(g_info->hooks);
	INIT_LIST_HEAD(&g_info->wqueue);
  kvm_page_track_register_notifier(kvm, &g_info->track_node);
	return 0;
}

/* Get access log. */
int mlearn_log(struct kvm *kvm, void __user *argp)
{
	struct mem_access_entry *maccess = NULL;
	int bkt;
	hash_for_each(g_info->access_log, bkt, maccess, link) {
		u64 gva = maccess->gva;
		if (!mlearn_add_packet_log (gva))
			return 1;
	}
	if (copy_to_user(argp, &g_packet, sizeof(struct mem_access_packet)))
		return 1;
	g_packet.size = 0;
	return 0;
}
