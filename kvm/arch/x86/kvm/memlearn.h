#ifndef _MEMLEARN_H_
#define _MEMLEARN_H_

#define PACKET_MAX_SIZE 2048

struct mlearn_info {
  struct kvm *kvm;
  struct kvm_page_track_notifier_node track_node;
  struct hlist_head access_log[24];
  struct hlist_head hooks[24];
  struct list_head wqueue;
};

struct mem_access_entry {
  struct hlist_node link;
  unsigned long gva;
  gpa_t gpa;
  u64 gfn;
  int rwx;
};

struct mem_hook_entry {
  struct hlist_node link;
  struct list_head qlink;
  u64 gfn;
  int rwx;
};

struct mem_access_packet {
  u64 gvas[PACKET_MAX_SIZE];
  int size;
};

int mlearn_init(struct kvm *kvm);
int mlearn_flush(struct kvm *kvm);
int mlearn_log(struct kvm *kvm, void __user *argp);
int mlearn_update_hook_all(struct kvm *kvm, u64 gfn);
#endif
