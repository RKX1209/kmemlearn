/*
 * Memlearn client - QEMU client for kmemlearn
 *
 * Author:
 *    Ren Kimura <rkx1209dev@gmail.com>
 */

#include "qemu/osdep.h"
#include "qemu-common.h"
#include "qmp-commands.h"
#include "cpu.h"
#include "memlearn.h"
#include "qemu/osdep.h"
#include "sysemu/kvm_int.h"
#include "sysemu/cpus.h"

#include <sys/ioctl.h>
#include <linux/kvm.h>
#include <stdio.h>

static struct mem_access_packet g_packet;
static GThread *clk_thread;
static GSList *g_epochs;

Epoch *ml_epoch_new(struct mem_access_packet *packet) {
  Epoch *epoch;
  int i;
  epoch = (Epoch *)malloc(sizeof(Epoch));
  if (!epoch)
    return NULL;
  epoch->size = packet->size;
  epoch->gvas = (uint64_t *)malloc(sizeof(uint64_t) * epoch->size);
  for (i = 0; i < packet->size; i++) {
    epoch->gvas[i] = packet->gvas[i];
  }
  return epoch;
}

void unit_dot(void) {
  KVMState *s = kvm_state;
  long res;
  int i;
  res = kvm_vm_ioctl(s, KVM_TRACE_ACCESS_LOG, &g_packet);
  for (i = 0; i < g_packet.size; i++) {
    //printf ("0x%016lx, ", g_packet.gvas[i]);
  }
  //printf ("\n");
  res = kvm_vm_ioctl(s, KVM_TRACE_ACCESS_FLUSH, 0);
  Epoch *epoch = ml_epoch_new(&g_packet);
  if (!epoch)
    return;
  g_epochs = g_slist_append (g_epochs, (void*)epoch);
}

void export_json(const char *path) {
  if (!path)
    return;
  int e, idx;
  FILE *fd = fopen (path, "w");
  GSList *node;
  fprintf(fd, "{\n");
  for (e = 0; node = g_slist_nth(g_epochs, e); e++) {
    Epoch *epoch = (Epoch *)node->data;
    fprintf(fd, "  \"%d\": {\n", e);
    for (idx = 0; idx < epoch->size; idx++) {
      uint64_t gva = epoch->gvas[idx];
      if (idx == epoch->size - 1)
        fprintf(fd, "  \"0x%016lx\":1\n", gva);
      else
        fprintf(fd, "  \"0x%016lx\":1,", gva);
    }
    if (!g_slist_nth(g_epochs, e + 1)) {
      fprintf(fd, "  }\n");
    } else {
      fprintf(fd, "  },\n");
    }
  }
  fprintf(fd, "}\n");
  fprintf (stderr, "saved to %s\n", path);
  fclose(fd);
}

static gpointer clock_thread(gpointer opaque) {
  while (true) {
    unit_dot();
    usleep(100000);
  }
}

static GThread *clock_thread_create(GThreadFunc fn) {
  GThread *thread;
  thread = g_thread_new("clock-thread", fn, NULL);
  return thread;
}

void qmp_kvm_hook_start(Error **error) {
  KVMState *s = kvm_state;
  long res;
  if (clk_thread)
    return;
  printf("[QEMU] Hook Start\n");
  res = kvm_vm_ioctl(s, KVM_TRACE_ACCESS_ENABLE, 0);
  clk_thread = clock_thread_create(clock_thread);
}

void qmp_kvm_hook_stop(Error **error) {
  KVMState *s = kvm_state;
  long res;
  printf("[QEMU] Hook Stop\n");
  res = kvm_vm_ioctl(s, KVM_TRACE_ACCESS_DISABLE, 0);
  export_json ("memlearn.json");
}

void qmp_kvm_hook_flush(Error **error) {
  KVMState *s = kvm_state;
  long res;
  printf("[QEMU] Flush\n");
  res = kvm_vm_ioctl(s, KVM_TRACE_ACCESS_FLUSH, 0);
}

void qmp_kvm_hook_log(Error **error) {
  KVMState *s = kvm_state;
  long res;
  printf("[QEMU] Get log\n");
  res = kvm_vm_ioctl(s, KVM_TRACE_ACCESS_LOG, &g_packet);
  int i;
  for (i = 0; i < g_packet.size; i++) {
    //printf ("0x%016lx, ", g_packet.gvas[i]);
  }
  //printf ("\n");
}
