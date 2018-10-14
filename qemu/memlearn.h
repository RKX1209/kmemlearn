#ifndef __MEMLEARN_H__
#define __MEMLEARN_H__

#define PACKET_MAX_SIZE 2048

void qmp_kvm_hook_start(Error **error);
void qmp_kvm_hook_stop(Error **error);

struct mem_access_packet {
  uint64_t gvas[PACKET_MAX_SIZE];
  int size;
};

typedef struct epoch_t {
  uint64_t *gvas;
  int size;
} Epoch;

#endif
