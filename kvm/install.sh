#!/bin/bash
rmmod kvm_intel && \
rmmod kvm && \
insmod arch/x86/kvm/kvm.ko && \
insmod arch/x86/kvm/kvm-intel.ko
