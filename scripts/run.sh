#!/bin/bash

#data dir
DD=/data/
GPU=0

# preparation

#python2 prep_data.py

# I'd rather fix the args in script, to avoid losing evaluation condition -- doi
CMD="python2 -u ./train_vgg.py -g $GPU"
#OUT="res.out"
#echo '' > ${OUT}
ROOTKITS=($(ls -1 $DD | grep -v normal))
for valid in ${ROOTKITS[@]}; do
  ARGS="-o results/$valid -t $DD$valid 1"
  echo ${CMD} ${ARGS}
  #${CMD} ${ARGS} | tee -a ${OUT}
  ${CMD} ${ARGS}
done
#python2 train.py -i $DD/normal 0 $DD/adore 1 $DD/normal 0 $DD/afkit 1 $DD/normal 0 $DD/diam 1 -t $DD/srootkit 1
