#!/bin/bash

#data dir
DD=./data
GPU=0

# preparation

#python2 prep_data.py

# I'd rather fix the args in script, to avoid losing evaluation condition -- doi
CMD="python2 -u ./train_vgg.py -g $GPU -e 10 -o ./results/kmemlearn.log"
SLICE_MAX=6

function valid_rootkit() {
    ROOTKIT=$1
    VALIDS=($(ls -1 $DD/$ROOTKIT*))
    for i in `seq 1 $SLICE_MAX`
    do
	ARGS="-s $i -t $DD/normal "
	for valid in ${VALIDS[@]}; do
	    ARGS="${ARGS} $valid"
	done
	echo ${CMD} ${ARGS}
	${CMD} ${ARGS}
    done
}

declare ROOTKITS=("adore" "afkit" "diam" "kbeast" "srootkit" "suterusu")

for valid in ${ROOTKITS[@]}; do
    valid_rootkit $valid
done
