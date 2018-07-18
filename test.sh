#!/usr/bin/env sh
exp=vgg_SCNN_DULR_w9
#data=./data/CULane
#	-val ${data}/list/test.txt \
#	-save experiments/predicts/${exp} \

data=./data
rm ./gen/laneTest.t7
th testLane.lua \
	-model experiments/pretrained/vgg_SCNN_DULR_w9.t7 \
	-data ${data} \
	-val ${data}/example-swarm-data/test.txt \
	-save ${data}/example-swarm-data/predicts \
	-dataset laneTest \
	-shareGradInput true \
	-nThreads 2 \
	-nGPU 1 \
	-batchSize 1 \
	-smooth true
