#!/bin/bash

python imagenet.py --attack-iter 1 --attack-epsilon 1 --attack-step-size 1 -a resnet50 --train-batch 256 --num_classes 1000 --data /path/of/ImageNet --epochs 105 --schedule 30 60 90 100 --gamma 0.1 -c checkpoints/imagenet/advresnet-resnet50-smoothing --gpu-id 0,1,2,3,4,5,6,7 --lr_schedule step --mixbn