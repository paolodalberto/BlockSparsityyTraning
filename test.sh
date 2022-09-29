#!/bin/bash

#export IMAGENET_OPT=ADAM
#export EPOCH=2

#python /dockerx/test.py


export PYTHONPATH=/dockerx/keras_applications
export IMAGENET_OPT=SGD
export MACHINE=xsjfislx31
export IMAGENET=/imagenet


if [ $1 == 'installtfmot' ]
then
     pip install -q tensorflow_model_optimization
fi

if [ $1 == 'reset' ]
then
    #rm  /dockerx/logs-$MACHINE/*
    export EPOCH=1
    unset ZEROS

    python /dockerx/test.py
fi

#exit

export EPOCH=3
export ZEROS="zeros"

for i in 0 1 2 3 4 #5 6 7 8 9 10 
do
    echo $i
    python /dockerx/test_user.py  #| grep -v ETA  
done
