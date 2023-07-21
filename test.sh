#!/bin/bash

#export IMAGENET_OPT=ADAM
#export EPOCH=2

#python /dockerx/test.py


export PYTHONPATH=/dockerx/keras_applications
export IMAGENET_OPT=SGD
export MACHINE=xsjfislx32
export IMAGENET=/imagenet
export TF_CPP_MIN_LOG_LEVEL=3 
export HESSIAN=1
export BLOCK_SPARSE=1

if [[ $1 == 'installtfmot' ]]
then
     pip install -q tensorflow_model_optimization
fi

if [[ $1 == 'reset' ]]
then
    #rm  /dockerx/logs-$MACHINE/*
    export EPOCH=1
    unset ZEROS

    python3 /dockerx/test.py
fi

#exit

export EPOCH=1

unset ZEROS
for i in #0 #
do
    echo $i
    python3 /dockerx/test_user.py  #| grep -v ETA  
done

export ZEROS="zeros"
export EPOCH=10
python3 /dockerx/test_user.py  #| grep -v ETA
exit
for j in 0 #1 2 3 4
do
    for i in 1 2 3 4 5 6 7 8 9 10
    do
	echo $j$i
	python3 /dockerx/test_user.py  #| grep -v ETA  
    done
done
for i in #1 2 3 4 5 6 7
do
    echo $i
    python3 /dockerx/test_user.py  #| grep -v ETA  
done
