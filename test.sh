#!/bin/bash

#export IMAGENET_OPT=ADAM
#export EPOCH=2

#python /dockerx/test.py


export PYTHONPATH=/dockerx/keras_applications
export IMAGENET_OPT=SGD
export MACHINE=xsjfislx31
export IMAGENET=/imagenet


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
for i in 0 #
do
    echo $i
    python3 /dockerx/test_user.py  #| grep -v ETA  
done

export ZEROS="zeros"
export EPOCH=1

#1 conv1 tf.Tensor(47.0512, shape=(), dtype=float32)
#2 res2a_branch2a tf.Tensor(0.33241957, shape=(), dtype=float32)
#3 res2a_branch2b tf.Tensor(1.3766584, shape=(), dtype=float32)
#4 res2a_branch2c tf.Tensor(0.1637766, shape=(), dtype=float32)
#5 res2a_branch1 tf.Tensor(1.4589565, shape=(), dtype=float32)
#6 res2b_branch2a tf.Tensor(0.12948115, shape=(), dtype=float32)
#7 res2b_branch2b tf.Tensor(0.40479589, shape=(), dtype=float32)
#8 res2b_branch2c tf.Tensor(0.090712845, shape=(), dtype=float32)
#9 res2c_branch2a tf.Tensor(0.17190307, shape=(), dtype=float32)
#10 res2c_branch2b tf.Tensor(1.0139707, shape=(), dtype=float32)
#1 res2c_branch2c tf.Tensor(0.16286492, shape=(), dtype=float32)
#2 res3a_branch2a tf.Tensor(0.3589161, shape=(), dtype=float32)
#3 res3a_branch2b tf.Tensor(1.0507424, shape=(), dtype=float32)
#4 res3a_branch2c tf.Tensor(0.2132502, shape=(), dtype=float32)
#5 res3a_branch1 tf.Tensor(1.1937575, shape=(), dtype=float32)
#6 res3b_branch2a tf.Tensor(0.27463067, shape=(), dtype=float32)
#7 res3b_branch2b tf.Tensor(1.9278331, shape=(), dtype=float32)
#8 res3b_branch2c tf.Tensor(0.15542035, shape=(), dtype=float32)
#9 res3c_branch2a tf.Tensor(0.26320142, shape=(), dtype=float32)
#20 res3c_branch2b tf.Tensor(2.4598718, shape=(), dtype=float32)
#1 res3c_branch2c tf.Tensor(0.3569789, shape=(), dtype=float32)
#2 res3d_branch2a tf.Tensor(0.70138085, shape=(), dtype=float32)
#3 res3d_branch2b tf.Tensor(3.5594418, shape=(), dtype=float32)
#4 res3d_branch2c tf.Tensor(0.30948037, shape=(), dtype=float32)
#5 res4a_branch2a tf.Tensor(1.3568444, shape=(), dtype=float32)
#6 res4a_branch2b tf.Tensor(12.78583, shape=(), dtype=float32)
#7 res4a_branch2c tf.Tensor(0.44594717, shape=(), dtype=float32)
#8 res4a_branch1 tf.Tensor(3.0487494, shape=(), dtype=float32)
#9 res4b_branch2a tf.Tensor(1.8634641, shape=(), dtype=float32)
#30 res4b_branch2b tf.Tensor(3.119961, shape=(), dtype=float32)
#1 res4b_branch2c tf.Tensor(0.31921563, shape=(), dtype=float32)
#2 res4c_branch2a tf.Tensor(1.6879866, shape=(), dtype=float32)
#3 res4c_branch2b tf.Tensor(4.8080354, shape=(), dtype=float32)
#4 res4c_branch2c tf.Tensor(0.2952157, shape=(), dtype=float32)
#5 res4d_branch2a tf.Tensor(2.9562912, shape=(), dtype=float32)
#6 res4d_branch2b tf.Tensor(4.247236, shape=(), dtype=float32)
#7 res4d_branch2c tf.Tensor(0.5024186, shape=(), dtype=float32)
#8 res4e_branch2a tf.Tensor(1.7780411, shape=(), dtype=float32)
#9 res4e_branch2b tf.Tensor(9.265429, shape=(), dtype=float32)
#40 res4e_branch2c tf.Tensor(0.64654565, shape=(), dtype=float32)
#1 res4f_branch2a tf.Tensor(2.0564718, shape=(), dtype=float32)
#2 res4f_branch2b tf.Tensor(12.431566, shape=(), dtype=float32)
#3 res4f_branch2c tf.Tensor(1.4092813, shape=(), dtype=float32)
#4 res5a_branch2a tf.Tensor(4.2546854, shape=(), dtype=float32)
#5 res5a_branch2b tf.Tensor(26.731361, shape=(), dtype=float32)
#6 res5a_branch2c tf.Tensor(1.7825948, shape=(), dtype=float32)
#7 res5a_branch1 tf.Tensor(4.3340225, shape=(), dtype=float32)
#8 res5b_branch2a tf.Tensor(27.046268, shape=(), dtype=float32)
#9 res5b_branch2b tf.Tensor(17.32937, shape=(), dtype=float32)
#50 res5b_branch2c tf.Tensor(2.2907424, shape=(), dtype=float32)
#1 res5c_branch2a tf.Tensor(24.449781, shape=(), dtype=float32)
#2 res5c_branch2b tf.Tensor(26.119617, shape=(), dtype=float32)
#3 res5c_branch2c tf.Tensor(7.0661697, shape=(), dtype=float32)

for j in 0 1 2 3 4
do
    for i in 1 2 3 4 5 6 7 8 9 10
    do
	echo $j$i
	python3 /dockerx/test_user.py  #| grep -v ETA  
    done
done
for i in 1 2 3 4 5 6 7
do
    echo $i
    python3 /dockerx/test_user.py  #| grep -v ETA  
done
