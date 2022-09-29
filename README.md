# You have a network and you want to introduce (COUT, CIN)  block sparsity

Here you will find three files:
`sparse_conv_2d.py, test.sh, test_user.py`

The basic idea is simple: the test.sh is a bash command that sets some
of the global variables. We use a lot of them here and we apologize
for it in advance.
``` sh export PYTHONPATH=/dockerx/keras_applications
export IMAGENET_OPT=SGD
export MACHINE=xsjfislx31
export IMAGENET=/imagenet
```

For me, `/dockerx/` is the directory for this code. you will find
`kera_application` sub directory accordingly. It goes without saying
that you need a location for the training and validation set and the
type of optimization the training will use. 

`EPOCH, ZEROS, MODEL` are used to specify the number of epochs for
training, whether you like to sparsify the MODEL. There is somewhere
the further variable `SHAPES` specifying the input shapes. For
example, (224,224) for resnet and (299,299) for inception_v3.



So if you like to download and start working using resnet 50 `bash
/dockerx/test.sh reset` The idead is that we sue a model the resnet50
and we download the original weight. We train the new model for one
epoch and we store it.

You will notice that now th convolution will have three weights:
kernel, bias, and gamma. The last one is the mask we are going to use
to sparsify the convolution.

A convolution has a kernel (H,W,CIN, COUT). For all purposes, we do
not touch the H and W dimension.  We abstract the weight as (CIN,
COUT). Take now a block 8x8 (8 CIN and 8 COUT). Our mask L is of size
(CIN/8, COUT/8) and a 1 means we keep the 8x8 block, a 0 means that we
foce to zero that block. The new convolution will take the
(H,W,CIN,COUT) and multiply it for a repeated mask LL accordingly to
the block sizes.

Once we run once the sparsification, the resnet is stored as a sparse
resnet. The mask L is part of the state of the convolution.  By
default, we try to sparsify half of the weights. Every time you start
the process we zero 20% of the available blocks (see the step=20 in
the following code excerpt and the sparse rate is 0.5).

```py     ###
    ##  This is the main function to zero at least one volume of the
    ##  kernel weights. Once we zero the gamma, they stay gamma, You
    ##  can use a differefnt volume measure.
    ##
    ###
    def zeros_volumes_per_row(self,
                              sparse_rate= 0.5,
                              by_lambda = True,
                              verbose = True,
                              step =20):
```

Every time you execute `bash test.sh q` you will introduce at most 20%
zeros per layer (of the 1 every 2 blocks).  By default, we compute the
variance of the volume and we remove the lowest 20%. If for any
reasons, we are pruning an $row$, we give back the zero and search for
the next ones. We try to avoid a complete row and column pruning.  


In principle, in 5 iterations you should be able to zero all 50%. But
the slowest pace of the process is one volume at a time. This can take
a while. In practice, you will be able to zero about 1 in 4 blocks
(25%) in 5 iterations. Due to constraints, salt and pepper, you will
need up 13 iterations. Once we reach the bound there are no further
zeroing and we are just training and (over-fitting).

For resenet50, 50% zeros means 2-3 % loss accuracy (validation). For
inception_v3, about 5-7%. Note that not all convolutions should be
eligible for sparsification. The first layer in resnet50 (having only 3
channel) is not eligible.


To create the docker to run this
```
 ## docker pull rocm/rocm-terminal
 docker pull rocm/tensorflow:latest
 docker run -it --network=host --device=/dev/kfd --device=/dev/dri --ipc=host --shm-size 16G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v /YOUR_DIRECTORY_GOES_HERE/sparsity_by_training/:/dockerx -v /scratch/imagenet_resize/:/imagenet_resize -v /scratch/imagenet/:/imagenet rocm/tensorflow:latest
```




