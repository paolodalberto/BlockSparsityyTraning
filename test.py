import tensorflow as tf
import tensorflow.keras   as keras
from tensorflow.keras.utils import to_categorical
import math
import numpy as np
import keras.backend as K
from keras.callbacks import LambdaCallback
from copy import deepcopy
from sparse_conv_2d import SparseBlockConv2d
from keras_applications.sparseresnet50 import ResNet50
#from keras_applications.resnet50 import ResNet50

def check(model2, step : bool = False ):
    for l in model2.layers:
        if type(l) == SparseBlockConv2d:
            w = l.get_weights()[1]
            if not l.sparse_training or  (w.shape[0]==1 and w.shape[1]==1):
                continue
            print("Layer %s weight shape %s " % (l.name, str(w.shape)))
            la = l.get_weights()[0]
            M = np.max(la)
            m = np.min(la)
            print("Gamma   Max %f and Min %f Range %f " % (M,m, M-m ))
            w = l.get_weights()[1]
            M = np.max(w)
            m = np.min(w)
            print("Weights Max %f and Min %f Range %f" % (M,m,M-m))

            ## this will print statistics per row and overall
            l.comp_volumes_per_row(verbose=False)

            if step:
                import pdb;
                pdb.set_trace()

def set_block_sparsity(
        model2,
        ratio_file_name: str = None,
        by_lambda : bool =  False,
        step : bool = False,
        prob : float = 0.50
):

    from os.path import exists
    import random 
    FE = exists(ratio_file_name) if ratio_file_name is not None else False
    D = {}
    
    if FE:
        F = open(ratio_file_name, 'r')

        lines = F.read().split("\n")
        for l in lines:
            k,v = l.split()
            v = float(v)
            D[k] = v
    
    for l in model2.layers:
        if type(l) == SparseBlockConv2d :
            A = True
            while A:
                r = l.zeros_volumes_per_row(
                    D[l.name] if l.name in D else 0.5, by_lambda,step
                )
                if r.find("No")<0 :
                    print(r,l.name,l.gamma.shape)

                A = False
                #if False and r.find("No")==0 :
                #    print(l.gamma)
                #    #import pdb; pdb.set_trace()
            
            if step: import pdb; pdb.set_trace()
    

def resize_with_crop(image, label):
    i = image
    i = tf.cast(i, tf.float32)
    i = tf.image.resize_with_crop_or_pad(i, 224, 224)
    i = tf.keras.applications.resnet50.preprocess_input(i)
    return (i, label)

            
IMAGENET = True
CIFAR = False

def boom(model2, opt, train_ds,val_ds, x : int = 1 ):
        

    model2.compile(optimizer=opt,
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
                   metrics=['accuracy'])
    
    print("Evaluate")
    result = model2.evaluate(val_ds, verbose = 2)

    if False and weights and 'ZEROS' in os.environ:
        # and os.environ['ZEROS']=='zeros'
        print("Done")
    else: 

        checkpoint_filepath = logs+'/checkpoint'
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)
    
    
        print("Train", os.environ["EPOCH"])
        epoch = int(os.environ["EPOCH"]) if "EPOCH" in os.environ else 30
        epoch *=x 
        model2.fit(
            train_ds,
            epochs=epoch,
            validation_data = val_ds,
            verbose = 2,
            callbacks = [model_checkpoint_callback],#[print_weights]
        )
        # You can also evaluate or predict on a dataset.

        print("saving the model")
        model2.save_weights(logs+"/my_weights.model")
        #import pdb; pdb.set_trace()
        #check(model2)

        print("Evaluate")
        result = model2.evaluate(val_ds)
        print(result)
        Z = dict(zip(model2.metrics_names, result if type(result) is list else [result]))
        print(Z)


if __name__ == "__main__":

    import os
    machine = os.environ['MACHINE']
    OPT     = os.environ['IMAGENET_OPT']

    
    BLOCK   = os.environ['BLOCK'] if 'BLOCK' in os.environ else "8x8"
    if BLOCK == "8x8":
        from keras_applications.sparseresnet50 import ResNet50
    elif BLOCK == "4x4":
        from keras_applications.sparseresnet50_4x4 import ResNet50
    elif BLOCK == "4x8":
        from keras_applications.sparseresnet50_4x8 import ResNet50
    elif BLOCK == "8x4":
        from keras_applications.sparseresnet50_8x4 import ResNet50
    else:
        from keras_applications.sparseresnet50 import ResNet50
    print("MACHINE", machine)
    weights = None
    logs = "/dockerx/logs" + "-" + machine
    files = os.listdir(logs)
    
    for f in files:
        if f.find("my_weights.model")>=0:
            weights = "my_weights.model"

    if CIFAR:
        from keras.datasets import cifar10

        #import pdb; pdb.set_trace()
        # load dataset
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        # convert from integers to floats
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        # normalize to range 0-1
        x_train = x_train / 255.0
        x_test = x_test / 255.0
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        val_ds   = tf.data.Dataset.from_tensor_slices((x_test, y_test))



    if   IMAGENET :

        data_dir = os.environ['IMAGENET']
        
        machine = os.environ['IMAGENET']
        
        
        print("reading training set")
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir+"/train/", 
            #subset="training",
            seed = 123,
            label_mode = 'int',
            image_size=(224, 224),
            batch_size=128
        )
        # Preprocess the images
        train_ds = train_ds.map(resize_with_crop)
        #train_ds = train_ds[0:50000]

        print("reading validation set")
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir+"/val",
            #subset="validation",
            label_mode = 'int',
            image_size=(224, 224),
            batch_size=128
        )
        val_ds = val_ds.map(resize_with_crop)
        
        
        
    #batch_size = len(x_train)//500
    
    # summarize loaded dataset
    #print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
    #print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))
    #Train: X=(50000, 32, 32, 3), y=(50000, 1)
    #Test: X=(10000, 32, 32, 3), y=(10000, 1)
    # one hot encode target values
    

    #    import pdb; pdb.set_trace()
    
    if CIFAR:
        print("CIFAR")
        inputs = keras.layers.Input(shape=(32,32,3))
        model2 = ResNet50(
            True,
            None,inputs,(32,32,3),
            classes = 10,
            backend=K,
            layers = keras.layers,
            models = keras.models,
            utils   = keras.utils
        )
    if IMAGENET:
        print("IMAGENET MODEL")
        inputs = keras.layers.Input(shape=(224,224,3))
        model2 = ResNet50(
            True,
            'imagenet' if weights is None else None, #None,
            inputs,(224,224,3),
            classes = 1000,
            sparse_training = False if weights is None else True,
            backend=K,
            layers = keras.layers,
            models = keras.models,
            utils   = keras.utils
        )
        
    if weights is not None:
        print("reading the weights") 
        model2.load_weights(logs+"/my_weights.model")
        #check(model2)
        #import pdb; pdb.set_trace()
    if 'ZEROS' in os.environ:
        # and os.environ['ZEROS']=='zeros'
        set_block_sparsity(model2, None, False ,False)# False if machine == "lx31" else True )
        #import pdb; pdb.set_trace()


    print_weights = LambdaCallback(
        on_epoch_end=lambda batch,
        logs: print([ l.get_weights() for l in model2.layers]))

    print("Compile")

    if OPT == 'ADAM':
        print("ADAM")
        opt = keras.optimizers.Adam(learning_rate=0.001)
        boom(model2, opt, train_ds,val_ds)

    elif OPT == 'SGD':
        print("SGD")
        opt = keras.optimizers.SGD(learning_rate=0.001)
        boom(model2, opt, train_ds,val_ds)
    elif OPT == "MIX":
        print("MIX")
        opt2 = keras.optimizers.SGD(learning_rate=0.001)
        opt1 = keras.optimizers.Adam(learning_rate=0.001)
        print("ADAM")
        boom(model2, opt1, train_ds,val_ds)
        print("SGD")
        boom(model2, opt2, train_ds,val_ds,x=5)

    if False and weights is not None:
        check(model2)
        

