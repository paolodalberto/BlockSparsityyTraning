import tensorflow as tf
import tensorflow.keras   as keras
from tensorflow.keras.utils import to_categorical
import math
import numpy as np
import keras.backend as K
from keras.callbacks import LambdaCallback
from copy import deepcopy
from sparse_conv_2d import SparseBlockConv2d, MakeItBlockSparse, \
    check, set_block_sparsity,MakeItBackToDense

#from keras_applications.sparseresnet50 import ResNet50
#from keras_applications.resnet50 import ResNet50


    


            
IMAGENET = True
CIFAR = False
DENSE=False

def boom(model2, opt, train_ds,val_ds, x : int = 1 ):
        
    #import pdb; pdb.set_trace()
    model2.compile(optimizer=opt,
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
                   metrics=['accuracy'])
    
    print("Evaluate")
    result = model2.evaluate(val_ds, verbose = 2 if 'ZEROS' in os.environ else 1)

    
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
        verbose = 2 if 'ZEROS' in os.environ else 1,
        callbacks = [model_checkpoint_callback],#[print_weights]
    )
    print("saving the model")
    model2.save_weights(logs+"/my_weights.model")

    print("Evaluate")
    result = model2.evaluate(val_ds)
    print(result)
    Z = dict(zip(model2.metrics_names, result if type(result) is list else [result]))
    print(Z)


if __name__ == "__main__":

    import os
    machine = os.environ['MACHINE']
    OPT     = os.environ['IMAGENET_OPT']

    DENSE = False
    try:
        DENSE = os.environ['DENSE']
        DENSE = True
    except:
        DENSE = False
    try:
        MODEL  = os.environ['MODEL']
        shapes = eval(os.environ['SHAPES'])
    except:
        MODEL  = 'resnet50'
        shapes = (224,224)

    if MODEL == 'resnet50':
        from keras_applications.resnet50 import ResNet50 as Model
    elif MODEL == 'inceptionv3':
        from keras_applications.inception_v3 import InceptionV3 as Model
    else:
        exit(-1)

    x,y = shapes
    print(shapes)
    #import pdb; pdb.set_trace()
    def resize_with_crop(image, label):
        i = image
        i = tf.cast(i, tf.float32)
        i = tf.image.resize_with_crop_or_pad(i, x, y)
        i = tf.keras.applications.resnet50.preprocess_input(i)
        return (i, label)

    print("MACHINE", machine)
    weights = None
    logs = "/dockerx/logs" + "-" + machine
    files = os.listdir(logs)
    
    for f in files:
        if f.find("my_weights.model")>=0:
            weights = "my_weights.model"
        elif DENSE and f.find("my_weights_dense.model")>=0:
            weights = "my_weights_dense.model"
            
    if   IMAGENET :

        data_dir = os.environ['IMAGENET']
        machine = os.environ['IMAGENET']
        
        print("reading training set")
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir+"/train/", 
            #subset="training",
            seed = 123,
            label_mode = 'int',
            image_size=(x, y),
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
            image_size=(x, y),
            batch_size=128
        )
        val_ds = val_ds.map(resize_with_crop)
        
    
    if IMAGENET:
        print("IMAGENET MODEL")
        inputs = keras.layers.Input(shape=(x,y,3))
        original_model = Model(
            True,
            'imagenet' if weights is None else None, #None,
            inputs,(x,y,3),
            classes = 1000,
            backend=K,
            layers = keras.layers,
            models = keras.models,
            utils   = keras.utils
        )
        #import pdb; pdb.set_trace()
        model2 = MakeItBlockSparse(original_model)
        
        #model2.summary()
        #import pdb; pdb.set_trace()
        
    if weights is not None:
        print("reading the weights") 
        model2.load_weights(logs+"/"+weights)
        #check(model2)
        #model2.summary()
        #import pdb; pdb.set_trace()

    if 'ZEROS' in os.environ:
        set_block_sparsity(model2, None, False ,False)
        #model2.summary()
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
        
    if DENSE :
        model3 = MakeItBackToDense(model2)
        model3.save_weights(logs+"/my_weights_dense.model")
                
