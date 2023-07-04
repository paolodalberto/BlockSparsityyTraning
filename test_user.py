import tensorflow as tf
import tensorflow.keras   as keras
from tensorflow.keras.utils import to_categorical
import math
import numpy as np
import keras.backend as K
from keras.callbacks import LambdaCallback
from copy import deepcopy
from sparse_conv_2d import SparseBlockConv2d, MakeItBlockSparse, \
    check, set_block_sparsity,set_block_sparsity_priority,MakeItBackToDense, GradientGradient

#from keras_applications.sparseresnet50 import ResNet50
#from keras_applications.resnet50 import ResNet50


def get_hessian_j(model, inputs, outputs,param):
    with tf.device('/CPU:0'):
        with tf.GradientTape() as outer_tape:
            with tf.GradientTape() as inner_tape:
            
            
                loss = model.loss(outputs, model(inputs))

            grads = inner_tape.gradient(loss, params)
        h = outer_tape.gradient(grads, params)
    return h
def get_hessian_(model, inputs):
    loss = tf.reduce_sum(model(inputs))
    return tf.hessians(loss, inputs)
    


            
IMAGENET = True
CIFAR = False
DENSE=False

def boom(model2, opt, train_ds,val_ds, x : int = 1 ):

    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    
    model2.compile(optimizer=opt,
                   loss=loss, 
                   metrics=['accuracy'])
    
            
        #
        #for i in val_ds:
        #    #T = get_hessian_(model2, i[0])
        #    T = get_hessian_j(model2, i[0],i[1])

    print("Evaluate")
    result = model2.evaluate(val_ds, verbose = 2 if 'ZEROS' in os.environ else 1)
    

    checkpoint_filepath = logs+'/checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
    #import pdb; pdb.set_trace()
    A = False
    if A and 'HESSIAN'  in os.environ:
        from hessian import  HessianMetrics, GradientTwo
        from fit import  FITMetrics, Gradient
            
        qtest = train_ds.take(50)


        if True:
            # Fisher/Information
            F = FITMetrics(model2, qtest)
            R = F.trace_hack_paolo(save_gradient=True)
            
        
        else: # True :

            # gradient of gradient information about this solution point 
            G = GradientTwo(model2,qtest)
            R = G.trace_hack_paolo(save_trace=True)
            GradientGradient = True
            
        if False :
            # gradient information about this solution point 
            G2 = Gradient(model2,qtest)
            R2 = G2.trace_hack_paolo()
            GradientGradient = False


        print(R)
        #import pdb; pdb.set_trace()
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

    print("Evaluate")
    result = model2.evaluate(val_ds)
    print(result)
    Z = dict(zip(model2.metrics_names, result if type(result) is list else [result]))
    print(Z)

    if 'HESSIAN'  in os.environ:
        hessian_computed = False
        for layer in model2.layers:
            if  type(layer) in [SparseBlockConv2d ] and layer.get_hessian() != 1.0:
                hessian_computed = True
                #import pdb; pdb.set_trace()
                break
        hessian_computed = False
        #import pdb; pdb.set_trace()
        print(hessian_computed)
        
        if not hessian_computed:
            from hessian import  HessianMetrics, GradientTwo
            from fit import  FITMetrics, Gradient
            
            qtest = train_ds.take(100)
            if False:
                # Fisher/Information
                F = FITMetrics(model2, qtest)
                R = F.trace_hack_paolo(save_gradient=True)
            
        
            else: # True :

                # gradient of gradient information about this solution point 
                G = GradientTwo(model2,qtest)
                R = G.trace_hack_paolo(save_trace=True)
                GradientGradient = True
            
            if False :
                # gradient information about this solution point 
                G2 = Gradient(model2,qtest)
                R2 = G2.trace_hack_paolo()
                GradientGradient = False


            print(R)
            #import pdb; pdb.set_trace()
            
                
    print("saving the model")
    model2.save_weights(logs+"/my_weights.model")
    



    
if __name__ == "__main__":

    import os
    machine = os.environ['MACHINE']
    OPT     = os.environ['IMAGENET_OPT']

    row_sparse = 1 
    if 'COL_SPARSE' in os.environ:
        row_sparse = 1 
    elif 'ROW_SPARSE' in os.environ:
        row_sparse = 0 
    elif 'BLOCK_SPARSE' in os.environ:
        row_sparse = 2 
    
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
        
        print("reading training set",data_dir+"/train/")
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir+"/train/", 
            #subset="training",
            seed = 123,
            label_mode = 'int',
            image_size=(x, y),
            #batch_size=128
            batch_size=64
        )
        # Preprocess the images
        train_ds = train_ds.map(resize_with_crop)
        #train_ds = train_ds[0:50000]

        print("reading validation set", data_dir+"/val")
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir+"/val",
            #subset="validation",
            label_mode = 'int',
            image_size=(x, y),
            #batch_size=128
            batch_size=64
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
        #
        
    if weights is not None:
        print("reading the weights") 
        model2.load_weights(logs+"/"+weights)
        #check(model2)
        #model2.summary()
        #import pdb; pdb.set_trace()



    hessian_computed = False
    if 'HESSIAN'  in os.environ:
        for layer in model2.layers:
            if  type(layer) in [SparseBlockConv2d ] and layer.get_hessian() != 1.0:
                hessian_computed = True
                #import pdb; pdb.set_trace()
                break
            
    #hessian_computed = False
    
    
    #import pdb; pdb.set_trace()
    
    if (not 'HESSIAN'  in os.environ or  hessian_computed) and 'ZEROS' in os.environ:
        #set_block_sparsity(model2, None, False ,False,row_sparse)
        set_block_sparsity_priority(model2, 0.5,row_sparse)

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
                
