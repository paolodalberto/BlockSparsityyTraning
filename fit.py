import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.keras import backend
from tensorflow.python.ops import gradients

from sparse_conv_2d import SparseBlockConv2d

class FITMetrics:
    """
    Class for computing FIT  metrics:
      
    """

    def __init__(self, model,  ds):
        """
        model: Keras model
        loss_fn: loss function
        x: input data
        y: target data

        NOTE: For now, just operates on a single batch of data
        """
        self.model = model
        self.ds = ds




    def trace_hack_ef(self, layer):
        params = [  v for v in layer.trainable_variables        ]
        num_data = 0
        
        ef  = 0.0
        for ds in self.ds:
            with tf.GradientTape() as inner_tape:
                loss = self.model.loss( ds[1],self.model(ds[0],training=True))
                
            grads = inner_tape.gradient(loss, params)
            ef += sum(grads[0].numpy().flatten()**2)* ds[0].shape[0]
            num_data += ds[0].shape[0]
        temp_hv = ef/num_data
        return temp_hv,None 



    def trace_hack_paolo(self):
        """
        Compute the trace of the Hessian using Hutchinson's method
        max_iter: maximimum number of iterations used to compute trace
        tolerance: tolerance for convergence
        """
        trace = 0.0
        trace_vhv = []
        layer_trace_vhv = []
        trace_weights = []
        for layer in self.model.layers:
            if len(layer.trainable_variables) > 0 and type(layer) in [SparseBlockConv2d or tf.keras.layers.Conv2D]:
                hv,_ = self.trace_hack_ef(layer)
                trace_vhv.append(hv)
                print(layer.name, hv)
                layer.set_hessian(hv)

            #break  # Compute for encoder only
        return np.mean(trace_vhv)

