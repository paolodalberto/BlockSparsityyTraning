import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.keras import backend
from tensorflow.python.ops import gradients

from sparse_conv_2d import SparseBlockConv2d

class HessianMetrics:
    """
    Class for computing Hessian metrics:
        - The top 1 (k) eigenvalues of the Hessian
        - The top 1 (k) eigenvectors of the Hessian
        - The trace of the Hessian
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
        self.x = [ i[0] for i in ds]
        self.y = [ i[1] for i in ds]
        self.batch_size = self.x[0].shape[0]
        #self.batched_x = tf.data.Dataset.from_tensor_slices(self.x).batch(batch_size)
        #self.batched_y = tf.data.Dataset.from_tensor_slices(self.y).batch(batch_size)
        self.layer_indices = self.get_layers_with_trainable_params(model)
        np.random.seed(83158011)

    def get_layers_with_trainable_params(self, model):
        """
        Get the indices of the model layers that have trainable parameters
        """
        layer_indices = []
        for i, layer in enumerate(model.layers):
            if len(layer.trainable_variables) > 0 and type(layer) in [SparseBlockConv2d or tf.keras.layers.Conv2D]:
                layer_indices.append(i)
        return layer_indices

    def hessian_vector_product_hack_paolo(self, v, super_layer_idx=None):
        """
        Compute the Hessian vector product of Hv, where
        H is the Hessian of the loss function with respect to the model parameters
        v is a vector of the same size as the model parameters

        Based on: https://github.com/tensorflow/tensorflow/blob/47f0e99c1918f68daa84bd4cac1b6011b2942dac/tensorflow/python/eager/benchmarks/resnet50/hvp_test.py#L62
        """
        # Compute the gradients of the loss function with respect to the model parameters
        params = [
            v
            for v in self.model.layers[super_layer_idx].trainable_variables
        ]
        num_data = 0
        #import pdb;
        temp_hv = [tf.zeros_like(p) for p in params]
        for ds in self.ds:
            with tf.GradientTape() as outer_tape:
                with tf.GradientTape() as inner_tape:

                    loss = self.model.loss( ds[1],self.model(ds[0],training=True))
                    
                grads = inner_tape.gradient(loss, params)

            hv = outer_tape.gradient(grads, params, output_gradients=v)
            
            #  pdb.set_trace()
            temp_hv = [
                THv1 + Hv1 * float(self.batch_size) for THv1, Hv1 in zip(temp_hv, hv)
            ]
            num_data += self.batch_size
        temp_hv = [THv1 / float(num_data) for THv1 in temp_hv]
        eigenvalue = tf.reduce_sum(
            [tf.reduce_sum(THv1 * v1) for THv1, v1 in zip(temp_hv, v)]
        )
        # Compute the Hessian vector product
        return temp_hv, eigenvalue

    def trace_hack_paolo(self, max_iter=100, tolerance=1e-3):
        """
        Compute the trace of the Hessian using Hutchinson's method
        max_iter: maximimum number of iterations used to compute trace
        tolerance: tolerance for convergence
        """
        trace = 0.0
        trace_vhv = []
        layer_trace_vhv = []
        trace_weights = []
        for sl_i in self.layer_indices:
            super_layer = self.model.layers[sl_i]
            
            params = [
                v
                for v in self.model.layers[sl_i].trainable_variables
            ]
            curr = [ ]
            for i in range(max_iter):
                #while True:
                v = [np.random.uniform(size=p.shape) for p in params]
                # Generate Rademacher random variables
                for vi in v:
                    vi[vi < 0.5] = -1
                    vi[vi >= 0.5] = 1
                v = [tf.convert_to_tensor(vi, dtype=tf.dtypes.float32) for vi in v]
                # Compute the Hessian vector product
                hv,ei = self.hessian_vector_product_hack_paolo(v, super_layer_idx=sl_i)
                # Compute the trace
                curr_trace_vhv = [tf.reduce_sum(vi * hvi) for (vi, hvi) in zip(v, hv)]
                print(super_layer.name, curr_trace_vhv)
                curr.append(curr_trace_vhv[0])
                
            layer_trace_vhv.append(np.mean(curr))
            print(super_layer.name, layer_trace_vhv[-1],curr)
            super_layer.set_hessian(np.mean(curr))
        return np.mean(trace_vhv), layer_trace_vhv




class GradientTwo:
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


    def gradient(self, layer):
        params = [  v for v in layer.trainable_variables        ]
        num_data = 0
        
        ef  = params[0]*0
        for ds in self.ds:
            with tf.GradientTape() as outer_tape:
                with tf.GradientTape() as inner_tape:

                    loss = self.model.loss( ds[1],self.model(ds[0],training=True))
                    
                grads = inner_tape.gradient(loss, params[0])

            h = outer_tape.gradient(grads, params[0])
            ef += h.numpy()/ds[0].shape[0]
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
        tr = [ ]
        for layer in self.model.layers:
            if len(layer.trainable_variables) > 0 and type(layer) in [SparseBlockConv2d or tf.keras.layers.Conv2D]:
                #import pdb; pdb.set_trace()
                hv,_ = self.gradient(layer)
                
                layer.set_gradient(hv)
                t = tf.reduce_sum(hv)
                print(layer.name, t)
                tr += [ t]
                
            #break  # Compute for encoder only
        return np.mean(tr)
