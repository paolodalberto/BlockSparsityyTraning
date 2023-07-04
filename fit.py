import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.keras import backend
from tensorflow.python.ops import gradients

from sparse_conv_2d import SparseBlockConv2d


class CoVariance:
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


    def gradient(self, layer, v : list ):
        params = [  v for v in layer.trainable_variables        ]
        num_data = 0
        
        ef  = params[0]*0
        for ds in self.ds:
            with tf.GradientTape() as inner_tape:
                loss = self.model.loss( ds[1],self.model(ds[0],training=True))
                
            grads = inner_tape.gradient(loss, params)
            g = grads[0].numpy()/ds[0].shape[0]
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

class Gradient:
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
            with tf.GradientTape() as inner_tape:
                loss = self.model.loss( ds[1],self.model(ds[0],training=True))
                
            grads = inner_tape.gradient(loss, params)
            ef += grads[0].numpy()/ds[0].shape[0]
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

class FITMetrics:
    """
    Class for computing FIT  metrics:
      
    """

    def __init__(self, model,  ds, label : int = 1000, sparse_rate : int = 0.5 ):
        """
        model: Keras model
        loss_fn: loss function
        x: input data
        y: target data

        NOTE: For now, just operates on a single batch of data
        """
        self.model = model
        self.ds = ds
        self.label = label
        self.sparse_rate = sparse_rate




    def trace_hack_ef(self, layer):
        params = [  v for v in layer.trainable_variables        ]
        num_data = 0
        
        ef  = 0.0
        for ds in self.ds:
            with tf.GradientTape() as inner_tape:
                loss = self.model.loss( ds[1],self.model(ds[0],training=True))
                
            grads = inner_tape.gradient(loss, params)
            ef += sum((grads[0].numpy()/ds[0].shape[0]).flatten()**2)
            num_data += ds[0].shape[0]
        temp_hv = ef/num_data
        return temp_hv,None 


    def trace_hack_ef_kl(self, layer):
        params = [  v for v in layer.trainable_variables        ]
        num_data = 0
        kl = tf.keras.losses.KLDivergence()
        ef  = 0.0
        count = 0
        egrad = params[0]*0
        for ds in self.ds:
            V = np.zeros(shape = ((ds[1].shape[0],1000)))
            #import pdb; pdb.set_trace()
            for y in range(ds[1].shape[0]):   V[y,ds[1][y]] = 1
            #import pdb; pdb.set_trace()
            #H = get_hessian_j(self.model, ds[0], ds[1],params[0])
            
            with tf.GradientTape() as inner_tape:
                pred = self.model(ds[0],training=True)
                loss =  kl(V,pred)
                
                grads = inner_tape.gradient(loss, params[0])
            
                egrad += grads
            ef += sum((grads.numpy()).flatten()**2)
            num_data += ds[0].shape[0]
            count +=1
            
        temp_hv = ef/count 
        egrad = egrad/count 
        
        return temp_hv,egrad



    def trace_hack_paolo(self,save_gradient : bool = False):
        """
        Compute the trace of the Hessian using Hutchinson's method
        max_iter: maximimum number of iterations used to compute trace
        tolerance: tolerance for convergence
        """
        trace = 0.0
        trace_vhv = []
        layer_trace_vhv = []
        trace_weights = []

        H =  []
        
        for l in self.model.layers:
            if type(l) == SparseBlockConv2d :
                H.append([l,l.get_hessian()])
                
        H = sorted(H, key = lambda x: -x[1])
        G = False

        
        for layer,h in H:
            if len(layer.trainable_variables) > 0 and type(layer) in [SparseBlockConv2d or tf.keras.layers.Conv2D]:

                gamma = layer.get_gamma()
                CIN, COUT = gamma.shape
                Z = int(self.sparse_rate*COUT*CIN)
                NZG = CIN*COUT - sum(sum(gamma))
                A = CIN//8==0 or  COUT//8==0
                if A or   Z == NZG:
                    continue
                
                #import pdb; pdb.set_trace()
                hv,g = self.trace_hack_ef_kl(layer)
                trace_vhv.append(hv)
                print("fit",CIN, COUT,layer.name, hv,  Z, NZG)
                layer.set_hessian(hv)
                if save_gradient: layer.set_gradient(g)

            #break  # Compute for encoder only
        return np.mean(trace_vhv)

