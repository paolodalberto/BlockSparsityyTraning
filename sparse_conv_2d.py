import tensorflow as tf
import tensorflow.keras   as keras
import math
import numpy as np
import keras.backend as K
from keras.callbacks import LambdaCallback
from copy import deepcopy
import scipy.stats as stats
import re


GradientGradient = True 


if False :
    import tensorflow_model_optimization as tfmot

    ###
    ## This will be used to quantize a sparse model The idea is to
    ## quantize the map and train it.  At this time is just a place
    ## holder and we will have to manage the BatchNorm
    ###
    class DefaultSparseConvQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
        # Configure how to quantize weights.
        def get_weights_and_quantizers(self, layer):
          return [
              (layer.kernel,
               tfmot.quantization.keras.quantizers.LastValueQuantizer(
                   num_bits=8, symmetric=True, narrow_range=False, per_axis=False
               )),
              (layer.bias,
               tfmot.quantization.keras.quantizers.LastValueQuantizer(
                   num_bits=32, symmetric=True, narrow_range=False, per_axis=False
               )),
              (layer.gamma,
               tfmot.quantization.keras.quantizers.LastValueQuantizer(
                   num_bits=8, symmetric=True, narrow_range=False, per_axis=False
               ))
          ]
        
        # Configure how to quantize activations.
        def get_activations_and_quantizers(self, layer):
            return [
                (layer.activation,
                 tfmot.quantization.keras.quantizers.MovingAverageQuantizer(num_bits=8,
                                                                            symmetric=False,
                                                                            narrow_range=False,
                                                                            per_axis=False)
                )
            ]
        
        def set_quantize_weights(self, layer, quantize_weights):
          # Add this line for each item returned in `get_weights_and_quantizers`
          # , in the same order
          layer.kernel = quantize_weights[0]
          layer.bias   = quantize_weights[1]
          layer.gamma  = quantize_weights[2]
        
        def set_quantize_activations(self, layer, quantize_activations):
          # Add this line for each item returned in `get_activations_and_quantizers`
          # , in the same order.
          layer.activation = quantize_activations[0]
        
        # Configure how to quantize outputs (may be equivalent to activations).
        def get_output_quantizers(self, layer):
          return []
        
        def get_config(self):
          return {}

"""
###
##  Plase holder, this will be removed because the quantizer cannot
##  recognize it
##
###
class DefaultBNQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    # Configure how to quantize weights.
    def get_weights_and_quantizers(self, layer):
      return [
          (w , tfmot.quantization.keras.quantizers.LastValueQuantizer(num_bits=8, symmetric=True, narrow_range=False, per_axis=False)) for w in layer.weights
      ]

    # Configure how to quantize activations.
    def get_activations_and_quantizers(self, layer):
        #return [(layer.activation, tfmot.quantization.keras.quantizers.MovingAverageQuantizer(num_bits=8, symmetric=False, narrow_range=False, per_axis=False))]
        pass 
    def set_quantize_weights(self, layer, quantize_weights):
        # Add this line for each item returned in `get_weights_and_quantizers`
        # , in the same order
        layes.set_weights(quantize_weights)
        
    def set_quantize_activations(self, layer, quantize_activations):
        # Add this line for each item returned in `get_activations_and_quantizers`
        # , in the same order.
        #layer.activation = quantize_activations[0]
        pass

    # Configure how to quantize outputs (may be equivalent to activations).
    def get_output_quantizers(self, layer):
      return []

    def get_config(self):
      return {}
"""

###
##   Do you want to try training for the mask ?  We could not but you
##   can try. In this class you can find several version of
##   regularization so that the 0/1 map could be inferred using
##   training and a loss function.  Consider the problem like a
##   constraint problem and we transform the constraints into
##   lagrangian. Again, it did not work for us.
##   This code has been developed in collaboration with Ismail Bustani
###
class LambdaRegularizer(tf.keras.regularizers.Regularizer):

    ## ~ max(x) approximation 
    def lse(self,alpha,x):
        return (1/alpha)*tf.reduce_logsumexp( x*alpha)

    ## ~ max(x) approximation 
    def lse_0(self,alpha,x):
        return (1/alpha)*tf.reduce_logsumexp( (x)*alpha+0 )
    
    ###
    ## a : how close to max the LSE is 
    ## b : penalty L2
    ## c : penalty L1
    ## t : density how many non zero in ratio
    ###
    def __init__(self, a = 100.0, b =1/1000, c=1/1000,t=0.5):
        self.alpha = a ;  self.beta  = b  ;  self.gamma = c;        self.theta = t
        
    ## -(max(gamma)- min(gamma)) + beta*L2(gamma-T) + g*L1(gamma)
    def item_0(self, Gamma):
        T = np.ceil(Gamma.shape[0]*self.theta)          ## non zero blocks       
        a = self.lse(self.alpha, Gamma)                 # ~ max gamma
        b = self.lse(self.alpha, -Gamma)                # ~ min gamma
        c = self.beta*tf.reduce_sum(tf.square(Gamma-T)) # L2(gamma -T)
        d = self.gamma*tf.reduce_sum(tf.abs(Gamma))     # L1(gamma)
        #return  c + d -(a+b)
        return  c+d -(a+b)*self.gamma
        #return  d -(a+b)


    ## max(-Gamma,0) + max(Gamma -1, 0)  + -(max(gamma)- min(gamma)) + beta*L2(gamma-T) + g*L1(gamma)  
    def item_1(self, Gamma):
        T = np.ceil(Gamma.shape[0]*self.theta)          ## non zero blocks       
        a = self.lse(self.alpha, Gamma)                 # ~ max gamma
        b = self.lse(self.alpha, -Gamma)                # ~ min gamma
        c = self.beta*tf.reduce_sum(tf.square(Gamma-T)) # L2(gamma -T)
        d = self.gamma*tf.reduce_sum(tf.abs(Gamma))     # L1(gamma)

        aa = self.lse_0(self.alpha,-Gamma)
        bb = self.lse_0(self.alpha,Gamma-1)

        #return  c + d -(a+b)
        return  aa+bb+c+d +(a+b)*self.gamma 
        #return  d -(a+b)

    ## max(-Gamma,0) + max(Gamma -1, 0)  + -(min(gamma)/max(gamma)) + beta*L2(gamma-T) + g*L1(gamma)  
    def item_2(self, Gamma):
        T = np.ceil(Gamma.shape[0]*self.theta)          ## non zero blocks       
        a = self.lse(self.alpha, Gamma)                 # ~ max gamma
        b = self.lse(self.alpha, -Gamma)                # ~ min gamma
        c = self.beta*tf.reduce_sum(tf.square(Gamma-T)) # L2(gamma -T)
        d = self.gamma*tf.reduce_sum(tf.abs(Gamma))     # L1(gamma)

        aa = self.lse_0(self.alpha,-Gamma)
        bb = self.lse_0(self.alpha,Gamma-1)

        #return  c + d -(a+b)
        return  aa+bb+c+d +(b/(a+0.000001))*self.gamma 
        #return  d -(a+b)

    ## for each output block, we compute the sub-loss function
    def __call__(self, x):
        
        reg = self.item_2(x[:,0])
        for i in range(1,x.shape[1]):
            reg += self.item_2(x[:,i])

        #print("call sub", x.shape,reg.shape)
        #import pdb; pdb.set_trace()
        return reg
    
    def get_config(self):
        return {
            'alpha' : self.alpha, 'beta'  : self.beta,
            'gamma' : self.gamma, 'theta' : self.theta
        }

###
## We can compare the correlation between the value of the mask and
## the volume of the kernel. 
##
###
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

###
##   We take a model with sparse convolutions.
##
##   As option we take a file where each convolution may require a
##   different sparsity ratio.
##
##   We can choose to use only the mask ... this does not work The
##   step is just to use more verbose steps.
##
##   This changes the lambda of the sparse convolution, so when you
##   call this will change the state of the model, the default is to
##   attempt to zero up to 10% of the volumes
###
def set_block_sparsity(
        model2,
        ratio_file_name: str = None,
        by_lambda      : bool =  False,
        step           : bool = False,
        row            : int  = 0
        
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

    H =  []

    for l in model2.layers:
        if type(l) == SparseBlockConv2d :
            H.append(l.get_hessian())
    
    H = sorted(H)
    Q_step = [20, 15, 10] 
    
    for l in model2.layers:
        if type(l) == SparseBlockConv2d :

            i= H.index(l.get_hessian())
            j = 0 if i<len(H)/3 else 1 if i<2*len(H)/3 else 2
            z_step = Q_step[j]
            print(l.name, i,j,len(H),z_step)
            if row ==0 :
                r = l.zeros_volumes_per_row(
                    D[l.name] if l.name in D else 0.5,
                    by_lambda,step,z_step
                )
            elif row ==1 :
                r = l.zeros_volumes_per_col(
                    D[l.name] if l.name in D else 0.5, by_lambda,step,
                    z_step
                )
            elif row ==2 :
                r = l.zeros_volumes_per_block(
                    D[l.name] if l.name in D else 0.5, step, z_step,[2,4]
                )
                
            if r.find("No")<0 :
                print(r,l.name,l.gamma.shape)
                
            
            if step: import pdb; pdb.set_trace()


            
            
def set_block_sparsity_priority(
        model2,
        sparse_rate    : float = 0.5,
        row            : int  = 0
):
    by_lambda = step = False

    H =  []

    for l in model2.layers:
        if type(l) == SparseBlockConv2d :
            H.append([l,l.get_hessian()])
    
    #H = sorted(H, key = lambda x: x[1])

    #import pdb; pdb.set_trace()
    G = False
    for l,h in H:
        if type(l) == SparseBlockConv2d and h >0:

            #import pdb; pdb.set_trace()
            gamma = l.get_gamma()
            CIN, COUT = gamma.shape
            Z = int(sparse_rate*COUT*CIN)
            NZG = CIN*COUT - sum(sum(gamma))
            A = CIN>1 and COUT>1

            while A : 
            
                if Z == NZG:
                    A = False
                    continue

        
                z_step = 100
                #import pdb; pdb.set_trace()
                print(l.name, len(H),z_step)
                if row ==0 :
                    r = l.zeros_volumes_per_row(
                        sparse_rate,
                        by_lambda,step,z_step
                    )
                elif row ==1 :
                    r = l.zeros_volumes_per_col(
                        sparse_rate,
                        by_lambda,step,
                        z_step
                    )
                elif row ==2 :
                    r = l.zeros_volumes_per_block(
                        sparse_rate, step, z_step,[2,4]
                    )
                
                if r.find("No")<0 :
                    print(r,l.name,l.gamma.shape)
                else:
                    A = False
                    gamma = l.get_gamma()
                    NZG = CIN*COUT - sum(sum(gamma))
                    
                    if r.find("Nothing")<0:  G = True
                    else:  G = False
        if G: break

    return None





    
###
##  We take every convolution and create a sparse convolution.  A
##  sparse convolutio has a bit map like matrix we call gamma. Gamma
##  is determined by the block dimension which is 8x8. Now take the
##  channel out (N) and the channel input (C) and create a matrix N/8
##  x C/8.  This is a weight or a bit map that will multiply the
##  correspondent volume of the kernel. Gamma can be trainablteste (see
##  the normalization above) but we could not make it out: so as
##  default it is not trainable.
##
##  The sparse convolution will compute a convolution about
##  Gamma*Kernel, zeroing the kernel volume accordingly to the map in
##  Gamma.
###
def MakeItBlockSparse(
        model,
        dim : tuple = (8,8)
):

    def apply_sparsification(layer):
        if isinstance(layer,tf.keras.layers.Conv2D):
            #import pdb; pdb.set_trace()
            config = layer.get_config()
            config.update(
                {
                    'block_dim'           : dim ,
                    'gamma_regularization': None,
                    'drop_out_prob'       : None, 
                    'sparse_training'     : None,
                }
            )
            
            new = SparseBlockConv2d.from_config(config)
            new.build(layer.input_shape)
            new.kernel.assign(layer.kernel)
            if new.bias is not None and any(new.bias):
                new.bias.assign(layer.bias)
                
            new._inbound_nodes.extend(layer._inbound_nodes)
            new._outbound_nodes.extend(layer._outbound_nodes)
            
            return new
        return layer


    model_new = tf.keras.models.clone_model(
        model,
        clone_function = apply_sparsification
    )
    return model_new

###
##  This routine is not tested.  but we take a sparse convolution and
##  create a dense one (with all the zero explicit)
###

def MakeItBackToDense(
        model,
        dim : tuple = (8,8)
):
    #count = 0 
    def apply_densification(layer):
        if isinstance(layer,SparseBlockConv2d):
            
            config = layer.get_config()
            del config['block_dim']
            del config['gamma_regularization']
            del config['drop_out_prob']
            del config['sparse_training']
            #count+=1
            #if count < 5:
            #    import pdb; pdb.set_trace()
            new = tf.keras.layers.Conv2D.from_config(config)
            new.build(layer.input_shape)
            new.kernel.assign(layer.kernel * layer.get_gamma())
            if new.bias: new.bias.assign(layer.bias)

            return new
        return layer


    model_new = tf.keras.models.clone_model(
        model,
        clone_function = apply_densification
    )
    return model_new


###
##  The SparseBlockConv2d is an extension of a regular Conv2D keras
##  style. There is always a bias and the main difference is an
##  introduction of a Gamma mask. The construction of Gamma is based
##  on the block_dim = default 8x8.
##
##  You can specify if the gamma is trainable or not. Althought the
##  defualt specification is True, in practice is always false. The
##  gammas are set to zero during training little by little.  In
##  principle, we could train them but we could not find any
##  correlation between the value of Gamma and the volume measure.
###

class SparseBlockConv2d(tf.keras.layers.Conv2D):
    def __init__(self,
                 block_dim : tuple=  (8,8),  # ( B_cin, B_cout)
                 sparse_training : bool = True,
                 drop_out_prob : float = 0.5, ## this is not really
                                              ## used
                 # I wanted to believe we could train and find the gamma and thus the zeros.
                 #gamma_regularization =  tf.keras.regularizers.l2(), 
                 gamma_regularization =  None, #LambdaRegularizer(),
                 **kwargs):

        ## we are calling Conv2D
        super(SparseBlockConv2d, self).__init__(**kwargs)

        self.gamma_dim = block_dim
        self.gamma_regularization = gamma_regularization
        self.drop_out_prob = drop_out_prob
        self.sparse_training = sparse_training
        
        
    ## so we can create a class from a dictionary only    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def set_hessian(self, h): self.hessian.assign([h])
    def get_hessian(self): return self.hessian[0]

    def set_gradient(self, h): self.gradient.assign(h)
    def get_gradient(self): return self.gradient

    ## we can create a config and then an instance
    def get_config(self):
        config = super(SparseBlockConv2d, self).get_config()
        
        config.update(
            {
                'block_dim': self.gamma_dim ,
                #'gamma': self.gamma ,
                'gamma_regularization': self.gamma_regularization ,
                'drop_out_prob': self.drop_out_prob,
                'sparse_training': self.sparse_training,
                'hessian'        : self.hessian
            }
        )
        
        return config


    ## this is used when reading the weight from a model that uses
    ## conv2d,
    ## this is obsolete since we are using clone
    def uptick(self,m:str):
        match = re.search("_\d+$",m)
        print(m,match)
        
        if not match:
            m +="_1"
        else:
            b = m[match.span()[0]+1:]
            k = str(int(b)+1)
            m = re.sub("_\d+","_"+k,m)
            #import pdb; pdb.set_trace()
            #print(match,m)
        return m

    ## this is obsolete since we are using clone
    def downtick(self,m:str):
        match = re.search("_\d+$",m)
        print(m,match)
        
        if not match:
            None
        else:
            b = m[match.span()[0]+1:]
            k = str(int(b)-1)
            if k=="0" or k =="-1":
                k = ""
            else:
                k = "_" + k
            m = re.sub("_\d+",k,m)
            #import pdb; pdb.set_trace()
            print(match,m)
        return m

    ## this is obsolete since we are using clone
    ## reading the hfd5 inceptionv3
    def set_weights_from_baseline_2(self,F):

        m = self.name.replace("sparse_block_","").replace("SparseBlock","")
        m = self.uptick(m)
            
        containers = F[m]
        #import pdb; pdb.set_trace()
        for i in containers.keys():
            for j in containers[i].keys():
                W = np.asarray(containers[i][j])
            if W.shape == self.kernel.shape:
                self.kernel.assign(W)
            elif W.shape == self.bias.shape:
                self.bias.assign(W)
            else:
                print("WTF",i)

    ## this is obsolete since we are using clone
    ## reading resnet50
    def set_weights_from_baseline(self,containers):

        #import pdb; pdb.set_trace()
        for i in containers.keys():
            W = np.asarray(containers[i])
            try:
                print(W.shape)
            except:
                for j in containers[i].keys():
                    W = np.asarray(containers[i][j])
            if W.shape == self.kernel.shape:
                self.kernel.assign(W)
            elif W.shape == self.bias.shape:
                self.bias.assign(W)
            else:
                print("WTF",i)

                
    ###
    ## This is the main interface to create a custom layer in Keras
    ## 

    def build(self, input_shape):
        super(SparseBlockConv2d, self).build(input_shape)
        

        dims, repeats = [], []
        for i in range(2):
            dim = math.ceil(self.kernel.shape[i + 2] / self.gamma_dim[i])
            repeat = math.ceil(self.kernel.shape[i + 2] / dim)
            dims.append(dim)
            repeats.append(repeat)
            
        if self.bias is None:
            self.bias = self.add_weight(
                name='bias',
                shape=self.kernel.shape[-1],
                initializer=keras.initializers.zeros
            )
            
        
        
        trainable =   self.sparse_training and (self.kernel.shape[0]>1 or self.kernel.shape[1]>1)

        ## be patient with me you have no idea how much I struggle to
        ## find out if Gamma is trainable and then find out that I
        ## could not

        trainable = False #not trainable
               
        self.gamma = self.add_weight(
            name='lambda',
            shape=dims,
            #initializer=keras.initializers.RandomUniform(minval=0.3, maxval=0.95),
            initializer=keras.initializers.ones,
            regularizer=self.gamma_regularization if trainable else None,
            trainable=trainable
        )
        self.hessian = self.add_weight(
            name='hessian',
            shape=[1],
            initializer=keras.initializers.ones,
            trainable=False
        )
        self.gradient = self.add_weight(
            name='gradient',
            shape=self.kernel.shape,
            initializer=keras.initializers.zeros,
            trainable=False
        )

        ## we overwrite the convolution operation so Gamma kick in
        self.repeats = repeats
        conv_op = deepcopy(self.convolution_op)
        self.convolution_op = lambda inputs, kernel: conv_op(
            inputs,
            kernel * self.get_gamma()
        )

        
    def get_gamma(self,T = 0.00000001):
        gamma = self.gamma

        for i in range(2):
            gamma = tf.repeat(gamma, self.repeats[i], axis=i)
        return gamma[:self.kernel.shape[2], :self.kernel.shape[3]]

#    def set_weights(weights):
#        import pdb; pdb.set_trace()
#        self.gamma.assign(weights[0])
#        super(SparseBlockConv2d,self).set_weights(weights[1:])

    def get_weights(self):
        return super(SparseBlockConv2d,self).get_weights() + \
            [self.gamma.numpy()] 


    def axis(self):
        shape = self.gamma.numpy().shape
        if shape[0] ==1:
            return 1
        if shape[0]<= shape[1]:
            return 0
        return 1

    ###
    ## Order means that we define the order (increasing) for a vector
    ##
    ### 
    def get_gamma_l1_order(self):
        return np.argsort(np.fabs(self.gamma.numpy()),
                          axis=self.axis()
        )
    def get_gamma_l1_order_all(self):
        return np.argsort(np.fabs(self.gamma.numpy().flatten()))


    def constant(self, x):
        return np.sum(x) == x[0]*len(x)


    def V_for_volume(self, X,volume):
        vo = np.zeros(self.gamma.shape)

        for i in range(vo.shape[0]):
        
            B1 = (i+1)*self.gamma_dim[0] 
            if B1 > X.shape[2]:
                B1 = X.shape[2]
            for j in range(vo.shape[1]):
                
                B2 = (j+1)*self.gamma_dim[1]
                if B2 > X.shape[3]:
                    B2 = X.shape[3]
                subvolume = X[:,:,
                              i*self.gamma_dim[0]: B1,
                              j*self.gamma_dim[1]: B2
                ] 
                vo[i,j] = volume(subvolume)
        return vo

    def absolute_order(self,vo):
        TT= np.argsort(vo.flatten())
        TTQ = TT*1
        for i in range(len(TT)):
            TTQ[TT[i]] = i
        TTQ.resize(vo.shape)
        return TTQ


    
    
    def get_gamma_w_cnout(self,
                          X ,
                          volume,
                          step = False):

        vo = self.V_for_volume(X,volume)
        TTQ = self.absolute_order(vo)

        ## absolute order
        t = np.argsort(vo)
        if vo.shape[0]>1:
            for i in range(vo.shape[1]):
                qq = np.argsort(vo[:,i])
                for j in range(vo.shape[0]):
                    t[qq[j],i] = j
                


        ## column order and total order
        return t,TTQ

    
    def get_gamma_w_cnout_2(
            self,
            X ,
            volume,
            step = False):
        vo = self.V_for_volume(X,volume)
        TTQ =self.absolute_order(vo)

    
        ## absolute order
        t = np.argsort(vo)
        if vo.shape[0]>1:
            for i in range(vo.shape[0]):
                qq = np.argsort(vo[i,:])
                for j in range(vo.shape[1]):
                    t[i,qq[j]] = j
                

        ## row order and absolute order
        return t,TTQ 

    def get_gamma_w_cnout_3(
            self,
            X ,
            volume,
            block = [2,4]
    ):

        vo = self.V_for_volume(X,volume)
        TTQ =self.absolute_order(vo)
        
        
        CIN, COUT = vo.shape
        t = np.argsort(vo)
        if CIN>1:
            PO = math.ceil(COUT/block[0])
            PI = math.ceil(CIN/block[0])
            
            
            for o in range(PO):
                RO = min((o+1)*PO,COUT)
                for i in range(PI):
                    RI = min((i+1)*PO,COUT)
                    t[o*PO:RO,i*PI:RI] = self.absolute_order(
                        v[o*PO:RO,i*PI:RI]
                    )
        ## order per block and total
        return t,TTQ 


    ###
    ## Given a volume, a sub tensor of a weight tensor, we compute a
    ## measure of the volume. Smaller the measure less important the
    ## volume is and we can zero it
    ###
    def volume_by_determinant(self,X):
        shape = X.shape
        #import pdb; pdb.set_trace()
        Y = X.reshape(shape[3]*shape[2], shape[0] * shape[1])
        Y = np.matmul(Y,Y.transpose())
        return np.linalg.det(Y)
    def volume_by_euclidian(self,X, T = 0.00000001):
        Y = X.flatten()
        S = np.sum(Y*Y)
        return S if np.abs(S)>T else 0.0
    def volume_by_variance(self,X):
        S =  np.var(X)
        return S #if np.abs(S)>T else 0.0
    def volume_by_mean(self,X):
        S =  np.mean(X)
        return S #if np.abs(S)>T else 0.0
    def volume_by_l1(self,X, T = 0.00000001):
        S =  np.sum(np.fabs(X).flatten())
        return S #if np.abs(S)>T else 0.0

    ###
    ## this is used to compare if the Gamma and the volume measure are
    ## correlated, anf you get a p-value for free.
    ###
    def correlation(self, L, W,f):
        rhos = []
        #import pdb; pdb.set_trace()
        if True or self.axis() ==0 :
            if L.shape[0]>1:
                for i in range(L.shape[1]):
                    r_ = stats.kendalltau(
                        L[:,i],
                        W[:,i]
                    )
                    if math.isnan(r_.correlation):
                        import pdb; pdb.set_trace()
                        print(L[:,i])
                        print(W[:,i])
                    rhos.append([i,r_])
            else:
                rhos.append([-1,stats.kendalltau(
                    L[0:],
                    W[0,:]
                )])
        else:
            if L.shape[1]>1:
                for i in range(L.shape[0]):
                    r_ = stats.kendalltau(
                        L[i,:],
                        W[i,:]
                    )
                    if math.isnan(r_.correlation):
                        import pdb; pdb.set_trace()
                        print(L[i,:])
                        print(W[i,:])

                    rhos.append([i,r_])
            else:
                rhos.append([-1,stats.kendalltau(
                    L[0:],
                    W[0,:])])
        return rhos
    
    def comp_volumes_per_row(self, funcs = None, verbose = True):

        if funcs is None:
            funcs = [
                #self.volume_by_determinant,
                ("euclidian", self.volume_by_euclidian),
                ("variance"  , self.volume_by_variance),
                ("l1"        , self.volume_by_l1)#,
                #("mean"     , self.volume_by_mean)
            ]
            
        
        L   = self.get_gamma_l1_order()
        #L   = np.fabs(self.gamma.numpy())
        if verbose: print(L.shape)
        LWs = []
        
        Rs = []
        R = []
        #import pdb; pdb.set_trace()
        for n,f in funcs:
            
            W,_  = self.get_gamma_w_cnout(
                self.kernel.numpy(),
                f
            )
            LW,_ = self.get_gamma_w_cnout(
                (self.get_gamma()*self.kernel).numpy(),
                f,verbose
            )

            
            R.append(self.correlation(L,W,f))

            Rs.append(self.correlation(L,LW,f))
            #import pdb; pdb.set_trace()

            
        Header = "row-KTau  - LW "+ " ".join([s[0] for s in funcs]) +\
                 "  W "+" ".join([s[0] for s in funcs])
        Row = "%4d -   "+" ".join([ '% 1.3f' for s in funcs])+\
              "   "+" ".join([ '% 1.3f' for s in funcs])
        Rows= "%4s- LW "+" ".join([ '% 1.3f' for s in funcs])+\
              " W "+" ".join([ '% 1.3f' for s in funcs])

        
        A = []
        Tot = []
        AA = []
        TTot = []
        for j in range(len(funcs)):
            M = [ r[1].correlation for r in Rs[j]]
            N = [ r[1].correlation for r in R[j]]
            A.append(np.mean(M)); AA.append(np.mean(N)); 
            Tot.append(np.sum(M));TTot.append(np.sum(N))
            
        
        if verbose:
            print(Header)
            for i in range(len(Rs[0])):
                W = [ Rs[j][i][1].correlation for j in range(len(funcs))]
                W = [i] + W
                V = [ R[j][i][1].correlation for j in range(len(funcs))]
                print(Row % (tuple(W+V)))
            print(Rows % (tuple(['mean'] + A + AA)))
            print(Rows % (tuple(['tot'] + Tot + TTot)))
        #import pdb; pdb.set_trace()
        
        return A



    def set_row(self, gamma, Z1c, QQ, L, Z1r):
        CIN, COUT = gamma.shape
        count =0
        for i in range(CIN):
            if sum(gamma[i,:])==0:
                #import pdb; pdb.set_trace()
                ## largest element column
                jj = QQ[i,:] == COUT-1
                ## next largest row in the column above
                zup = True
                while zup and Z1c[jj]<CIN:
                    ii =(L[:,jj]==Z1c[jj]).flatten()
                    if gamma[ii,jj] ==1:
                        zup = False
                    else:
                        Z1c[jj] += 1
                                    
                count += gamma[ii,jj]
                gamma[i,jj]  = 1
                gamma[ii,jj] = 0
                #import pdb; pdb.set_trace()
            elif False and  COUT-sum(gamma[i,:])>Z1r[i]:
                jj = QQ[i,:] >= Z1r[i]
                gamma[i,jj] ==1
                count += sum(gamma[ii,jj])
                gamma[i,jj]  = 1
                
                #import pdb; pdb.set_trace()
        for i in range(COUT):
            if  CIN-sum(gamma[:,i])>Z1c[i]:
                #import pdb; pdb.set_trace()
                jj = L[:,i] >= Z1c[i]
                gamma[jj,i] =1
                count += sum( jj)
        return count
    def set_col(self, gamma, Z1c, QQ, L, Z1r):
        CIN, COUT = gamma.shape
        count =0
        
        for i in range(CIN):
            if  COUT-sum(gamma[i,:])>Z1r[i]:
                #import pdb; pdb.set_trace()
                jj = L[i,:] >= Z1r[i]
                gamma[i,jj] =1
                count += sum(jj)
        #print(CIN,COUT)
        for i in range(COUT):
            #print("set_col",i,Z1c[i])
            if sum(gamma[:,i])==0:
                #import pdb; pdb.set_trace()
                ## largest element column
                jj = QQ[:,i] == CIN-1
                ## next largest row in the column above
                zup = True
                while zup and Z1r[jj]<COUT:
                    ii =(L[jj,:]==Z1r[jj]).flatten()
                    if gamma[jj,ii] ==1:
                        zup = False
                    else:
                        Z1r[jj] += 1
                                    
                count += sum(gamma[jj,ii])
                gamma[jj,i]  = 1
                gamma[jj,ii] = 0
                #
            elif False and CIN-sum(gamma[:,i])>Z1c[i]:
                #import pdb; pdb.set_trace()
                jj = QQ[:,i] >= Z1c[i]
                gamma[jj,i] =1
                count += sum( jj)
              
        return count
    
    

    ## QQ is a total Order L is a column order
    def reinstate_row_col(self,gamma,Gamma,L,Zr, i):
        gamma[i,L[i,:]>=Zr[i]] = Gamma[i, L[i,:]>=Zr[i]]
    def reinstate_col_col(self,gamma,Gamma, QQ,Zc, i):
        gamma[QQ[:,i]>=Zc[i],i] = Gamma[QQ[:,i]>=Zc[i],i]

    ## QQ is a total Order L is a row order
    def reinstate_col_row(self,gamma,Gamma,L,Zc, i):
        gamma[L[:,i]>=Zc,i] = Gamma[L[:,i]>=Zc,i]
    def reinstate_row_row(self,gamma,Gamma, QQ,Zr, i):
        gamma[i,QQ[i,:]>=Zr[i]] = Gamma[i,QQ[i,:]>=Zr[i]]
        
        
    ## return the empty rows
    def pruned_row(self, gamma):
        CIN, COUT = gamma.shape
        legal = True
        count = []
        for i in range(CIN):
            if sum(gamma[i,:])==0:
                legal = False
                count.append(i)
        return legal, count

    ## return the columns with too many zeros
    def legal_col(self, gamma, Zc):
        CIN, COUT = gamma.shape
        legal = True
        count = []

        for i in range(COUT):
            if CIN-sum(gamma[:,i])>Zc:
                count.append(i)
                legal=False
        return legal, count
    ## no empty rows and just right columns
    def legal(self,gamma,c):
        r0 = self.pruned_row(gamma)
        r1 = self.legal_col(gamma,c)
        return r0[0] and r1[0], [r0[1], r1[1]]


    ## return the columns of gamma that are empty
    def pruned_col(self, gamma):
        CIN, COUT = gamma.shape
        legal = True
        count = []
        for i in range(COUT):
            if sum(gamma[:,i])==0:
                legal = False
                count.append(i)
        return legal, count

    ## return the rows having too many zeros
    def legal_row(self, gamma, Zc):
        CIN, COUT = gamma.shape
        legal = True
        count = []
        
        for i in range(CIN):
            if COUT-sum(gamma[i,:])>Zc:
                count.append(i)
                legal=False
        return legal, count

    ## no empty col and just right rows
    def legal_p(self,gamma,c):
        r0 = self.pruned_col(gamma)
        r1 = self.legal_row(gamma,c)
        return r0[0] and r1[0], [r0[1], r1[1]]

    ###
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


        zero_rate = 1 - sparse_rate
        import pdb; pdb.set_trace()
        count = 0

        
        if by_lambda:
            L   = self.get_gamma_l1_order()
        else:
            L,TT = self.get_gamma_w_cnout(
                (np.max(np.finfo(np.float32).eps+ tf.abs(self.get_gradient()))*\
                 (self.kernel**2 if GradientGradient else self.kernel) *self.get_gamma()
                ).numpy(),
                #self.volume_by_variance
                #self.volume_by_l1
                self.volume_by_euclidian
            )
            
        #

        tra = True or (self.kernel.shape[0]>1 or self.kernel.shape[1]>1)

        
        if tra and len(L.shape)==2 and L.shape[0]>1 and L.shape[1]>1:
            ## Gamma :  cin x cout 
            
            CIN, COUT = L.shape
            
            ## how many zero we would like to have
            Zr = int(zero_rate*COUT)    ## per row
            Zc = int(zero_rate*CIN)     ## per col 
            Z = int(zero_rate*COUT*CIN) ## per all

            
            Gamma = self.gamma.numpy().astype(int)
            Gammat = np.transpose(Gamma)
            # number of zero per row
            GZc   = CIN -sum(Gamma)
            # and per column
            GZr   = COUT - sum(Gammat)
            ## total number of zeros
            NZG = CIN*COUT - sum(sum(Gamma))

            
            ## either increment of 1 or step% of what we need, this is
            ## the number of zeros increment
            K = max(1,int(step*Z/100))
            # but no more that the one we need 
            Z1 = min(NZG+K, Z)

#            print(K,Z)
#            import pdb; pdb.set_trace()
            
            # per dimensions (we are after the columns x output
            Kc = max(1,int(step*Zc/100))
            Kr = max(1,int(step*Zr/100))
            # but no more that the one we need 
            Z1c = np.minimum(GZc+Kc, Zc)
            Z1r = np.minimum(GZr+Kr, Zr)


            import pdb; pdb.set_trace()

            ## using teh total oder I compute row orders
            QQ = TT*1
            for j in range(CIN):
                S = np.argsort(TT[j,:])
                for i in range(COUT):
                    QQ[j,S[i]] = i
            
            if NZG < Z :
                
                # we still have zeros to give, we use the minimum set
                # We set all zeros if we can
                gamma = Gamma*1
                print(self.legal(gamma,Zc))
                gamma[TT<Z1] =0
                l = self.legal(gamma,Zc)
                print(l)
                for i in l[1][1]:
                    before = CIN - sum(gamma[:,i])
                    #gamma[L[:,i]>=Zc,i] = Gamma[L[:,i]>=Zc,i]
                    self.reinstate_col_row(gamma,Gamma,L,Zc, i)
                    after = CIN - sum(gamma[:,i])
                    count += before - after
                    legal=False

                ## No ZERO col ( not output channel pruning)
                for i in r[1][0]:
                    before = COUT - sum(gamma[i,:])
                    #gamma[i,QQ[i,:]>=Z1r[i]] = Gamma[i,QQ[i,:]>=Z1r[i]]
                    self.reinstate_row_row(gamma,Gamma, QQ,Z1r, i)
                    after = COUT - sum(gamma[i,:])
                    count += before - after
                    legal=False

                legal = l[0]
                if legal :
                    self.gamma.assign(gamma)
                    l = self.legal(gamma,Zc)
                    
                    if l[0] is False:
                        print(l)
                        import pdb; pdb.set_trace()
                    return "Min Global %d %d " % (CIN*COUT - sum(sum(gamma)),Z )
                else:
                    #import pdb; pdb.set_trace()
                    NZG1 = CIN*COUT - sum(sum(gamma))
                    if NZG1 - NZG> math.ceil(K/2):
                        ## I could introduce 1+ zeros .. good enough
                        l = self.legal(gamma,Zc)
                        
                        if l[0] is False:
                            print(l)
                            import pdb; pdb.set_trace()
                        self.gamma.assign(gamma)
                        return "Min Global adjusted %d %d %d instead of %d we add %d" % (
                            CIN*COUT - sum(sum(gamma)),Z,count,K,  NZG1 - NZG
                        )
                    else:
                        cc = count
                        ## K is the increment we would like to achieve
                        ## randomly we introduce the least zeros per column 
                        for i in np.random.permutation(COUT):
                            if count >0 :
                                before = CIN - sum(gamma[:,i])
                                gamma[L[:,i]<Z1c[i],i] =0
                                after = CIN - sum(gamma[:,i])
                                count -= after-before
                        c= count
                        
                        Tamma = gamma * 1

                        count = 0
                        l = self.legal(gamma,Zc)
                        legal = l[0]
                        while count <4  and not legal:
                            nz = self.set_row(gamma,Z1r,QQ,L,Z1c)
                            l = self.legal(gamma,Zc)
                            print(nz,l)
                            count +=1
                            legal = l[0]

                        
                        self.gamma.assign(gamma)
                        if legal :
                            return "Min by column %d %d " % (CIN*COUT - sum(sum(gamma)),Z )
                        else:
                            return "Min by column %d %d adjusted %d %d %d" % (
                                CIN*COUT - sum(sum(gamma)),Z, cc,c, count
                            )
                ## choice by column


            return "No Zeros %d %d " %  (NZG, Z)
        return "Nothing to do"
    ###
    ##  This is the main function to zero at least one volume of the
    ##  kernel weights. Once we zero the gamma, they stay gamma, You
    ##  can use a differefnt volume measure.
    ##
    ###
    def zeros_volumes_per_col(self,
                              sparse_rate= 0.5,
                              by_lambda = True,
                              verbose = True,
                              step =20):


        zero_rate = 1 - sparse_rate

        count = 0
        
        K = self.kernel**2  if GradientGradient else  self.kernel
        AA = (np.max(np.finfo(np.float32).eps+ tf.abs(self.get_gradient()))*K*self.get_gamma()).numpy()
        
        if by_lambda:
            L   = self.get_gamma_l1_order()
        else:
            L,TT = self.get_gamma_w_cnout_2(
                AA,
                #self.volume_by_variance
                #self.volume_by_l1
                self.volume_by_l1
            )
            
        #

        
        tra = True or (self.kernel.shape[0]>1 or self.kernel.shape[1]>1)
        
        if tra and len(L.shape)==2 and L.shape[0]>1 and L.shape[1]>1:
            ## Gamma :  cin x cout 
            
            CIN, COUT = L.shape
            #print("L", CIN, COUT)
            ## how many zero we would like to have
            Zr = int(zero_rate*COUT)    ## per row
            Zc = int(zero_rate*CIN)     ## per col 
            Z = int(zero_rate*COUT*CIN) ## per all

            
            Gamma = self.gamma.numpy().astype(int)
            #print("GAMMA", Gamma.shape)
            Gammat = np.transpose(Gamma)
            # number of zero per row
            GZc   = CIN -sum(Gamma)
            # and per column
            GZr   = COUT - sum(Gammat)
            ## total number of zeros
            NZG = CIN*COUT - sum(sum(Gamma))
            #print("GZc", GZc.shape)
            #print("GZr",GZr.shape)
            
            ## either increment of 1 or step% of what we need, this is
            ## the number of zeros increment
            K = max(1,int(step*Z/100))
            # but no more that the one we need 
            Z1 = min(NZG+K, Z)

#            print(K,Z)
#            import pdb; pdb.set_trace()
            
            # per dimensions (we are after the columns x output
            Kc = max(1,int(step*Zc/100))
            Kr = max(1,int(step*Zr/100))
            # but no more that the one we need 
            Z1c = np.minimum(GZc+Kc, Zc)
            Z1r = np.minimum(GZr+Kr, Zr)



            
            QQ = TT*1
            for j in range(COUT):
                S = np.argsort(TT[:,j])
                for i in range(CIN):
                    QQ[S[i],j] = i
            
            if NZG < Z :
                #import pdb; pdb.set_trace()
                
                # we still have zeros to give, we use the minimum set
                # if legal
                gamma = Gamma*1
                print(self.legal_p(gamma,Zr))
                gamma[TT<Z1] =0
                l = self.legal_p(gamma,Zr)
                print(l)

                
                count = 0
                ## not too much CIN pruning 
                for i in l[1][1]:
                    before = COUT-sum(gamma[i,:])
                    #gamma[i,L[i,:]>=Z1r[i]] = Gamma[i, L[i,:]>=Z1r[i]]
                    self.reinstate_row_col(gamma,Gamma,L,Z1r, i)
                    after = COUT-sum(gamma[i,:])
                    count += before - after
                    
                ## No ZERO col ( not output channel pruning)
                for i in l[1][0]:
                    before = CIN - sum(gamma[:,i])
                    # gamma[QQ[:,i]>=Z1c[i],i] = Gamma[QQ[:,i]>=Z1c[i],i]
                    self.reinstate_col_col(gamma,Gamma, QQ,Z1c, i)
                    after = CIN - sum(gamma[:,i])
                    count += before - after

                legal = l[0]    
                
                if legal :
                    self.gamma.assign(gamma)
                    l = self.legal_p(gamma,Zr)
                    
                    if l[0] is False:
                        print(l)
                        import pdb; pdb.set_trace()
                        self.legal_p(gamma,Zr)
                        
                    return "Min Global %d %d " % (CIN*COUT - sum(sum(gamma)),Z )
                else:
                    
                    NZG1 = CIN*COUT - sum(sum(gamma))
                    if NZG1 - NZG> math.ceil(K/2):
                        ## I could introduce 1+ zeros .. good enough
                        l = self.legal_p(gamma,Zr)
                        
                        if l[0] is False:
                            print(l)
                            import pdb; pdb.set_trace()
                            self.set_col(gamma,Z1c,QQ,L,Z1r)
                            print(self.legal_p(gamma,Zr))
                            
                        self.gamma.assign(gamma)
                        return "Min Global adjusted %d %d %d instead of %d we add %d" % (
                            CIN*COUT - sum(sum(gamma)),Z,count,K,  NZG1 - NZG
                        )
                    else:
                        Z0 = CIN*COUT - sum(sum(gamma))
                        print("Zeros", Z0)
                        cc = count
                        ## K is the increment we would like to achieve
                        ## randomly we introduce the least zeros per column 
                        for i in np.random.permutation(CIN):
                            if count >0 :
                                before = COUT - sum(gamma[i,:])
                                try:
                                    gamma[i,L[i,:]<Z1r[i]] =0
                                except Exception as e :
                                    print(e)
                                    import pdb; pdb.set_trace()
                                after = COUT - sum(gamma[i,:])
                                count -= after-before
                        c= count
                        
                        count = 0
                        #
                        l = self.legal_p(gamma,Zr)
                        legal = l[0]
                        nz=0
                        Z1 = CIN*COUT - sum(sum(gamma))
                        print("ZERO PERM", Z1,l)
                        while count <4  and not legal:
                            #import pdb; pdb.set_trace()
                            #print(COUT,len(Z1c), Z1c)
                            nz += self.set_col(gamma,Z1c,QQ,L,Z1r)
                            l = self.legal_p(gamma,Zr)
                            Z2 = CIN*COUT - sum(sum(gamma))
                            print("Circle", nz,Z2,l)
                            count += 1
                            legal = l[0]

                        
                                               
                        if legal :
                            self.gamma.assign(gamma)
                            S =  "JUCE Min by col %d instead %d " % (CIN*COUT - sum(sum(gamma)),Z )
                        else:
                            import pdb; pdb.set_trace()
                            S =  "Min by col %d %d adjusted %d %d %d %d" % (
                                CIN*COUT - sum(sum(gamma)),Z, cc,c, count, nz
                            )
                            
                        
                        return S
                ## choice by column


            return "No Zeros %d %d " %  (NZG, Z)
        return "Nothing to do"
    ###
    ##  This is the main function to zero at least one volume of the
    ##  kernel weights. Once we zero the gamma, they stay gamma, You
    ##  can use a differefnt volume measure.
    ##
    ###
    def zeros_volumes_per_block(
            self,
            sparse_rate= 0.5,
            verbose = True,
            step =20,
            blocks = [ 4, 2]
            
    ):


        zero_rate = 1 - sparse_rate

        count = 0

        
        L,TT = self.get_gamma_w_cnout(
            (self.get_gradient()*self.kernel*self.get_gamma()).numpy(),
            self.volume_by_variance
        )

        
        tra = True or (self.kernel.shape[0]>1 or self.kernel.shape[1]>1)
        
        if tra and len(L.shape)==2 and L.shape[0]>1 and L.shape[1]>1:
            ## Gamma :  cin x cout 
            
            CIN, COUT = L.shape
            #print("L", CIN, COUT)
            ## how many zero we would like to have
            Zr = int(zero_rate*COUT)    ## per row
            Zc = int(zero_rate*CIN)     ## per col 
            Z = int(zero_rate*COUT*CIN) ## per all
            ZBound = int(zero_rate*COUT*CIN) ## per block

            Gamma = self.gamma.numpy().astype(int)
            
            ## per each block
            Zbound = numpy.zeros(blocks[0]*blocks[1]).reshape(blocks)
            Zblock = numpy.zeros(blocks[0]*blocks[1]).reshape(blocks)
            for i in range(blocks[0]):
                AI = min(CIN,(i+1)*math.ceil(CIN/blocks[0]))
                for j in range(blocks[1]):
                    BI = min(CIN,(j+1)*math.ceil(COUT/blocks[1]))

                    Zbound[i,j] = math.ceil(zero_rate*COUT/j*CIN/i)
                    Zblock[i,j] = math.ceil(COUT/j*CIN/i) -\
                                  sum(Gamma[i*math.ceil(CIN/blocks[0]):AI,
                                            j*math.ceil(COUT/blocks[1]):BI])
                    
            
            Gamma = self.gamma.numpy().astype(int)
            #print("GAMMA", Gamma.shape)
            Gammat = np.transpose(Gamma)
            # number of zero per row
            GZc   = CIN -sum(Gamma)
            # and per column
            GZr   = COUT - sum(Gammat)
            ## total number of zeros
            NZG = CIN*COUT - sum(sum(Gamma))
            #print("GZc", GZc.shape)
            #print("GZr",GZr.shape)
            
            ## either increment of 1 or step% of what we need, this is
            ## the number of zeros increment
            K = max(1,int(step*Z/100))
            # but no more that the one we need 
            Z1 = min(NZG+K, Z)

#            print(K,Z)
#            import pdb; pdb.set_trace()
            
            # per dimensions (we are after the columns x output
            Kc = max(1,int(step*Zc/100))
            Kr = max(1,int(step*Zr/100))
            Kb = max(1,int(step*Zbound/100))
            
            # but no more that the one we need 
            Z1c = np.minimum(GZc+Kc, Zc)
            Z1r = np.minimum(GZr+Kr, Zr)
            Z1b = np.minimum(Zblock+Kb, Zbound)

            
            if NZG < Z :
                
                # we still have zeros to give, we use the minimum set
                # if legal
                gamma = Gamma*1
                print(self.legal_p(gamma,Zr))
                for i in range(blocks[0]):
                    AI = min(CIN,(i+1)*math.ceil(CIN/blocks[0]))
                    for j in range(blocks[1]):
                        BI = min(CIN,(j+1)*math.ceil(COUT/blocks[1]))
                        
                        gamma[L[i,j]<Z1b[i,j]] = 0

                legal  = True
                count = 0
                for i in range(CIN):
                    if sum(gamma[i,:])==0:
                        index = TT.index(max(TT[i,:]))
                        gamma[i,index] =1
                        count+=1
                for i in range(COUT):
                    if sum(gamma[:,i])==0:
                        index = TT.index(max(TT[:,i]))
                        gamma[i,index] =1
                        count+=1
                self.gamma.assign(gamma)
                S =  "Min by block %d %d adjusted %d" % (
                    CIN*COUT - sum(sum(gamma)),Z, count
                )
                                    
                return S
                ## choice by column


            return "No Zeros %d %d " %  (NZG, Z)
        return "Nothing to do"
    def comp_volumes_overall(self):

        funcs = [
            #self.volume_by_determinant,
            ("euclidian", self.volume_by_euclidian),
            ("variance"  , self.volume_by_variance),
            ("l1"        , self.volume_by_l1)#,
            #("mean"     , self.volume_by_mean)
        ]
        
        
        L   = self.get_gamma_l1_order_all()
        #print(L)
        LWs = []
        
        for n,f in funcs:
            
            LW,_  = self.get_gamma_w_cnout(
                self.kernel.numpy(),
                f
            )
            LLW,_ = self.get_gamma_w_cnout(
                (self.get_gamma()*self.kernel).numpy(),
                f
            )
            LWs.append(LW)
            LWs.append(LLW)
            rhos = stats.kendalltau(
                L.flatten(),
                LW.flatten()
            )
            if False:    
                print(n,"W")
                print(rhos.correlation)
            
            rhos = stats.kendalltau(
                L.flatten(),
                LLW.flatten()
            )
            print(n,"LW")
            print(rhos.correlation)

        #import pdb; pdb.set_trace()
        

if __name__ == "__main__":

    # This returns a tensor
    #import pdb; pdb.set_trace()
    input_shape = (224,224,4)
    inputs_train = np.array([tf.random.normal(input_shape) for i in range(10000)])
    

    inputs = keras.layers.Input(shape=(224,224,4))
    X = SparseBlockConv2d(
        block_dim = (2,2),
        filters = 4,
        kernel_size = (1,1),
        activation = 'relu',
        padding ='same',
        #input_shape = input_shape
    )(inputs)
    



    print_weights = LambdaCallback(on_epoch_end=lambda batch, logs: print([ l.get_weights() for l in model2.layers]))


    model2 = keras.models.Model(inputs= inputs, outputs = X)
    model2.compile(optimizer='adam',loss='mean_squared_error')
    model2.fit(inputs_train,inputs_train,
               epochs=50,
               validation_split=0.1,shuffle=False,
               callbacks = [], #[print_weights]
    )
    import pdb; pdb.set_trace()
    print([ l.get_weights() for l in model2.layers])
