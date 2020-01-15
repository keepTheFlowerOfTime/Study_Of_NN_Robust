from keras.layers import Layer
import keras.backend as K
import tensorflow as tf
from helper.keras_op import stand_out_value,step_by_step,to_one
class FeatureExtractLayer(Layer):
    def __init__(self,**kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        kwargs['trainable']=False
        super(FeatureExtractLayer,self).__init__(**kwargs)


    def call(self,inputs):
        output=stand_out_value(inputs)
        
        return output

    def compute_output_shape(self,input_shape):
        return input_shape

class FocusLayer(Layer):
    def __init__(self,channel,lower_bound=0,upper_bound=1.,**kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        self.channel=channel
        self.lower_bound=lower_bound
        self.upper_bound=upper_bound
        super(FocusLayer,self).__init__(**kwargs)

    def call(self,input):
        return step_by_step(input,0,1.,16)

    
    def compute_output_shape(self,input_shape):
        output_shape=list(input_shape)
        output_shape[-1]=output_shape[-1]*self.channel
        return tuple(output_shape)

class ToOneLayer(Layer):
    def __init__(self,axis,**kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        self.axis=axis
        super(ToOneLayer,self).__init__(**kwargs)

    def call(self,input):
        return [to_one(x,self.axis) for x in input]

    
    def compute_output_shape(self,input_shape):
        return input_shape

class DropFeature(Layer):
    def __init__(self,prob,**kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        self.prob=prob
        super(DropFeature,self).__init__(**kwargs)

    def build(self,input_shape):
        
        self.previous_input_shape=[input_shape[0],input_shape[1]]
        self.built = True

    def call(self,input):
        #v=tf.zeros_like(input)
        mask=K.random_uniform(K.shape(input),0.,1.)
        return tf.where(mask>self.prob,input,0)
        #return drop(input,self.prob)

    
    def compute_output_shape(self,input_shape):
        return input_shape
