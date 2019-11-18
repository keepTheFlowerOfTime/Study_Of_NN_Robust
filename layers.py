from keras.layers import Layer
import keras.backend as K
from helper.keras_op import stand_out_value,step_by_step
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
