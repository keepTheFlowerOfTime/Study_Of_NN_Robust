from keras import Model
import numpy as np

def start_with(content,words):
    flags=[content.startswith(w) for w in words]
    flags=np.array(flags)

    r=np.sum(flags)
    return False if r==0 else True  

def copy_from(origin:Model,target:Model,need_layer=['conv2d'],fixed=False):
    for layer in target.layers:
        layer_name=layer.name
        if not start_with(layer_name,need_layer):
            continue
        try:
            origin_layer=origin.get_layer(name=layer_name)
            origin_layer.set_weights(layer.get_weights())
            if fixed:
                origin_layer.trainable=False
        except ValueError as identifier:
            pass
        
