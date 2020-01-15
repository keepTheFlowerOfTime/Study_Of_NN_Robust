import keras.backend as K
import numpy as np
import tensorflow as tf
import keras.layers as op


from helper.preprocess import get_weight
t=-1/9
StandOutWeight=get_weight(np.array([
            [t,t,t],
            [t,1+t,t],
            [t,t,t]
        ]))
#StandOutWeight=tf.Variable(StandOutWeight,dtype=tf.float32)

def logistic(a,lower_a=None,upper_a=None,lower_expect=None,upper_expect=None):
    need_scale=False

    if lower_expect!=None and upper_expect!=None:
        need_scale=True

    if not need_scale:
        return 1/(1+tf.exp(a))
    else:
        if lower_a is None: lower_a=a.min()
        if upper_a is None: upper_a=a.max()

        assert upper_a>=lower_a

        scale_ratio=(upper_expect-lower_expect)/(upper_a-lower_a)

    scale_a=(a-lower_a)*scale_ratio+lower_expect

    return 1/(1+tf.exp(-scale_a))

def to_one(data,axis):
    max_value=tf.reduce_max(data,axis)
    result=op.multiply([data,1/max_value])
    return result

def stand_out_value(data):
    result=tf.nn.conv2d(data,StandOutWeight,strides=[1,1,1,1],padding='SAME')
    #K.batch_set_value
    return K.abs(to_one(result,[1,2]))

def step_by_step(data,lower_bound,upper_bound,channel):
    """
    input_shape:b,w,h,c
    output_shape: (b,w,h,c*level_up_c)
    """
    step=(upper_bound-lower_bound)/channel
    
    result=[]
    lower_step=upper_bound-step
    upper_step=upper_bound
    for _ in range(channel):
        temp=tf.where((lower_step<=data)&(data<upper_step),data,0)
        result.append(temp)
        lower_step-=step
        upper_step-=step

    return K.concatenate(result,3)


def standardization(data):
    u,v=tf.nn.moments(data,0)
    std=tf.sqrt(v)

    return (data-u)/std


# def drop(data,shape,prob,default_value=0):
#     """
#     given a certain prob to drop the feature.Proposed to test the coherence of feature.
#     """
#     sample=K.random_uniform_variable(data.get_shape(),0,1)
#     #sample=tf.compat.v1.random_uniform(data.get_shape(),minval=0,maxval=1,dtype=tf.float32)
#     return tf.where(sample>prob,data,default_value)