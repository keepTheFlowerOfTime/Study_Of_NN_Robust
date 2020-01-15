"""
This file use to test the model which exp need.All the model was trained by CIFAR10 dataset.Notice,all data may or may not
be done something pretreatment.Besides,we didn't change the structure of model.

CIFAR10 dataset(60000) will spilt to three part.
Train_Data (45000)
Validation_Data (5000)
Test_Data (10000)
"""

import tensorflow.compat.v1 as tf

from nn_robust_attacks.setup_cifar import CIFARModel
from data_set import CifarFix

from helper.preprocess import stand_out_value,sample
from helper.copy import copy_from
import numpy as np
from STModel import STModelFactory,ModelAdapter
from resnet import ResNet
from helper.statistic import uniformlize,counter
from helper.viewer import rect_show
BATCH_SIZE=1

def test_with_handle(sess,example,handle=None):
    def default_handle(r:list,d):
        r.append(d)

    if handle is None:
        handle=default_handle

    data,model=example
    x = tf.placeholder(tf.float32, (None, model.image_size, model.image_size, model.num_channels))
    y=model.predict(x)
    r = []
    for i in range(0,len(data.test_data),BATCH_SIZE):
        pred = sess.run(y, {x: data.test_data[i:i+BATCH_SIZE]})
        handle(r,pred)
        #print(pred)
        #print('real',data.test_labels[i],'pred',np.argmax(pred))
        #r.append(np.argmax(pred,1) == np.argmax(data.test_labels[i:i+BATCH_SIZE],1))
    
    #print(np.mean(r))
    return np.array(r)

def test_with_result(sess,example):
    def handle(r,d):
        for i in range(BATCH_SIZE):
            r.append(d[i])

    return test_with_handle(sess,example,handle)

def test(sess,example):
    data,model=example
    x = tf.placeholder(tf.float32, (None, model.image_size, model.image_size, model.num_channels))
    y=model.predict(x)
    r = []
    for i in range(0,len(data.test_data),BATCH_SIZE):
        pred = sess.run(y, {x: data.test_data[i:i+BATCH_SIZE]})
        #print(pred)
        #print('real',data.test_labels[i],'pred',np.argmax(pred))
        r.append(np.argmax(pred,1) == np.argmax(data.test_labels[i:i+BATCH_SIZE],1))
    
    #print(np.mean(r))
    return np.array(r)

def test_twist(sess,data,model):
    x = tf.placeholder(tf.float32, (None, model.image_size, model.image_size, model.num_channels))
    y=model.predict(x)
    r = []
    for i in range(0,len(data),BATCH_SIZE):
        pred = sess.run(y, {x: data[i:i+BATCH_SIZE]})
        #print(pred)
        #print('real',data.test_labels[i],'pred',np.argmax(pred))
        r.append(pred)
    
    #print(np.mean(r))
    return np.concatenate(r,0)

#0.781
def normal_model_test(test_handle=test):
    """
    this test didn't do anything on CIFAR10 data set.We will see the result as base value.
    """
    model_path='../models/cifar'
    data=CifarFix(test_mode='n',need_train_data=False)
    

    r=None

    with tf.Session() as sess:
        model=CIFARModel(model_path,sess)
        r=test_handle(sess,(data,model))

    return r

#0.472
def shuffle_model_test():
    """
    operation shuffle only change the pos of pixel,not the value itself.

    all data will follow the same shuffle rule.It's means if we do shuffle operation
    for two same image,the result still same. 

    Test will return the precise ratio in Test_Data.
    """
    seed=5387319283
    model_path='models/cifar_shuffle_{}'.format(seed)
    data=CifarFix(train_mode='o',test_mode='sn',need_train_data=False,seed=seed)
    

    precise_ratio=None

    with tf.Session() as sess:
        model=CIFARModel(model_path,sess)
        precise_ratio=test(sess,(data,model))

    return precise_ratio


# mode 'b' 0.663
# mode 'n' 0.654
# mode 'b-fix' 0.661
def draft_model_test():
    """
    The main idea of this test is hold the most important info in image.That's means we will do something transform on origin image.
    We call the new image "draft".
    The generation of "draft" is according to the origin image's color gradient.We will erase some pixel where
    these color gradient too small. 
    """
    model_path='models/draft_b_fix'
    data=CifarFix(need_train_data=False,test_mode='d',args={'d':[.6,'b']})
    

    precise_ratio=None
    with tf.Session() as sess:
        model=CIFARModel(model_path,sess)
        precise_ratio=test(sess,(data,model))

    return precise_ratio

#0.7482
def stand_out_test():
    model_path='models/cifar_stand_out'
    data=CifarFix(need_train_data=False)

    data.test_data=stand_out_value(data.test_data)
    with tf.Session() as sess:
        model=CIFARModel(model_path,sess)
        precise_ratio=test(sess,(data,model))

    return precise_ratio


def new_model_test():
    model_path='models/cifar_stand_out'
    data=CifarFix(need_train_data=False)   
    factory=STModelFactory()
    with tf.Session() as sess:
        model=factory.get_predict_model(model_path)
        precise_ratio=test(sess,(data,model))
    
    return precise_ratio

def craft_model_test():
    model_path='models/cifar_craft'
    data=CifarFix(need_train_data=False)   
    factory=STModelFactory()
    with tf.Session() as sess:
        model=factory.get_craft_model(model_path,False)
        precise_ratio=test(sess,(data,model))
    return precise_ratio
def check_presoftmax():
    model_path='models/cifar_stand_out'
    data=CifarFix(need_train_data=False)
    factory=STModelFactory()
    a1=None
    a2=None
    
    with tf.Session() as sess:
        #model=factory.get_predict_model(model_path)
        model=CIFARModel('models/cifar')
        x = tf.placeholder(tf.float32, (None, model.image_size, model.image_size, model.num_channels))
        y=model.predict(x)
        #origin_model=CIFARModel('models/cifar')
        a1=sess.run(y,{x:data.test_data[0:1]})
        #a2=origin_model.predict(data.test_data[0:1])

    return a1,None#,a2


def twist_test():
    def add_twist(target,number):
        l=0.01
        batch,w,h,c=target.shape
        twist=np.random.rand(1,w,h,c)
        twist=sample(twist,0.2)
        twist*=l
        #twist-=l
        result=[]
        result.append(twist)
        for i in range(batch):
            for j in range(number):
                result.append((j+1)*twist+target[i:i+1])
        
        return np.concatenate(result,0)
    model_path='models/cifar'
    data=CifarFix(test_mode='n',need_verify_data=False)
    start=np.random.randint(0,100)
    number=1
    test_data=add_twist(data.train_data[start:start+number],15)

    result=None

    with tf.Session() as sess:
        model=CIFARModel(model_path,sess)
        model.model.pop()
        model.model.pop()
        result=test_twist(sess,test_data,model)

    result*=result[0]

    return np.mean(np.where(result>=0,1,0),axis=1)

def Model_test(data,model_handle,ratio,model=None,test_handle=test):

    with tf.Session() as sess:
        if model is None:
            model=model_handle(ratio)
        precise_ratio=test_handle(sess,(data,model))
    return precise_ratio
#print(np.mean(normal_model_test()))
#print(np.mean(shuffle_model_test()))
#print(np.mean(draft_model_test()))
#print(np.mean(stand_out_test()))
#print(np.mean(craft_model_test()))
#print(twist_test())
#print(a2)

# normal cnn
# [0.7897 0.7896 0.7838 0.7776 0.7765 0.7685 0.7662 0.7589 0.7549 0.7419
#  0.7319]

# res_net
# 0.9108 0.9083 0.9058 0.9003 0.8942 0.8817 0.8725 0.8611 0.8498 0.8256
#  0.8049 0.7772
data=CifarFix(test_mode='n',need_train_data=False)

r=normal_model_test(test_with_result)
r=uniformlize(r,20)
labels,numbers=counter(r)
rect_show(list(labels),list(numbers))
# r=[0,0.01]
# for i in range(1,11):
#     r.append(0.05*i)
# #handle=lambda a:STModelFactory().compare_model(drop_feature=a,restore='../models/cifar_combine_compare')
# handle=lambda a:ModelAdapter(ResNet().get_model('../models/cifar_resnet',a))
# result=[np.mean(Model_test(data,handle,ratio)) for ratio in r]
# print(np.array(result))

model_t=ResNet().get_model('../models/cifar_resnet',need_softmax=False)

#model_n,rt=ResNet().get_random_model(round_number=2)

#copy_from(rt,model_t)
#copy_from(model_n,model_t,['dense'])

#print(np.mean(Model_test(data,None,None,model=model_n)))
