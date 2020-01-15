import tensorflow as tf
import nn_robust_attacks.setup_cifar as cifar
import nn_robust_attacks.train_models as train
from helper.keras_op import stand_out_value
from keras.models import Sequential,Model
from keras.layers import Conv2D,Flatten,Dense,Activation,MaxPooling2D,ReLU,InputLayer,Dropout,Multiply
import keras.backend as k
from data_set import CifarFix

from layers import FeatureExtractLayer,FocusLayer,DropFeature

import numpy as np
class STModelFactory:
    def __init__(self):
        pass


    # def preprocess(self,data:CifarFix):
    #     if data.__dict__.get('train_data') is not None:
    #         data.train_data=stand_out_value(data.train_data)
    #     if data.__dict__.get('test_data') is not None:
    #         data.test_data=stand_out_value(data.test_data)
    #     if data.__dict__.get('validation_data') is not None:
    #         data.validation_data=stand_out_value(data.validation_data)

    def adjust_windows(self,size=None,data:CifarFix=None):
        def _adjust_windows(data:CifarFix):
            train_data=data.train_data

        if size is None:
            size=_adjust_windows(data)

        self.size=size

    
    def get_predict_model(self,restore=None):
        m=self.trainable_layer()
        if restore is not None:
            m.load_weights(restore)
        
        m=self.add_robust_layer(m)

        return ModelAdapter(m)

    def get_train_model(self):
        m=STModelFactory.default_train_cnn()
        return ModelAdapter(m)

    def get_craft_model(self,restore=None,is_train=False):
        m=STModelFactory.hand_craft_model(is_train)
        if not is_train and restore is not None:
            m.load_weights(restore)
        
        return ModelAdapter(m)

    def get_combine_model(self,restore=None,is_train=False):
        origin=STModelFactory.fix_cnn_model(True)
        knowledge=STModelFactory.fix_cnn_model(True)
        dot_result=Multiply()([origin.output,knowledge.output])
        dense_1=Dense(256)(dot_result)
        dense_1=ReLU()(dense_1)
        if is_train:
            dense_1=Dropout(0.5)(dense_1)
        dense_1=Dense(256)(dense_1)
        dense_1=ReLU()(dense_1)
        dense_1=Dense(10)(dense_1)
        
        ret=Model(inputs=[origin.input,knowledge.input],outputs=dense_1)
        if restore is not None:
            ret.load_weights(restore)

        return KModel(ret)
            
    def add_robust_layer(self,out):
        m=Sequential()
        
        for i in range(len(out.layers)):
            if(i==0):
                m.add(FeatureExtractLayer())
            m.add(out.layers[i])
        return m

    def trainable_layer(self):
        return STModelFactory.default_predict_cnn()

    def compare_model(self,is_train=False,restore=None,sess=None,drop_feature=0):
        model=STModelFactory.compare_cnn_model(is_train,drop_feature)
        if restore is not None:
            model.load_weights(restore)
        return ModelAdapter(model,sess)

    @staticmethod
    def compare_cnn_model(is_train=False,drop_feature=0):
        model=Sequential()
        model.add(InputLayer(input_shape=(32,32,3)))
        model.add(Conv2D(64, (3, 3),padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Conv2D(128, (3, 3),padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(128, (3, 3),padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Flatten())
        if drop_feature>0:
            model.add(DropFeature(drop_feature))
        model.add(Dense(256))
        model.add(Activation('relu'))
        if is_train:
            model.add(Dropout(0.5))
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(10))
        return model

    @staticmethod
    def fix_cnn_model(is_train=False):
        model=Sequential()
        model.add(InputLayer(input_shape=(32,32,3)))
        model.add(Conv2D(64, (3, 3),padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Conv2D(128, (3, 3),padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(128, (3, 3),padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Flatten())
        
        return model

    @staticmethod
    def hand_craft_model(flag_train=False):
        model=Sequential()
        model.add(InputLayer(input_shape=(32,32,3)))
        if not flag_train:
            model.add(FeatureExtractLayer())
        model.add(FocusLayer(16))
        model.add(Conv2D(32,(1,1),use_bias=False))
        model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(32,(1,1),use_bias=False))
        model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
        model.add(Activation('relu'))

        model.add(Flatten())
        model.add(Dense(128,use_bias=False))
        model.add(Activation('relu'))
        if flag_train:
            model.add(Dropout(0.5))
        model.add(Dense(128,use_bias=False))
        model.add(Activation('relu'))
        model.add(Dense(10,use_bias=False))
        return model
    
    @staticmethod
    def default_cnn(is_train=True):
        model=Sequential()
        model.add(InputLayer(input_shape=(32,32,3)))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Conv2D(128, (3, 3)))
        model.add(Activation('relu'))
        model.add(Conv2D(128, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation('relu'))
        if is_train:
            model.add(Dropout(0.5))
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(10))
        return model

class KModel:
    def __init__(self,model,sess=None):
        self.num_channels = 3
        self.image_size = 32
        self.num_labels = 10
        self.model=model

    def predict(self, data):
        return self.model([data,stand_out_value(data)])

class ModelAdapter:
    def __init__(self,model,sess=None):
        self.num_channels = 3
        self.image_size = 32
        self.num_labels = 10
        self.model=model

    def predict(self, data):
        return self.model(data)
    
## first model:0.62

#data=cifar.CifarFix('n','n')
#train.train_ST(data,STModel.model('models/stable'),'models/st_model_2',num_epochs=50)
