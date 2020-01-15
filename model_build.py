import tensorflow as tf
from keras.optimizers import SGD,Adam
from data_set import CifarFix
from helper.preprocess import stand_out_value
from helper.copy import copy_from
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

from STModel import STModelFactory
from resnet import ResNet,KModel_ResNet
import numpy as np
def train_ST(data,model,file_name,num_epochs=50,batch_size=128,train_temp=1,init=None):
    def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted/train_temp)

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True
    )
    model=model.model
    model.compile(loss=fn,
                  optimizer=sgd,
                  metrics=['accuracy'])
    
    model.fit(data.train_data, data.train_labels,
              batch_size=batch_size,
              validation_data=(data.validation_data, data.validation_labels),
              nb_epoch=num_epochs,
              shuffle=True)

    if file_name != None:
        model.save(file_name)

    return model

def train(train_data,train_labels,validation_data,validation_labels,model,file_name,num_epochs=50,batch_size=128,train_temp=1,init=None):
    def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted/train_temp)
    
    sgd=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True
    )

    model.compile(loss=fn,optimizer=sgd,metrics=['accuracy'])

    model.fit(train_data,train_labels,batch_size=batch_size,validation_data=(validation_data,validation_labels),nb_epoch=num_epochs,shuffle=True)

    if file_name is not None:
        model.save(file_name)

    return model

def train_stand_out():
    factory=STModelFactory()
    data=CifarFix(need_verify_data=False)
    factory.preprocess(data)
    path='../models/cifar_stand_out'

    train_ST(data,factory.get_train_model(),path)

def train_craft(is_test=True):
    factory=STModelFactory()
    data=CifarFix(need_verify_data=False)
    if is_test:
        sample_number=5000
        #data.validation_data=data.validation_data[:sample_number]
        data.train_data=data.train_data[:sample_number]
        #data.validation_labels=data.validation_labels[:sample_number]
        data.train_labels=data.train_labels[:sample_number]
    factory.preprocess(data)
    path='models/cifar_craft'

    train_ST(data,factory.get_craft_model(is_train=True),path)

def train_knowledge(is_test=True):
    factory=STModelFactory()
    data=CifarFix(need_verify_data=False)
    knowledge_train=stand_out_value(data.train_data)
    knowledge_validation=stand_out_value(data.validation_data)

    model=factory.get_combine_model(is_train=True).model
    train_data=[data.train_data,knowledge_train]
    train_labels=data.train_labels

    validation_data=[data.validation_data,knowledge_validation]
    validation_labels=data.validation_labels

    train(train_data,train_labels,validation_data,validation_labels,model,'../models/cifar_combine')

def train_compare(is_test=True):
    factory=STModelFactory()
    data=CifarFix(need_verify_data=False)
    train_ST(data,factory.compare_model(is_train=True),'../models/cifar_combine_compare')

def train_resnet(path,data,model,data_augmentation=True):
    # Training parameters
    batch_size = 128  # orig paper trained all networks with batch_size=128
    epochs = 200
    #data_augmentation = True
    num_classes = 10
    #path='../models/cifar_resnet_knowledge'
    #data=CifarFix(train_mode='o',need_verify_data=False)
    #model=ResNet().get_knowledge_model()
    model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=ResNet.lr_schedule(0)),
              metrics=['acc'])
    model.summary()

    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=path,
                                monitor='val_acc',
                                verbose=1,
                                save_best_only=True)

    lr_scheduler = LearningRateScheduler(ResNet.lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                cooldown=0,
                                patience=5,
                                min_lr=0.5e-6)

    callbacks = [checkpoint, lr_reducer, lr_scheduler]

    # Run training, with or without data augmentation.
    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(data.train_data, data.train_labels,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(data.validation_data, data.validation_labels),
                shuffle=True,
                callbacks=callbacks)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            # set input mean to 0 over the dataset
            featurewise_center=False,
            # set each sample mean to 0
            samplewise_center=False,
            # divide inputs by std of dataset
            featurewise_std_normalization=False,
            # divide each input by its std
            samplewise_std_normalization=False,
            # apply ZCA whitening
            zca_whitening=False,
            # epsilon for ZCA whitening
            zca_epsilon=1e-06,
            # randomly rotate images in the range (deg 0 to 180)
            rotation_range=0,
            # randomly shift images horizontally
            width_shift_range=0.1,
            # randomly shift images vertically
            height_shift_range=0.1,
            # set range for random shear
            shear_range=0.,
            # set range for random zoom
            zoom_range=0.,
            # set range for random channel shifts
            channel_shift_range=0.,
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            # value used for fill_mode = "constant"
            cval=0.,
            # randomly flip images
            horizontal_flip=True,
            # randomly flip images
            vertical_flip=False,
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)

        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(data.train_data)

        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(data.train_data, data.train_labels, batch_size=batch_size),
                            validation_data=(data.validation_data, data.validation_labels),
                            epochs=epochs, verbose=1, workers=4,
                            callbacks=callbacks)

data=CifarFix('o',need_verify_data=False)
#ResNet().preprocess(data)
base_model=ResNet().get_model(None,0.5,True)
#k_model=KModel_ResNet().get_model(status=1)
#copy_from(k_model,base_model,fixed=True)
train_resnet('../models/cifar_resnet_with_drop_feature',data,base_model)

