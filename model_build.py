import tensorflow as tf
from keras.optimizers import SGD
from data_set import CifarFix
from helper.preprocess import stand_out_value
from STModel import STModelFactory
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

train_compare()

