import tensorflow as tf
import time
from STModel import STModelFactory
from nn_robust_attacks.setup_cifar import CIFAR,CIFARModel
from nn_robust_attacks.test_attack import generate_data
from nn_robust_attacks.l2_attack import CarliniL2
from data_set import CifarFix
import numpy as np

def test_default(data):
    ps=[]
    ds=[]
    with tf.Session() as sess:
        model=CIFARModel("models/cifar", sess)
        attack = CarliniL2(sess, model, batch_size=9, max_iterations=1000, confidence=0)
        inputs, targets = generate_data(data, samples=1, targeted=True,
                                    start=0, inception=False)
        timestart = time.time()
        adv = attack.attack(inputs, targets)
        timeend = time.time()
        
        print("Took",timeend-timestart,"seconds to run",len(inputs),"samples.")


        for i in range(len(adv)):
            #print("Valid:")
            #show(inputs[i])
            #print("Adversarial:")
            #show(adv[i])
            p= np.argmax(model.model.predict(adv[i:i+1]),axis=1)
            d=np.sum((adv[i]-inputs[i])**2)**.5
            ps.append(p)
            ds.append(d)

        return ps,ds

def test_exp(data,path):
    import helper.preprocess
    ps=[]
    ds=[]
    tool=STModelFactory()
    with tf.compat.v1.Session() as sess:
        model=tool.get_combine_model(path,False)
        attack = CarliniL2(sess, model, batch_size=9, max_iterations=1000, confidence=0)
        inputs, targets = generate_data(data, samples=1, targeted=True,
                                    start=4, inception=False)
        timestart = time.time()
        adv = attack.attack(inputs, targets)
        timeend = time.time()
        
        print("Took",timeend-timestart,"seconds to run",len(inputs),"samples.")

        all_predict=model.model.predict([adv,helper.preprocess.stand_out_value(adv)])
        ps=np.argmax(all_predict,axis=1)
        ds=[]

        for i in range(9):
            d=np.sqrt(np.sum((adv[i]-inputs[i])**2)) 
            ds.append(d)

    return ps,ds
data=CIFAR()
#p,d=test_default(data)
p,d=test_exp(data,'../models/cifar_combine')
for i in range(len(p)):
    print(p[i],d[i])
#test_exp(data,'models/cifar_stand_out')

# normal test result
# 0 0.6948183567096975
# 1 0.47551249714434285
# 2 0.1643397532771848
# 3 0.19788176864535043
# 4 0.19696040854856967
# 5 0.4930673788947916
# 7 0.6769811441913917
# 8 0.47921493976963003
# 9 0.45051834127885787