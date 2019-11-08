import numpy as np
import nn_robust_attacks.setup_cifar as cifar
import skimage.io

from helper.preprocess import stand_out_value
from helper.viewer import image_view
def view_markov():
    t=-1/9
    weights=np.array([[t,t,t],
             [t,1+t,t],
             [t,t,t]])
    
    

    path='models/cifar'
    model=cifar.CIFARModel(path).model

    target_layer=model.layers[0]

    kernal,bias=target_layer.get_weights()

    w,h,c_i,c_o=kernal.shape

    kernal=kernal.transpose([3,2,0,1])
    for i in range(c_o):
        for j in range(c_i):    
            nn_weights=kernal[i][j]
            a=nn_weights[1][1]/weights[1][1]

            b=nn_weights-a*weights
            b=b.reshape(-1)
            nn_weights=nn_weights.reshape(-1)
            print(np.mean(b),np.mean(nn_weights))

def view_stand_out():
    path='tiger.jpeg'
    img=skimage.io.imread(path)
    w,h,c=img.shape
    img=img.reshape([1,w,h,c])
    sta_img=stand_out_value(img)
    image_view(img,sta_img)

def view_stand_out_cifar():
    data=cifar.CifarFix(need_verify_data=False)
    stand_out=stand_out_value(data.train_data[0:5])
    image_view(stand_out)
view_stand_out()
#view_markov()
