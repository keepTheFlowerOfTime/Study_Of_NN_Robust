import numpy as np

from .activation import logistic
from .iter import for_each_batch,for_each_class

import keras.backend as K

def conv2d_im2col(Weight,X,stride=1,padding='same'):
    def im2col(img, ksize, stride=1):
        N, H, W, C = img.shape
        out_h = (H - ksize) // stride + 1
        out_w = (W - ksize) // stride + 1
        col = np.empty((N * out_h * out_w, ksize * ksize * C))
        outsize = out_w * out_h
        for y in range(out_h):
            y_min = y * stride
            y_max = y_min + ksize
            y_start = y * out_w
            for x in range(out_w):
                x_min = x * stride
                x_max = x_min + ksize
                col[y_start+x::outsize, :] = img[:, y_min:y_max, x_min:x_max, :].reshape(N, -1)
        return col
    Weight=np.transpose(Weight,[3,0,1,2])
    FN, ksize, ksize, C = Weight.shape
    if padding == 'same':
	    p = ksize // 2
	    X = np.pad(X, ((0, 0), (p, p), (p, p), (0, 0)), 'constant')
    N, H, W, C = X.shape
    col = im2col(X, ksize, stride)
    z = np.dot(col, Weight.reshape(Weight.shape[0], -1).transpose())
    z = z.reshape(N, int(z.shape[0] / N), -1)
    out_h = (H - ksize) // stride + 1
    return z.reshape(N, out_h, -1 , FN)

def get_weight(kernal,in_channel=3,out_channel=3):
    def forge_row(zero,value,length,index):
        r=[zero]*length
        r[index]=value
        return r
    zero=np.zeros_like(kernal)
    t=[forge_row(zero,kernal,in_channel,i) for i in range(out_channel)]
    w=np.array(t)
    return np.transpose(w,[2,3,1,0])

def color_compare(data,ignore_boundary=False,max_batch=3000):
    """
    data.shape=(b,w,h,c)
    """
    b,w,h,c=data.shape
    t=-1/9
    kernal=np.array([
            [t,t,t],
            [t,1+t,t],
            [t,t,t]
        ])
    weight=get_weight(kernal)
    r=np.concatenate([conv2d_im2col(weight,x) for x in for_each_batch(data,max_batch)],0)
    
    if ignore_boundary:
        r[:,0,:,:]=0
        r[:,:,0,:]=0
        r[:,w-1,:,:]=0
        r[:,:,h-1,:]=0

    return r

def stand_out_value(data):
    fake_gradient=color_compare(data).transpose([0,3,1,2])
    b,c,w,h=fake_gradient.shape
    largest_change=fake_gradient
    for i in range(2):
        largest_change=np.max(largest_change,axis=2)
    
    for i in range(b):
        for j in range(c):
            fake_gradient[i][j]/=largest_change[i][j]

    fake_gradient=fake_gradient.transpose([0,2,3,1])

    return logistic(fake_gradient,-1,1,-5,5)

def sample(data,ratio):
    shape=data.shape
    mask=np.random.random_sample(data.shape)
    mask=np.where(mask>ratio,0,1)
    
    return (mask.reshape(-1)*data.reshape(-1)).reshape(shape)
def standardization(data):
    u=np.average(data)
    std=np.std(data)

    return (data-u)/std
