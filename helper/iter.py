import numpy as np
import keras.backend as K

def for_each_class(data,labels,func,class_number=10):
    """
    func(class_index,data,previous_return)
    """
    b=data.shape[0]
    class_temp=[None]*class_number
    for i in range(b):
        class_index=np.argmax(labels[i])
        r=func(class_index,data[i],class_temp[class_index])
        class_temp[class_index]=r
    
    return class_temp

def for_each_batch(data,max_batch=3000):
    """
    see the first dimension of data as batch dimension.
    This function will split a big batch to some smaller batch.
    """
    b=data.shape[0]
    if b<=max_batch:
        yield data
    else:
        number=b//max_batch
        if b%max_batch!=0: number+=1
        batches=np.split(data,number,0)
        for e in batches:
            yield e


    
        