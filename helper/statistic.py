import numpy as np
import collections

def uniformlize(data,block_number=10):
    shape=np.shape(data)
    if len(shape)>1:
        data=np.reshape(data,[-1])
    
    max_v=np.max(data)
    min_v=np.min(data)

    level_req=(max_v-min_v)/block_number

    data-=min_v
    
    data/=level_req
    data-=0.001
    return np.array(data,dtype=np.int32)

def counter(sample):
    counter=collections.Counter(sample)
    keys=list(counter.keys())
    keys.sort()
    return keys,[counter[k] for k in keys]