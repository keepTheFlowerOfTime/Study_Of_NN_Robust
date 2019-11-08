import matplotlib.pyplot as plt

def image_view(*kargs):
    """
    pic: a picture which has shape (width,height,channel) where channel always equal 1 or 3
    """
    number=0
    images=list(kargs)
    for d in images:
        if len(d.shape)==3:
            number+=1
        else:
            number+=d.shape[0]

    index=0
    for pic in images:
        if len(pic.shape)==3:
            plt.subplot(1,number,index+1)
            plt.imshow(pic)
            index+=1
        else:
            for b in pic:
                plt.subplot(1,number,index+1)
                plt.imshow(b)
                index+=1
    plt.axis('off')
    plt.show()