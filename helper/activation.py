import numpy as np

def logistic(a,lower_a=None,upper_a=None,lower_expect=None,upper_expect=None):
    """
    function y=1/(1+exp(-x))
    lower_a,upper_a: domain of x
    lower_expect,upper_expect:the domain which scale
    """
    def _logistic(a):
        return 1/(1+np.exp(-a))

    #return _logistic(a)

    need_scale=False

    if lower_expect!=None and upper_expect!=None:
        need_scale=True

    if not need_scale:
        return _logistic(a)
    else:
        if lower_a is None: lower_a=a.min()
        if upper_a is None: upper_a=a.max()

        assert upper_a>=lower_a

        scale_ratio=(upper_expect-lower_expect)/(upper_a-lower_a)

    scale_a=(a-lower_a)*scale_ratio+lower_expect

    return _logistic(scale_a)
