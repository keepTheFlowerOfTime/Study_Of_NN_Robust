from keras.layers import Input
from keras import Model
"""
Knowledge Model
Multi-Input
parallel Feature Extract
Mix Function
Final Decision 
"""

class KnowledgeModule:
    def __init__(self):
        self.handle=None

    def call(self,inputs,is_train):
        if self.handle is not None:
            return self.handle(inputs,is_train)
        else:
            return inputs

    def __call__(self,x):
        args=x
        inputs=args[0]
        is_train=False
        if len(x)>1:
            is_train=args[1]

        return self.call(inputs,is_train)

def k_module(handle):
    m=KnowledgeModule()
    m.handle=handle
    return m


class KnowledgeModel:
    """
    each part of model should inherent from KnowledgeModule,otherwise it may cause some trouble
    """
    def __init__(self,input_shape,feature_extract_models,mix_function,final_decision):
        self.input=Input(shape=input_shape)
        self.feature_extract_models=feature_extract_models
        self.mix_function=mix_function
        self.final_decision=final_decision

    def build(self,restore=None,is_train=False):
        model_input=self.input
        feature_extracts=self.feature_extract_models
        mix_function=self.mix_function
        decision_model=self.final_decision
        features=[m([model_input,is_train]) for m in feature_extracts]

        decision_input=mix_function([features,is_train])
        result=decision_model([decision_input,is_train])
        
        m=Model(inputs=model_input,output=result)
        if restore is not None:
            m.load_weights(restore)
        return m