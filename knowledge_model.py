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

    def call(self,inputs,status):
        if self.handle is not None:
            return self.handle(inputs,status)
        else:
            return inputs

    def __call__(self,x):
        args=x
        inputs=args[0]
        status=KnowledgeModel.Status_Feature_Train
        if len(x)>1:
            status=args[1]

        return self.call(inputs,status)

def k_module(handle):
    m=KnowledgeModule()
    m.handle=handle
    return m


class KnowledgeModel:
    Status_Feature_Train=0
    Status_Decision_Train=1
    Status_Test=2
    """
    each part of model should inherent from KnowledgeModule,otherwise it may cause some trouble
    """
    def __init__(self,input_shape,feature_extract_models,mix_function,final_decision):
        self.input=Input(shape=input_shape)
        self.feature_extract_models=feature_extract_models
        self.mix_function=mix_function
        self.final_decision=final_decision

    def build(self,restore=None,status=0):
        model_input=self.input
        feature_extracts=self.feature_extract_models
        mix_function=self.mix_function
        decision_model=self.final_decision
        features=[m([model_input,status]) for m in feature_extracts]

        decision_input=mix_function([features,status])
        result=decision_model([decision_input,status])
        
        m=Model(inputs=model_input,output=result)
        if restore is not None:
            m.load_weights(restore)
        return m