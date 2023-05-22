import numpy as np
from Layers import Base

class SoftMax(Base.BaseLayer):

    def __init__(self):
        super().__init__()
        self.trainable = False

    def forward(self, input_tensor):    # softmax applied "row-wise" as described in pdf
        #print('input_tensor', input_tensor)
        input_tensor = input_tensor - np.amax(input_tensor)
        input_tensor = np.exp(input_tensor)
        sum = np.sum(input_tensor, axis=1)
        sum = sum[..., np.newaxis]
        return input_tensor / sum

    def backward(self, error_tensor):
        #print('error_tensor', error_tensor)
        soft = self.forward(error_tensor)
        temp = error_tensor - np.sum(error_tensor @ soft, axis=1)
        return soft @ temp
