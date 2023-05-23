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
        soft = self.input_tensor
        # print('y_hat.shape = ', soft.shape)
        # print('error_tensor', error_tensor)

        #temp = error_tensor - np.sum(np.diag(error_tensor.T @ soft), axis=0)

        temp = 0
        for j in range(0, soft.shape[1]):
            temp += np.multiply(error_tensor[:, j], soft[:, j])
        temp = (error_tensor.T - temp).T

        #print('temp', temp)
        #print('soft', soft)
        #print('shape multiply', np.multiply(soft, temp))
        return np.multiply(soft, temp)      # element wise multiplication
