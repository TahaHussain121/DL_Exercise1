import numpy as np
from Layers import Base

class SoftMax(Base.BaseLayer):

    def __init__(self):
        super().__init__()
        self.y = None
        self.trainable = False

    def forward(self, input_tensor):
        maximum = np.max(input_tensor, 1)
        reshaped_maximum = maximum.reshape(-1, 1)
        temp = input_tensor - reshaped_maximum

        self.output = np.exp(temp) / np.expand_dims(np.sum(np.exp(temp), axis=1), 1)
        return self.output

    def backward(self, error_tensor):
        scalar_rows = np.sum(np.multiply(error_tensor, self.output), axis=1)
        selected_rows = error_tensor
        selected_rows_transposed = selected_rows.T
        subtracted_rows = selected_rows_transposed - scalar_rows
        error = self.output * subtracted_rows.T

        return error



   # def forward(self, input_tensor):    # softmax applied "row-wise" as described in pdf
        #print('input_tensor', input_tensor)
    #   input_tensor = input_tensor - np.amax(input_tensor)
    #  input_tensor = np.exp(input_tensor)
    # sum = np.sum(input_tensor, axis=1)
    #    sum = sum[..., np.newaxis]
    #   self.y =input_tensor / sum
    #   print("this is output of forward")
    #   print(self.y)
    #   return self.y

    #def backward(self, error_tensor):
        # print('error_tensor', error_tensor)
        #s=np.sum(np.multiply(error_tensor,self.y),axis=1)
        #for()
        #temp = error_tensor - np.sum(error_tensor @ soft, axis=1)
        #return soft @ temp
       # return  error_tensor
