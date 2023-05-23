from Layers import Base
import numpy as np


class ReLU(Base.BaseLayer):

    def __int__(self):
        super().__init__()
        self.trainable = False
        self.input_tensor = None


    def forward(self, input_tensor):
        output = np.multiply(input_tensor, input_tensor > 0)
        self.input_tensor = input_tensor
        return output

    def backward(self, error_tensor):
        return np.multiply(error_tensor, self.input_tensor > 0)  # > 0
   # def forward(self, input_tensor):
    #    self.input_tensor = input_tensor
    #    input_tensor[input_tensor < 0] = 0
    #    return input_tensor

    #def backward(self, error_tensor):
    #   error_tensor[self.input_tensor < 0] = 0
    #    return error_tensor



