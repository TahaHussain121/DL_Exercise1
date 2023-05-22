#if __name__ == '__main__':
from Layers import Base  # the dot means the current folder / path

import numpy as np


class FullyConnected(Base.BaseLayer):
    # self.input_size = 4
    # self.output_size = 3
    def __init__(self, input_size, output_size):
        super().__init__()      # how on earth do I change inherited value???
        self.trainable = True
        self.weights = np.random.rand(input_size+1, output_size)
        #print('self.weights', self.weights)
        self._optimizer = None
        self.current_input = None
        self.current_error = None

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt_function):
        self._optimizer = opt_function

    def forward(self, input_tensor):
        input_tensor = np.concatenate((input_tensor.T, np.ones((1, input_tensor.shape[0])))).T
        self.current_input = input_tensor
        #print('(input_tensor + bias).shape', input_tensor.shape, '\n input_tensor', input_tensor)
        #print('self.weights.shape', self.weights.shape)
        output = input_tensor @ self.weights
        #print('output shape', output.shape)
        return output

    def backward(self, error_tensor):
        #print('shape error_tensor', error_tensor.shape)
        self.current_error = error_tensor
        weights_without_error = np.delete(self.weights, -1, axis=0)
        error_tensor_new = error_tensor @ weights_without_error.T

        #if isinstance(self._optimizer, (Sgd)):
        if type(self._optimizer).__name__ == 'Sgd':
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)

        return error_tensor_new

    @property
    def gradient_weights(self):
        #print('error', self.current_error)
        return (self.current_error.T @ self.current_input).T


# B = Base.BaseLayer()
# print(B.trainable)
# F = FullyConnected(2,2)
#
# print('F.trainable', F.trainable)
# print('B.trainable', B.trainable)
#print(help(F))

#print(Base.BaseLayer().trainable)