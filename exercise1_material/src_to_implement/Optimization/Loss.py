import numpy as np


class CrossEntropyLoss:

    def __int__(self):
        pass

    def forward(self, prediction_tensor, label_tensor):
        #print(label_tensor)
        #index_array = np.argwhere(label_tensor == 1)
        #print('index_array\n', index_array)
        loss = np.sum(-np.log(prediction_tensor[label_tensor == 1] + np.finfo(np.float64).eps))
        #print('indexed values\n', prediction_tensor[index_array])
        #print('loss = ', loss)
        return loss

    def backward(self, label_tensor):
        pass