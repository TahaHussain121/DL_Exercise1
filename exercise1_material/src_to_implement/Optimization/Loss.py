import numpy as np


class CrossEntropyLoss:

    def __int__(self):
        self.prediction_tensor = None

    def forward(self, prediction_tensor, label_tensor):
        #print(label_tensor)
        #index_array = np.argwhere(label_tensor == 1)
        self.prediction_tensor = prediction_tensor
        loss = np.sum(-np.log(prediction_tensor[label_tensor == 1] + np.finfo(np.float64).eps))
        #print('indexed values\n', prediction_tensor[index_array])
        #print('loss = ', loss)
        return loss

    def backward(self, label_tensor):
        return - (label_tensor / (self.prediction_tensor + np.finfo(np.float64).eps))
