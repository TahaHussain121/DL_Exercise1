from copy import deepcopy
class NeuralNetwork:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None

    def forward(self):
        input_tensor, label_tensor = self.data_layer.next() #tensoris just fancy word for array?
        self.label_tensor=label_tensor
        output = input_tensor
        for layer in self.layers:  #iterating through layers and updating the output and finally returning the output which will be the value from last layer
            output = layer.forward(output)

        return self.loss_layer.forward(output,label_tensor)

    def backward(self, label_tensor):
        error_tensor = self.loss_layer.backward(label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = deepcopy(self.optimizer)
        self.layers.append(layer)

    def train(self, iterations):
        for _ in range(iterations):
            loss = self.forward()
           # loss = self.loss_layer.forward(prediction, self.data_layer.label_tensor)
            self.loss.append(loss)
            self.backward(self.label_tensor)
            for layer in self.layers:
                if layer.trainable:
                    layer.weights = layer.optimizer.calculate_update(layer.weights, layer.gradient_weights)

    def test(self, input_tensor):
        output = input_tensor
        for layer in self.layers:
            output = layer.forward(output)
        return output
