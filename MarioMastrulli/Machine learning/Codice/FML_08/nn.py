from torch import nn

class NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NN, self).__init__()  # SimpleNN is a sub-class of nn.Module
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])# First layer -> Hidden Layer
        self.fc3 = nn.Linear(hidden_size[1], output_size)  # Hidden Layer -> Output
        self.sigmoid = nn.Sigmoid()  # Activation (it is a classification task) -> We produce predictions

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = self.fc3(x)
        return x