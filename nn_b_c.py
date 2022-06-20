import torch.nn as nn
import torch
import torch.optim as optim


class NN_Model(nn.Module):
    def __init__(self):
        super(NN_Model, self).__init__()
        # Number of input features is 11.
        self.layer_1 = nn.Linear(11, 64)
        self.layer_2 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return x


class set_model:

    def load_model(self, LEARNING_RATE, device):
        model = NN_Model()
        model.to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        return model, criterion, optimizer

    def binary_acc(self, y_pred, y_test):
        y_pred_tag = torch.round(torch.sigmoid(y_pred))

        correct_results_sum = (y_pred_tag == y_test).sum().float()
        acc = correct_results_sum / y_test.shape[0]
        acc = torch.round(acc * 100)

        return acc