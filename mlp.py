from sklearn.datasets import make_blobs
import numpy as np
import torch
import re
from tqdm import tqdm


class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.sigmoid(output)
        return output


# create random data points
def blob_label(y, label, loc):  # assign labels
    target = np.copy(y)
    for l in loc:
        target[y == l] = label
    return target


def getUserFLI(userLines, oneDict=False):
    features = []
    labels = []
    index = 0
    for line in userLines:
        if line[0] == '1':
            line = line[8:]+' '
            features.append([float(feature) for feature in re.findall("\:(\S+)\s",line)])
            labels.append(1)
            index = index + 1
        else:
            line = line[8:] + ' '
            features.append([float(feature) for feature in re.findall("\:(\S+)\s", line)])
            labels.append(0)
    features = np.array(features)
    labels = np.array(labels)
    if oneDict:
        return {'features': features, 'labels': labels, 'index': index}
    return features, labels, index


def read_data(f):
    lines = [line.strip() for line in f.readlines()]
    user = 0
    usersFLI = {}
    userLines = []
    for line in tqdm(lines):
        if line[0] == '#':
            user += 1
            usersFLI[user] = getUserFLI(userLines, oneDict=True)
            userLines = []
        else:
            userLines.append(line)
    print(usersFLI[1])
    print(usersFLI[196])
    return usersFLI


def all_mlp(train_fli, test_fli):
    for user_id in tqdm(range(1, len(train_fli))):
        preF = open('tLda/preLdaMlp.txt', "a")
        preF.write(f"# query {user_id}\n")

        x_train = torch.FloatTensor(train_fli[user_id]['features'])
        y_train = torch.FloatTensor(train_fli[user_id]['labels'])

        x_test = torch.FloatTensor(test_fli[user_id]['features'])
        y_test = torch.FloatTensor(test_fli[user_id]['labels'])

        each_mlp(x_train, y_train, x_test, y_test)
        preF.close()


def each_mlp(x_train, y_train, x_test, y_test):
    preF = open('tLda/preLdaMlp.txt', "a")

    # Model, Criterion, Optimizer
    model = Feedforward(100, 10)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # train the model
    model.eval()
    y_pred = model(x_test)
    before_train = criterion(y_pred.squeeze(), y_test)
    print("test loss before training", before_train.item())

    model.train()
    epoch = 20

    for epoch in range(epoch):
        optimizer.zero_grad()

        # forward pass
        y_pred = model(x_train)

        # compute loss
        loss = criterion(y_pred.squeeze(), y_train)

        print("Epoch {}: train loss: {}".format(epoch, loss.item()))

        # backward pass
        loss.backward()
        optimizer.step()

    # evaluation
    model.eval()
    y_pred = model(x_test)
    spe_user_pre = y_pred.detach().numpy().tolist()
    for tag_pre in spe_user_pre:
        preF.write(f"{tag_pre[0]}\n")

    print(y_pred.squeeze())
    after_train = criterion(y_pred.squeeze(), y_test)
    print("test loss after training", after_train.item())


if __name__ == "__main__":
    trainF = open("tLda/trainLda.dat", "r", encoding="utf-8")
    testF = open("tLda/testLda.dat", "r", encoding="utf-8")
    train_fli = read_data(trainF)
    test_fli = read_data(testF)
    all_mlp(train_fli, test_fli)