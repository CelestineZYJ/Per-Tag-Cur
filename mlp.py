from sklearn.datasets import make_blobs
import numpy
import torch


# create random data points
def blob_label(y, label, loc):  # assign labels
    target = numpy.copy(y)
    for l in loc:
        target[y == l] = label
    return target


# make data
def train_test():
    x_train, y_train = make_blobs(n_samples=40, n_features=2, cluster_std=1.5, shuffle=True)
    x_train = torch.FloatTensor(x_train)
    y_train = torch.FloatTensor(blob_label(y_train, 0, [0]))  # negative sample
    y_train = torch.FloatTensor(blob_label(y_train, 1, [1, 2, 3]))  # positive sample

    x_test, y_test = make_blobs(n_samples=10, n_features=2, cluster_std=1.5, shuffle=True)
    x_test = torch.FloatTensor(x_test)
    y_test = torch.FloatTensor(blob_label(y_test, 0, [0]))
    y_test = torch.FloatTensor(blob_label(y_test, 1, [1, 2, 3]))

    # Model, Criterion, Optimizer
    model = torch.nn.Feedforward(2, 10)
    criterion = torch.nn.BCELoss()
    optimizer = torch.nn.SGD(model.parameters(), lr=0.01)

    # train the model
    model.eval()
    y_pred = model(x_test)
    before_train = criterion(y_pred.squeeze(), y_test)
    print("test losdd before training", before_train.item())

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
    after_train = criterion(y_pred.squeeze(), y_test)
    print("test loss after training", after_train.item())
