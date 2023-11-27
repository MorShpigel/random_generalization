import torch
import torch.nn as nn
import numpy as np
from utilities import calc_accuracy
from models import DiagonalNetworks

def train_gd(model, optimizer, criterion, x_train, y_train, epochs=1000, loss_max_thr=1, loss_min_thr=0 ,print_loss=True):
    """
    This function trains a given model using gradient descent
    
    Input:
    model - model to train
    optimizer - optimizer to use
    criterion - loss function
    x_train - training data
    y_train - training labels
    epochs - number of epochs to train
    loss_thr - loss threshold to stop training (only if the training accuracy is 1.0)
    print_loss - whether to print the loss during training
    
    Output:
    iter - number of iterations it took to find the interpolating model
    status - "Sucsses" if found an interpolating model, "Failed" otherwise
    """
    for iter in range(epochs):
        model.train()
        optimizer.zero_grad()
        with torch.no_grad():
            train_acc, y_hat_normalized = calc_accuracy(model.forward_normalize, x_train, y_train)
            loss_train_normalized = criterion(y_hat_normalized, (y_train>0).float())
        y_hat = model.forward(x_train)
        loss_train = criterion(y_hat, (y_train>0).float())

        loss_train.backward()
        optimizer.step()
        if iter%100 == 0 and print_loss:
            print(f"iter:{iter}, Training loss:{loss_train}, Training accuracy:{train_acc}")
        if train_acc==1.0 and loss_train_normalized<loss_max_thr and loss_train_normalized>loss_min_thr:
            return iter, "Success"
        if loss_train_normalized<loss_min_thr:
            return iter, "Failed"
    return iter, "Failed"


def calc_GD_generalzation(x_train, y_train, x_test, y_test, lr=0.1, number_of_models_to_try=10, epochs=1000, loss_max_thr=1, loss_min_thr=0, criterion=nn.BCEWithLogitsLoss(), weight_decay=0):
    """
    This function calculates the avarage generalization of the gradient descent algorithm 
    by running the algorithm number_of_models times and averaging the resulting models test accuracy
    
    Input:
    x_train - training data
    y_train - training labels
    x_test - test data
    y_test - test labels
    lr - learning rate
    number_of_models - number of models to train
    epochs - number of epochs to train
    loss_thr - loss threshold to stop training (only if the training accuracy is 1.0)
    
    Output:
    test_acc_mean - avarage test accuracy of the models
    test_acc_std - standard deviation of the test accuracy of the models
    succ_counter - number of models that were trained successfully
    """
    d = x_train.shape[1]
    succ_counter = 0
    test_acc_vec = []
    for i in range(number_of_models_to_try):
        if i%100==0:
            print(f"GD: Initializing model number {i}")
        model_gd = DiagonalNetworks(d)
        optimizer = torch.optim.SGD(model_gd.parameters(), lr=lr, weight_decay=weight_decay)
        iter, status = train_gd(model_gd, optimizer, criterion, x_train, y_train, epochs=epochs, loss_max_thr=loss_max_thr, loss_min_thr=loss_min_thr, print_loss=False)
        if status=="Failed":
            continue
        succ_counter += 1
        test_acc, _ = calc_accuracy(model_gd, x_test, y_test)
        test_acc_vec.append(test_acc)
    if (len(test_acc_vec)):
        test_acc_mean = np.array(test_acc_vec).mean()
        test_acc_std = np.array(test_acc_vec).std()
    else:
        test_acc_mean = -1
        test_acc_std = -1
    return test_acc_mean, test_acc_std, succ_counter