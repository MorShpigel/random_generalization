import torch
import torch.nn as nn
import numpy as np
from utilities import calc_accuracy
from models import DiagonalNetworks
from create_data import create_sparse_data

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


def find_GD_generalzation(x_train, y_train, x_test, y_test, d, depth, epochs, optimizer_type, lr, weight_decay, criterion=nn.BCEWithLogitsLoss(), loss_max_thr=1, loss_min_thr=0, seed=0):
    """
    This function calls train_gd calculates the resulting test accuracy
    """
    # set seed
    torch.manual_seed(seed)
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_gd = DiagonalNetworks(d, depth=depth).to(device)
    if optimizer_type=='SGD':
        optimizer = torch.optim.SGD(model_gd.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        print("Optimizer not supported")
    iter, status = train_gd(model_gd, optimizer, criterion, x_train, y_train, epochs=epochs, loss_max_thr=loss_max_thr, loss_min_thr=loss_min_thr, print_loss=False)
    if status=="Failed":
        return torch.tensor(-1), status
    else:
        train_acc, y_hat = calc_accuracy(model_gd, x_train, y_train)
        loss_train = criterion(y_hat, (y_train>0).float())
        test_acc, _ = calc_accuracy(model_gd, x_test, y_test)
        return test_acc, status
    

def run_GD_generalzation_exp(N_train, N_test, dataset, data_seed, d, r, mu, number_of_GD_models_to_try, criterion_type, loss_max_thr, loss_min_thr, depth, epochs, optimizer_type, lr, weight_decay):
    """
    This function runs the find_GD_generalzation function number_of_GD_models_to_try times and returns the resulting test accuracy
    """
    x_train, y_train, x_test, y_test = create_sparse_data(n_train=N_train, n_test=N_test, r=r, dataset=dataset, normalized_flag=False, seed=data_seed, mu=mu, d=d)
    if criterion_type=='BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss()
    else:
        print("Criterion not supported")
    test_acc_vec = []
    status_vec = []
    for i in range(number_of_GD_models_to_try):
        test_acc, status = find_GD_generalzation(x_train, y_train, x_test, y_test, d, depth, epochs, optimizer_type, lr, weight_decay, criterion=criterion, loss_max_thr=loss_max_thr, loss_min_thr=loss_min_thr, seed=data_seed+i)
        test_acc_vec.append(test_acc.cpu().numpy())
        status_vec.append(status)
    return test_acc_vec, status_vec