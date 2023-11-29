import numpy as np
from utilities import calc_accuracy
from models import DiagonalNetworks_sampled
import torch
from torch import nn
import time
from create_data import create_sparse_data
# import multiprocessing as mp


def find_interpolating_sol(x_train, y_train, model, number_of_models_to_sample=1000, seed=0, criterion=nn.BCEWithLogitsLoss(), loss_max_thr=1, loss_min_thr=0):
    """
    This function finds an interpolating model for the given data
    
    Input:
    x_train - training data
    y_train - training labels
    model - model to train
    number_of_models - number of random models to sample
    seed - random seed
    
    Output:
    iter - number of iterations it took to find the interpolating model
    status - "Sucsses" if found an interpolating model, "Failed" otherwise
    """
    np.random.seed(seed)
    for iter in range(number_of_models_to_sample):
        model.sample_params()
        train_acc, y_hat = calc_accuracy(model.forward_normalize, x_train, y_train)
        loss_train = criterion(y_hat, (y_train>0).float())
        if train_acc==1.0 and loss_train<loss_max_thr and loss_train>loss_min_thr:
            return iter, "Success"
    return iter, "Failed"


def find_interpolating_sol_generalzation(x_train, y_train, x_test, y_test, dist, d, depth, number_of_models_to_sample=1000, seed=0, criterion=nn.BCEWithLogitsLoss(), loss_max_thr=1, loss_min_thr=0):
    """
    This function calls find_interpolating_sol to find an interpolating model for the given data and calculates the resulting test accuracy
    """
    model = DiagonalNetworks_sampled(dist, d, depth=depth)
    iter, status = find_interpolating_sol(x_train, y_train, model, number_of_models_to_sample, seed, criterion=criterion, loss_max_thr=loss_max_thr, loss_min_thr=loss_min_thr)
    if status=="Failed":
        return -1, status
    else:
        train_acc, y_hat = calc_accuracy(model.forward_normalize, x_train, y_train)
        loss_train = criterion(y_hat, (y_train>0).float())
        test_acc, _ = calc_accuracy(model.forward_normalize, x_test, y_test)
        return test_acc, status


def run_guess_and_check_generalzation_exp(dist, N_train, N_test, dataset, data_seed, d, r, mu, T_find_interpolating_sol, number_of_GNC_models_to_try, criterion_type, loss_max_thr, loss_min_thr, depth):
    """
    This function runs the guess and check algorithm and calculates the resulting test accuracy
    """
    x_train, y_train, x_test, y_test = create_sparse_data(n_train=N_train, n_test=N_test, r=r, dataset=dataset, normalized_flag=False, seed=data_seed, mu=mu, d=d)
    if criterion_type=='BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss()
    else:
        print("Criterion not supported")
    test_acc_vec = []
    status_vec = []
    for i in range(number_of_GNC_models_to_try):
        test_acc, status = find_interpolating_sol_generalzation(x_train, y_train, x_test, y_test, dist, d, depth, T_find_interpolating_sol, data_seed+i, criterion=criterion, loss_max_thr=loss_max_thr, loss_min_thr=loss_min_thr)
        test_acc_vec.append(test_acc.cpu().numpy())
        status_vec.append(status)
    return test_acc_vec, status_vec
