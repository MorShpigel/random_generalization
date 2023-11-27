import numpy as np
from utilities import calc_accuracy
from models import DiagonalNetworks_sampled
from torch import nn

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


def calc_guess_and_check_generalzation(x_train, y_train, x_test, y_test, dist="Normal", T_find_interpolating_sol=100000, number_of_GNC_models_to_try=10, seed=0, depth=2, criterion=nn.BCEWithLogitsLoss(), loss_max_thr=1, loss_min_thr=0):
    """
    This function calculates the avarage generalization of the guess and check algorithm 
    by running the algorithm number_of_GNC_models_to_try times and averaging the resulting models test accuracy
    
    Input:
    x_train - training data
    y_train - training labels
    x_test - test data
    y_test - test labels
    dist - distribution of the weights
    T_find_interpolating_sol - number of random models to sample
    number_of_GNC_models_to_try - number of models to (try to) obtain using the guess and check algorithm
    seed - random seed
    depth - depth of the model
    
    Output:
    test_acc_mean - avarage test accuracy of the models
    test_acc_std - standard deviation of the test accuracy of the models
    succ_counter - number of models that were trained successfully
    """
    d = x_train.shape[1]
    succ_counter = 0
    test_acc_vec = []
    model = DiagonalNetworks_sampled(dist, d, depth=depth)
    for i in range(number_of_GNC_models_to_try):
        if i%100==0:
            print(f"GNC: Trying model {i} out of {number_of_GNC_models_to_try}")
        iter, status = find_interpolating_sol(x_train, y_train, model, T_find_interpolating_sol, seed+i, criterion=criterion, loss_max_thr=loss_max_thr, loss_min_thr=loss_min_thr)
        if status=="Failed":
            continue
        train_acc, y_hat = calc_accuracy(model.forward_normalize, x_train, y_train)
        loss_train = criterion(y_hat, (y_train>0).float())
        # if loss_train>loss_max_thr or loss_train<loss_min_thr:
        #     continue
        succ_counter += 1
        test_acc, _ = calc_accuracy(model.forward_normalize, x_test, y_test)
        test_acc_vec.append(test_acc)
    if (len(test_acc_vec)):
        test_acc_mean = np.array(test_acc_vec).mean()
        test_acc_std = np.array(test_acc_vec).std()
    else:
        test_acc_mean = -1
        test_acc_std = -1
    return test_acc_mean, test_acc_std, succ_counter