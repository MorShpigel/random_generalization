import torch
import pickle 
import sys

def calc_accuracy(model, X, y):
    """
    This function calculates the accuracy of the model on a given data

    Input:
    model - model to train
    X - data matrix
    y - labels vector

    Output:
    acc - accuracy of the model
    y_hat - predicted labels
    """
    y_hat = model(X)
    predicton = (y_hat*y>0).float()
    acc = torch.sum(predicton)/len(y)
    return acc, y_hat


def get_run_name(args):
    suffix = f"run_N_train={args.N_train}_N_test={args.N_test}_d={args.d}_seed_num={args.seed_num}_loss_max_thr={args.loss_max_thr}_loss_min_thr={args.loss_min_thr}_depth={args.depth}"
    if args.weight_decay>0:
        suffix += f"_weight_decay={args.weight_decay}"
    return suffix


def get_suffix(args):
    suffix = f"N_train={args.N_train}_N_test={args.N_test}_dataset={args.dataset}_d={args.d}_mu={args.mu}_seed_num={args.seed_num}_T_find_interpolating_sol={args.T_find_interpolating_sol}_number_of_GNC_models_to_try={args.number_of_GNC_models_to_try}_gd_epochs={args.gd_epochs}_gd_lr={args.gd_lr}_loss_max_thr={args.loss_max_thr}_loss_min_thr={args.loss_min_thr}_depth={args.depth}"
    if args.weight_decay>0:
        suffix += f"_weight_decay={args.weight_decay}"
    return suffix


def get_file_name(name, args, checkpoint=False):
    suffix = get_suffix(args)
    if checkpoint:
        file_name = f"{args.results_folder}/{name}_{suffix}_checkpoint.pkl"
    else:
        file_name = f"{args.results_folder}/{name}_{suffix}.pkl"
    return file_name


def save_results(mean_test_acc_dict, std_test_acc_dict, args, checkpoint=False):
    with open(get_file_name("mean_test_acc_dict", args, checkpoint), 'wb') as f:
        pickle.dump(mean_test_acc_dict, f)
    with open(get_file_name("std_test_acc_dict", args, checkpoint), 'wb') as f:
        pickle.dump(std_test_acc_dict, f)


def is_debug():
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace is None:
        print("None!")
        return False
    elif gettrace():
        # print("Debug!")
        return True
    else:
        # print("No debug!")
        return False