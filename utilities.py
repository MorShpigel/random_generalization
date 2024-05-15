import torch
import pickle 
import sys
import pandas as pd
from filelock import FileLock
import os
import yaml

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


def fill_config(config, dist):
    """
    This function fills the configuration with default values
    """
    config_full = config.copy()
    config_full['dist'] = dist
    if dist=="GD":
        config_full['T_find_interpolating_sol'] = -1
        config_full['number_of_GNC_models_to_try'] = -1
    else:
        config_full['number_of_GD_models_to_try'] = -1
        config_full['epochs'] = -1
        config_full['lr'] = -1
        config_full['weight_decay'] = -1
        config_full['optimizer_type'] = 'None'
    return config_full


def save_result_to_csv(test_acc_vec, status_vec, iter_vec, dist, config, file_name):
    """
    This function saves the results of the experiment to a CSV file
    """
    # Convert the configuration to a DataFrame
    config_full = fill_config(config, dist)
    df_config = pd.DataFrame(config_full, index=[0])
    results_num = len(test_acc_vec)
    # Convert the results to a DataFrame
    if dist=="GD":
        df_config = pd.concat([df_config]*config_full['number_of_GD_models_to_try'], ignore_index=True)
        df_config2 = pd.DataFrame({'sample_seed': range(config_full['data_seed'], config_full['data_seed']+config_full['number_of_GD_models_to_try'])})
    else:
        # df_config = pd.concat([df_config]*config_full['number_of_GNC_models_to_try'], ignore_index=True)
        df_config = pd.concat([df_config]*results_num, ignore_index=True)
        df_config2 = pd.DataFrame({'sample_seed': range(config_full['data_seed'], config_full['data_seed']+results_num)})
    df_config = pd.concat([df_config, df_config2], axis=1)
    df_results = pd.DataFrame({'status': status_vec, 'iter': iter_vec, 'test_acc': test_acc_vec})
    # Concatenate the results and configuration
    df = pd.concat([df_config, df_results], axis=1)

    # Specify the order of the keys
    keys = ['N_test', 'N_train', 'dataset', 'd', 'mu', 'r', 'data_seed', 'depth', 'criterion_type', 'loss_max_thr', 'loss_min_thr', 'dist', 'T_find_interpolating_sol', 'number_of_GNC_models_to_try', 'number_of_GD_models_to_try', 'epochs', 'lr', 'weight_decay', 'optimizer_type', 'sample_seed', 'status', 'iter', 'test_acc']

    # Reorder the columns of the DataFrame
    df = df[keys]

    lock = FileLock(file_name + ".lock")
    with lock:
        # Check if file exists
        if not os.path.exists(file_name):
            df.to_csv(file_name, mode='w', index=False)
        # Check if file is empty
        elif os.stat(file_name).st_size == 0:
            df.to_csv(file_name, mode='a', index=False)
        else:
            df.to_csv(file_name, mode='a', header=False, index=False)


def print_configs(file="configs.yaml"):

    # Load the YAML file
    with open(file, 'r') as f:
        data = yaml.safe_load(f)

    # Convert the dictionary to a DataFrame
    df_dist = pd.DataFrame(data['experiment_dist'])
    df_gd = pd.DataFrame(data['experiment_GD'])
    
    # Print the DataFrame
    print(df_dist)
    print(df_gd)


    

    
