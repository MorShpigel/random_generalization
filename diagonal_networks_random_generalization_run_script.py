import numpy as np
from create_data import create_sparse_data
from guess_n_check import calc_guess_and_check_generalzation
from gd import calc_GD_generalzation
from utilities import save_results, get_suffix, is_debug, get_run_name
from matplotlib import pyplot as plt
import torch.nn as nn
import argparse
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--N_train', type=int, default=16,
                    help='number of training data samples')
parser.add_argument('--N_test', type=int, default=2000,
                    help='number of test data samples')
parser.add_argument('--dataset', type=str, default="random",
                    help='dataset')
parser.add_argument('--d', type=int, default=30,
                    help='data points dimension')
parser.add_argument('--mu', type=int, default=0,
                    help='mean of the data')
parser.add_argument('--seed_num', type=int, default=10,
                    help='number of seeds to try')
parser.add_argument('--T_find_interpolating_sol', type=int, default=100000,
                    help='number of random models to sample in order to find an interpolating model')
parser.add_argument('--number_of_GNC_models_to_try', type=int, default=1000,
                    help='number of models to (try to) obtain using the guess and check algorithm')
parser.add_argument('--gd_epochs', type=int, default=10000,
                    help='number of epochs to train')
parser.add_argument('--gd_lr', type=float, default=0.1,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0,
                    help='GD weight decay parameter') 
parser.add_argument('--loss_max_thr', type=float, default=0.3,
                    help='loss threshold to stop training (only if the training accuracy is 1.0)')
parser.add_argument('--loss_min_thr', type=float, default=0.2,
                    help='loss threshold to stop training (only if the training accuracy is 1.0)')
parser.add_argument('--depth', type=int, default=30,
                    help='depth of the model')
parser.add_argument('--results_folder', type=str, default="Results",
                    help='results folder')
args = parser.parse_args()

# define parameters
r_vec = np.arange(1, 10, 2)
mu = 0
dataset = args.dataset
N_train, N_test, d = args.N_train, args.N_test, args.d
# N_train, N_test, d = 10, 10, 30
# N_train, N_test, d = 25, 10, 50
criterion = nn.BCEWithLogitsLoss()
T_find_interpolating_sol = args.T_find_interpolating_sol
number_of_GNC_models_to_try = args.number_of_GNC_models_to_try
gd_epochs = args.gd_epochs
loss_max_thr = args.loss_max_thr
loss_min_thr = args.loss_min_thr
gd_lr = args.gd_lr
seed_vec = np.arange(args.seed_num)
dist_vec = ["Normal", "Normal_uv", "Laplace"]
depth = args.depth

# initialize results dictionary
mean_test_acc_dict = {}
std_test_acc_dict = {}
test_acc_dict = {}
test_acc_dict =	{
  "Normal": [],
  "Normal_uv": [],
  "Laplace": [],
  "GD": []
}
# run the experiment
for r in r_vec:
    if not is_debug():
        wandb_args = vars(args)
        wandb_args["r"] = r
        wandb.init(project='diagonal_networks_random_generalization', config=wandb_args, name=get_run_name(args))
    for seed_ind, seed in enumerate(seed_vec):
        x_train, y_train, x_test, y_test = create_sparse_data(n_train=N_train, n_test=N_test, r=r, dataset=dataset, normalized_flag=False, seed=seed, mu=mu, d=d)
        for dist in dist_vec:
            if seed%25==0:
                print(f"r={r}, seed={seed}, dist={dist}")
            test_acc_mean, test_acc_std, _ = calc_guess_and_check_generalzation(x_train, y_train, x_test, y_test, dist,  T_find_interpolating_sol, number_of_GNC_models_to_try, seed, depth=depth, criterion=criterion, loss_max_thr=loss_max_thr, loss_min_thr=loss_min_thr)
            test_acc_dict[dist].append(test_acc_mean)
        
        if seed%25==0:
            print(f"r={r}, seed={seed}, GD")
        test_acc_mean, test_acc_std, _ = calc_GD_generalzation(x_train, y_train, x_test, y_test, lr=gd_lr, number_of_models_to_try=number_of_GNC_models_to_try, epochs=gd_epochs, loss_max_thr=loss_max_thr, loss_min_thr=loss_min_thr, criterion=criterion, weight_decay=args.weight_decay)
        test_acc_dict["GD"].append(test_acc_mean)
        if not is_debug():
            res_last = [("seed", seed_vec[seed_ind]), ("mean_test_acc_normal", test_acc_dict["Normal"][-1]),
                                    ("mean_test_acc_normal_uv", test_acc_dict["Normal_uv"][-1]), ("mean_test_acc_laplace", test_acc_dict["Laplace"][-1]),
                                    ("mean_test_acc_gd", test_acc_dict["GD"][-1])]
            wandb.log(dict(res_last))
    for dist in ["Normal", "Normal_uv", "Laplace", "GD"]:
        print(dist, np.array(test_acc_dict[dist]).mean())
        mean_test_acc_dict[(r, dist)] = np.array(test_acc_dict[dist]).mean()
        std_test_acc_dict[(r, dist)] = np.array(test_acc_dict[dist]).std()
        # save checkpoint
        print("Saving checkpoint")
        save_results(mean_test_acc_dict, std_test_acc_dict, args, checkpoint=True)
        # if not is_debug():
        #     wandb.log({"mean_across_seeds_test_acc_normal": mean_test_acc_dict[(r,"Normal")], "mean_across_seeds_test_acc_normal_uv": mean_test_acc_dict[(r,"Normal_uv")],
        #                 "mean_across_seeds_test_acc_laplace": mean_test_acc_dict[(r,"Laplace")], "mean_across_seeds_test_acc_gd": mean_test_acc_dict[(r,"GD")]})

# save results
save_results(mean_test_acc_dict, std_test_acc_dict, args)

# plot results
for dist in ["Normal", "Normal_uv", "Laplace", "GD"]:
    mean_test_acc_vec = [mean_test_acc_dict[(r, dist)] for r in r_vec]
    plt.plot(r_vec, mean_test_acc_vec, label=dist)

plt.xlabel("Sparsity")
plt.ylabel("Mean Test Accuracy")
# plt.yscale("log")
plt.legend()
# plt.show()
plt.savefig(f"Figures/diagonal_networks_random_generalization_{get_suffix(args)}.png")