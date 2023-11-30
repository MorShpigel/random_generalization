import numpy as np
from scipy.optimize import minimize
from create_data import create_sparse_data
from utilities import save_results, get_suffix, is_debug, get_run_name
import torch
import torch.nn as nn
import argparse
from sklearn.linear_model import Lasso, Ridge

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
parser.add_argument('--seed_num', type=int, default=0,
                    help='number of seeds to try')
parser.add_argument('--results_folder', type=str, default="Results",
                    help='results folder')
args = parser.parse_args()


def calc_test_acc(x_test, y_test, w):
    y_hat = torch.matmul(x_test, torch.tensor(w).float())
    predicton = (y_hat*y_test>0).float()
    test_acc = torch.sum(predicton)/len(y_test)
    return test_acc


# def constraint(v, x, y):
#     return np.multiply(y, np.matmul(x, v.reshape(-1, 1)).squeeze())-1


# def solver(x, y, w_0, obj='Q', optim_tol=1e-5):
#     x0 = (w_0).reshape(-1, )

#     cons = {'type': 'ineq', 'fun': lambda v: constraint(v, x, y)}

#     if obj == 'L1':
#         objective = lambda v: np.linalg.norm(v.squeeze(), ord=1)
#     elif obj == 'L2':
#         objective = lambda v: np.linalg.norm(v.squeeze(), ord=2)
#     else:
#         raise ValueError('objective not supported.')

#     sol = minimize(
#         fun=objective,
#         x0=x0,
#         constraints=cons,
#         tol=optim_tol,
#         method='SLSQP',
#         options={
#             'maxiter': 1000000,
#             'disp': False
#         }
#     )
#     is_failed = (not sol.success)
#     if is_failed:
#         raise RuntimeError('Minimization Failed.')

#     return sol.x

# define parameters
r_vec = np.arange(1, 10, 2)
mu = 0
dataset = args.dataset
N_train, N_test, d = args.N_train, args.N_test, args.d
seed_vec = [args.seed_num]

# run the experiment
for r in r_vec:
    test_acc_dict =	{
        "L1": [],
        "L2": [],
        "null": []
    }
    for seed_ind, seed in enumerate(seed_vec):
        x_train, y_train, x_test, y_test = create_sparse_data(n_train=N_train, n_test=N_test, r=r, dataset=dataset, normalized_flag=False, seed=seed, mu=mu, d=d)
        x_train, y_train, x_test, y_test = x_train.cpu(), y_train.cpu(), x_test.cpu(), y_test.cpu() 
        # w_0 = np.random.randn(d, 1)
        # Create l1 baseline
        # w_L1 = solver(
        # x=x_train,
        # y=y_train,
        # w_0=w_0,
        # obj='L1',
        # optim_tol=1e-16
        # )
        lasso = Lasso(alpha=0.1)
        lasso.fit(x_train, y_train)
        w_L1 = lasso.coef_
        test_acc_dict['L1'].append(calc_test_acc(x_test, y_test, w_L1))
        
        # Create l2 baseline
        # w_L2 = solver(
        #     x=x_train,
        #     y=y_train,
        #     w_0=w_0,
        #     obj='L2',
        #     optim_tol=1e-8
        # )
        # Create a Ridge regression object
        ridge = Ridge(alpha=0.1)  # alpha is the regularization parameter
        # Fit the model to the training data
        ridge.fit(x_train, y_train)
        # Get the coefficients of the solution
        w_L2 = ridge.coef_
        test_acc_dict['L2'].append(calc_test_acc(x_test, y_test, w_L2))

        # Create null baseline
        w_null = torch.sign(torch.sum(y_train))*torch.ones((d, 1)).squeeze()
        test_acc_dict['null'].append(calc_test_acc(x_test, y_test, w_null))

    # save results
    for reg in ["L1", "L2", "null"]:
        mean_test_acc = np.array(test_acc_dict[reg]).mean()
        std_test_acc = np.array(test_acc_dict[reg]).std()
        file_name = f"N_train={args.N_train}_N_test={args.N_test}_dataset={args.dataset}_d={args.d}_mu={args.mu}_seed_num={args.seed_num}_r={r}_reg={reg}"
        torch.save({"mean_test_acc": mean_test_acc, "std_test_acc": std_test_acc}, f"{args.results_folder}/{file_name}.pt")