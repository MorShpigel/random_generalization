import torch
import torchvision
import numpy as np

def create_random_data(n, d, r, mu=0, seed=0):
    """
    This function creates a random data set
    
    Input:
    n - number of samples
    d - dimension of the data
    r - number of non zero entries in the ground truth vector
    mu - mean of the data
    seed - random seed
    
    Output:
    X - data matrix
    y - labels vector
    """
    torch.manual_seed(seed)
    X = torch.normal(mu, 1, (n, d))
    X = torch.div(X, torch.matmul(torch.norm(X, dim=1).view(len(X), 1), torch.ones(1, X.shape[1]))) # normalize
    # X = torch.tensor(np.random.normal(mu, 1, (n, d))).to(torch.float32)
    y = create_sparse_labels(X, r)
    return X, y


def create_sparse_labels(X, r):
    """
    This function creates a sparse labels vector
    
    Input:
    X - data matrix
    r - number of non zero entries in the ground truth vector
    
    Output:
    y - labels vector
    """
    d = X.shape[1]
    w_star = 1/np.sqrt(r)*torch.cat((torch.ones(r), torch.zeros(d-r))) # spares ground truth vector
    y_hat = torch.matmul(X, w_star)
    y = torch.sign(y_hat)
    return y


def create_data_Mnist(n_train=512, n_test=128, normalized_flag=True, class_A_label=0, class_B_label=1, r=1):
    """
    This function creates a data set from the MNIST dataset. The labels are produced using a sparse ground truth vector with r non zero entries.
    
    Input:
    n_train - number of training samples
    n_test - number of test samples
    normalized_flag - whether to normalize the data
    class_A_label - label of the first class
    class_B_label - label of the second class
    r - number of non zero entries in the ground truth vector
    
    Output:
    x_train - training data
    y_train - training labels
    x_test - test data
    y_test - test labels
    """
    
    train_data = torchvision.datasets.MNIST('./mnist/', train=True, download=True,
                            transform=torchvision.transforms.ToTensor())

    class_A_x = train_data.train_data[train_data.train_labels == class_A_label,:,:].flatten(start_dim=1).to(torch.float32)
    class_B_x = train_data.train_data[train_data.train_labels == class_B_label,:,:].flatten(start_dim=1).to(torch.float32)
    # class_A_y = train_data.train_labels[train_data.train_labels == class_A_label]
    # class_B_y = train_data.train_labels[train_data.train_labels == class_B_label]
    x_train = torch.cat((class_A_x[:n_train//2,:], class_B_x[:n_train//2,:]), 0)
    x_test = torch.cat((class_A_x[n_train//2:n_train//2+n_test//2,:], class_B_x[n_train//2:n_train//2+n_test//2,:]), 0)
    y_train = create_sparse_labels(x_train, r)
    y_test = create_sparse_labels(x_test, r)

    # normalize the data
    if normalized_flag:
        x_train = torch.div(x_train, torch.matmul(torch.norm(x_train, dim=1).view(len(x_train), 1), torch.ones(1, x_train.shape[1])))
        x_test = torch.div(x_test, torch.matmul(torch.norm(x_test, dim=1).view(len(x_test), 1), torch.ones(1, x_test.shape[1])))

    return x_train, y_train, x_test, y_test


def create_sparse_data(n_train, n_test, r=1, dataset="random", seed=0, mu=0, d=50, normalized_flag=True):
    """
    This function creates a sparse data set

    Input:
    n_train - number of training samples
    n_test - number of test samples
    r - number of non zero entries in the ground truth vector
    dataset - "random" or "MNIST"
    seed - random seed (only relevant for "random" dataset)
    mu - mean of the data (only relevant for "random" dataset)
    d - dimension of the data (only relevant for "random" dataset)
    normalized_flag - whether to normalize the data (only relevant for "MNIST" dataset)

    Output:
    x_train - training data
    y_train - training labels
    x_test - test data
    y_test - test labels
    """
    if dataset=="random":
        x_train, y_train = create_random_data(n_train, d, r, mu, seed)
        x_test, y_test = create_random_data(n_test, d, r, mu, seed+1)
    elif dataset=="MNIST":
        x_train, y_train, x_test, y_test = create_data_Mnist(n_train, n_test, normalized_flag, class_A_label=0, class_B_label=1, r=r)
    return x_train, y_train, x_test, y_test