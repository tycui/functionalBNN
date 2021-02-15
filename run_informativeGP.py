import torch
import numpy as np
import gpytorch
from bnn_posterior import MeanfieldNNPosterior
from fvi import FunctionalVI
from gradient_estimator import SpectralScoreEstimator
from data.data_generation import data_generating_yacht
from gp_prior import GPPrior, LinearRegression
from utils import pve
import argparse
import os
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser('Functional BNN on regression')
parser.add_argument('-nn', '--num_nodes', type=int, default=50)
parser.add_argument('-nl', '--num_layers', type=int, default=1)
parser.add_argument('-no', '--n_rand', type=int, default=20)
parser.add_argument('-nf', '--n_functions', type=int, default=100)
parser.add_argument('-in', '--injected_noise', type=float, default=0.01)
parser.add_argument('-eta', type=float, default=0.)
parser.add_argument('--n_eigen_threshold', type=float, default=0.99)
parser.add_argument('--learning_rate_gp', type=float, default=0.003)
parser.add_argument('--num_epoch_gp', type=int, default=5000)
parser.add_argument('-lrnn', '--learning_rate_bnn', type=float, default=0.001)
parser.add_argument('-b', '--batch_size', type=int, default=50)
parser.add_argument('-enn', '--num_epoch_bnn', type=int, default=20000)
parser.add_argument('--coeff_ll', type=float, default=1.)
parser.add_argument('--coeff_kl', type=float, default=1.)
parser.add_argument('--cuda', type=bool, default=True)
group = parser.add_mutually_exclusive_group()
group.add_argument('-rbf', '--RBF_Kernel', action='store_false')
args = parser.parse_args()

# load yacht dataset
x_train, y_train, x_test, y_test = data_generating_yacht(os.getcwd()+'/data/')
num_features = x_train.shape[-1]

# define a random generator
lower_ap = np.minimum(np.min(x_train.numpy()), np.min(x_test.numpy()))
upper_ap = np.maximum(np.max(x_train.numpy()), np.max(x_test.numpy()))
def rand_generator(n_rand, n_dim = num_features,  minval=lower_ap, maxval=upper_ap):
    return torch.rand((n_rand, n_dim)) * (maxval - minval) + minval

def sub_trainingset(ratio = 0.3):
    torch.manual_seed(1)
    perm = torch.randperm(len(x_train))
    x_train_perm = x_train[perm]; x_train_half = x_train_perm[:int(len(x_train) * ratio), :]
    y_train_perm = y_train[perm]; y_train_half = y_train_perm[:int(len(y_train) * ratio)]
    return x_train_half, y_train_half

# train a linear regression as informative prior
torch.manual_seed(1)
mean_function = LinearRegression(input_dim = x_train.shape[1])
mean_function.train_lr(x_train, y_train, args.learning_rate_gp, args.num_epoch_gp)
mean_function._detach()
print("PVE of the mean function on test set is {:.5f}".format(pve(mean_function(x_test), y_test).item()))

def validate_priors(x_train_half, y_train_half, prior_type):
    # torch.manual_seed(1)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    gradient_estimator = SpectralScoreEstimator(eta=args.eta, n_eigen_threshold=args.n_eigen_threshold)
    mfnn_noninfo = MeanfieldNNPosterior(num_features, args.num_nodes, args.num_layers)

    if prior_type == 'non_info':
        print('Without informative mean function:')
        gp_prior = GPPrior(x_train_half, y_train_half, likelihood)
        fvi = FunctionalVI(gp_prior, mfnn_noninfo, rand_generator, gradient_estimator, args.n_rand,
                                        args.n_functions, args.injected_noise, args.cuda)
        fvi.build_prior_gp(x_train_half, y_train_half, args.learning_rate_gp, args.num_epoch_gp)
    elif prior_type == 'info':
        print('With informative mean function:')
        gp_prior = GPPrior(x_train_half, y_train_half, likelihood, mean_function)
        fvi = FunctionalVI(gp_prior, mfnn_noninfo, rand_generator, gradient_estimator, args.n_rand,
                                        args.n_functions, args.injected_noise, args.cuda)
        fvi.build_prior_gp(x_train_half, y_train_half, args.learning_rate_gp, args.num_epoch_gp)
    elif prior_type == 'info_flex':
        print('With informative mean function and hyper-parameters from o mean GP:')
        gp_prior = GPPrior(x_train_half, y_train_half, likelihood)
        fvi = FunctionalVI(gp_prior, mfnn_noninfo, rand_generator, gradient_estimator, args.n_rand,
                                        args.n_functions, args.injected_noise, args.cuda)
        fvi.build_prior_gp(x_train_half, y_train_half, args.learning_rate_gp, args.num_epoch_gp)
        if args.cuda:
            fvi.gp_prior.mean_module = mean_function.cuda()
    else:
        raise Exception('Wrong prior types')

    fvi.init_training(x_train_half, args.learning_rate_bnn, args.batch_size, args.num_epoch_bnn, args.coeff_ll, args.coeff_kl)
    fvi.training(x_train_half, y_train_half)
    mse_test, ll_test, pve_test = fvi.build_evaluation(x_test, y_test)
    print('Evaluation on training set: MSE={:.5f} | logLL={:.5f} | PVE={:.5f}'.format(mse_test, ll_test, pve_test))
    return mse_test, ll_test, pve_test

# For different sizes of training set
RATIO = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]

prior_type = 'non_info' # 'non_info', 'info', 'info_flex'
writer = SummaryWriter("infoGP/"+prior_type)
i = 0
for ratio in RATIO:
    print("ratio is {:.1f}".format(ratio))
    x_train_half, y_train_half = sub_trainingset(ratio)
    mse_test, ll_test, pve_test = validate_priors(x_train_half, y_train_half, prior_type)
    writer.add_scalar('MSE', mse_test, i)
    writer.add_scalar('Likelihood', ll_test, i)
    writer.add_scalar('PVE', pve_test, i)
    i += 1

prior_type = 'info' # 'non_info', 'info', 'info_flex'
writer = SummaryWriter("infoGP/"+prior_type)
i = 0
for ratio in RATIO:
    print("ratio is {:.1f}".format(ratio))
    x_train_half, y_train_half = sub_trainingset(ratio)
    mse_test, ll_test, pve_test = validate_priors(x_train_half, y_train_half, prior_type)
    writer.add_scalar('MSE', mse_test, i)
    writer.add_scalar('Likelihood', ll_test, i)
    writer.add_scalar('PVE', pve_test, i)
    i += 1

prior_type = 'info_flex' # 'non_info', 'info', 'info_flex'
writer = SummaryWriter("infoGP/"+prior_type)
i = 0
for ratio in RATIO:
    print("ratio is {:.1f}".format(ratio))
    x_train_half, y_train_half = sub_trainingset(ratio)
    mse_test, ll_test, pve_test = validate_priors(x_train_half, y_train_half, prior_type)
    print(mse_test)
    print(ll_test)
    print(pve_test)
    writer.add_scalar('MSE', mse_test, i)
    writer.add_scalar('Likelihood', ll_test, i)
    writer.add_scalar('PVE', pve_test, i)
    i += 1



