import torch
import numpy as np
import gpytorch
from bnn_posterior import MeanfieldNNPosterior
from fvi import FunctionalVI
from gradient_estimator import SpectralScoreEstimator
from data.data_generation import data_generating_yacht
from gp_prior import GPPrior
import argparse
import os
import matplotlib.pyplot as plt
# tensorboard test
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter("runs/fbnn_rbf")

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


# run toy example
x_train, y_train, x_test, y_test = data_generating_yacht(os.getcwd()+'/data/')
num_features = x_train.shape[-1]
lower_ap = np.minimum(np.min(x_train.numpy()), np.min(x_test.numpy()))
upper_ap = np.maximum(np.max(x_train.numpy()), np.max(x_test.numpy()))

def rand_generator(n_rand, n_dim = num_features,  minval=lower_ap, maxval=upper_ap):
    return torch.rand((n_rand, n_dim)) * (maxval - minval) + minval

# ls = median_distance_global(x_train).astype('float32')
# # ls[abs(ls) < 1e-6] = 1.
# kernel = gp.kernels.RBF(input_dim=num_features, variance=torch.tensor(1.), lengthscale=torch.tensor(ls))

likelihood = gpytorch.likelihoods.GaussianLikelihood()
gp_prior = GPPrior(x_train, y_train, likelihood)


mfnn = MeanfieldNNPosterior(num_features, args.num_nodes, args.num_layers)
gradient_estimator = SpectralScoreEstimator(eta=args.eta, n_eigen_threshold=args.n_eigen_threshold)

fvi = FunctionalVI(gp_prior, mfnn, rand_generator, gradient_estimator, args.n_rand, args.n_functions,
                   args.injected_noise, args.cuda)
fvi.build_prior_gp(x_train, y_train, args.learning_rate_gp, args.num_epoch_gp)
fvi.init_training(x_train, args.learning_rate_bnn, args.batch_size, args.num_epoch_bnn, args.coeff_ll,
                  args.coeff_kl)
# tensorboard test
# fvi.training(x_train, y_train, writer)
fvi.training(x_train, y_train)

mse_test, ll_test = fvi.build_evaluation(x_train, y_train)
print('Evaluation on training set: MSE={:.5f} | logLL={:.5f}'.format(mse_test, ll_test))

mse_test, ll_test = fvi.build_evaluation(x_test, y_test)
print('Evaluation on testing set: MSE={:.5f} | logLL={:.5f}'.format(mse_test, ll_test))

print(fvi.posterior.get_obs_var)

