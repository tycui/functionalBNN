import torch
import numpy as np
import pyro.contrib.gp as gp
from bnn_posterior import MeanfieldNNPosterior
from fvi import FunctionalVI
from gradient_estimator import SpectralScoreEstimator
from toy_dataset import sin_toy
import argparse
import matplotlib.pyplot as plt

# tensorboard test
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter("runs/fbnn_rbf")

parser = argparse.ArgumentParser('Functional BNN on toy example')
parser.add_argument('-nn', '--num_nodes', type=int, default=100)
parser.add_argument('-nl', '--num_layers', type=int, default=2)
parser.add_argument('-no', '--n_rand', type=int, default=20)
parser.add_argument('-nf', '--n_functions', type=int, default=100)
parser.add_argument('-in', '--injected_noise', type=float, default=0.08)
parser.add_argument('-eta', type=float, default=0.)
parser.add_argument('--n_eigen_threshold', type=float, default=0.99)
parser.add_argument('--learning_rate_gp', type=float, default=0.003)
parser.add_argument('--num_epoch_gp', type=int, default=2500)
parser.add_argument('-lrnn', '--learning_rate_bnn', type=float, default=0.001)
parser.add_argument('-b', '--batch_size', type=int, default=50)
parser.add_argument('-enn', '--num_epoch_bnn', type=int, default=20000)
parser.add_argument('--coeff_ll', type=float, default=1.)
parser.add_argument('--coeff_kl', type=float, default=1.)
parser.add_argument('--cuda', type=bool, default=True)

group = parser.add_mutually_exclusive_group()
group.add_argument('-rbf', '--RBF_Kernel', action='store_true')
group.add_argument('-periodic', '--Periodic_Kernel', action='store_false')

args = parser.parse_args()


def rand_generator(n_rand):
    return torch.rand((n_rand, 1)) * 10. - 5.


# run toy example
toy_data = sin_toy()
x_train, y_train = toy_data.train_samples()
x_test, y_test = toy_data.test_samples()
num_features = x_train.shape[-1]

if args.RBF_Kernel:
    print('Using a RBF kernel Gaussian process as a prior')
    kernel_type = 'RBF'
    kernel = gp.kernels.RBF(input_dim=num_features, variance=torch.tensor(1.), lengthscale=torch.tensor(1.))
elif args.Periodic_Kernel:
    print('Using a periodic kernel Gaussian process as a prior')
    kernel_type = 'Periodic'
    kernel = gp.kernels.Periodic(input_dim=num_features, variance=torch.tensor(1.), lengthscale=torch.tensor(1.),
                                 period=torch.tensor(2.))

mfnn = MeanfieldNNPosterior(num_features, args.num_nodes, args.num_layers)
gradient_estimator = SpectralScoreEstimator(eta=args.eta, n_eigen_threshold=args.n_eigen_threshold)
obs_var = toy_data.y_std ** 2

fvi = FunctionalVI(kernel, mfnn, rand_generator, gradient_estimator, obs_var, args.n_rand, args.n_functions,
                   args.injected_noise, args.cuda)

fvi.build_prior_gp(x_train, y_train, args.learning_rate_gp, args.num_epoch_gp)

fvi.init_training(x_train, args.learning_rate_bnn, args.batch_size, args.num_epoch_bnn, args.coeff_ll,
                  args.coeff_kl)

# tensorboard test
# fvi.training(x_train, y_train, writer)
fvi.training(x_train, y_train)

# Plot
fvi.posterior.eval()
predictive_posterior = []
for i in range(2000):
    fvi.posterior.train()
    y_pred = fvi.posterior(x_test).detach().view(-1).numpy()
    predictive_posterior.append(y_pred)
pred_std = np.std(predictive_posterior, axis=0) + toy_data.y_std
pred_mean = np.mean(predictive_posterior, axis=0)
del predictive_posterior
x_test_plot = x_test.view(-1).numpy()
fig, ax = plt.subplots(1, 1)
ax.plot(x_train, y_train, 'g*', markersize=8)
ax.plot(x_test, y_test, 'r-', markersize=8)
ax.plot(x_test, pred_mean, 'b-', markersize=8)

for i in range(5):
    plt.fill_between(x_test_plot, pred_mean - i * 0.75 * pred_std,
                     pred_mean - (i + 1) * 0.75 * pred_std, linewidth=0.0,
                     alpha=1.0 - i * 0.15, color='lightblue')
    plt.fill_between(x_test_plot, pred_mean + i * 0.75 * pred_std,
                     pred_mean - (i + 1) * 0.75 * pred_std, linewidth=0.0,
                     alpha=1.0 - i * 0.15, color='lightblue')

plt.legend(['train', 'test', 'predicted mean'])
plt.show()
# plt.savefig('FBNN-' + kernel_type + '.pdf')
