import gpytorch
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
## the periodic kernel of gpytorch can provide inconsistent results with pyro

class LinearRegression(nn.Module):
    """
    A trained linear regression model as informative mean function of GP
    """
    def __init__(self, input_dim):
        super(LinearRegression, self).__init__()
        self.input_dim = input_dim

        scale = 1. * np.sqrt(6. / (input_dim + 1))
        # approximated posterior
        self.w = nn.Parameter(torch.Tensor(self.input_dim, 1).uniform_(-scale, scale))
        self.bias = nn.Parameter(torch.Tensor(1).uniform_(-scale, scale))

    def forward(self, x):
        y_pred = torch.mm(x, self.w) + self.bias
        return torch.mean(y_pred, 1)

    def train_lr(self, x, y, learning_rate, num_steps):
        criterion = nn.MSELoss()
        parameters = set(self.parameters())
        optimizer = optim.Adam(parameters, lr=learning_rate, eps=1e-3)

        for i in range(num_steps):
            optimizer.zero_grad()
            output = self.forward(x)
            loss = criterion(output, y.view(-1))
            loss.backward()
            optimizer.step()
            if i % int(num_steps / 10) == 0:
                print('>>> Epoch {:5d}/{:5d} | loss={:.5f} '.format(i,
                                                                    num_steps,
                                                                    loss.item()))

    def _detach(self):
        self.w.requires_grad_(False)
        self.bias.requires_grad_(False)
        print('Parameters are detached from the computational graph.')

class GPPrior(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, mean_function=None, kernel=None):
        super(GPPrior, self).__init__(train_x, train_y, likelihood)
        self.likelihood = likelihood
        if mean_function == None:
            self.mean_module = gpytorch.means.ZeroMean()
        else:
            self.mean_module = mean_function

        if kernel == None:
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        else:
            self.covar_module = gpytorch.kernels.ScaleKernel(kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)