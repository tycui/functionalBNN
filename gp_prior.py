import gpytorch
## the periodic kernel of gpytorch can provide inconsistent results with pyro

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