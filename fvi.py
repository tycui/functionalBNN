import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pyro
import pyro.contrib.gp as gp
import torch.distributions as distributions
from torch.utils.data import TensorDataset, DataLoader


class FunctionalVI(object):
    def __init__(self, prior_kernel, posterior, rand_generator, stein_estimator, obs_var=0.1, n_oodsamples=50,
                 n_functions=20, injected_noise=0.01):
        self.prior_kernel = prior_kernel
        self.posterior = posterior
        self._rand_generator = rand_generator
        self.stein_estimator = stein_estimator
        self.n_oodsamples = n_oodsamples
        self.n_functions = n_functions
        self.injected_noise = injected_noise
        self.obs_var = obs_var

    def build_function(self, x_random, noise_level=None):
        if noise_level != None:
            func_x_random = self.posterior.forward_multiple(x_random, self.n_functions)
            func_x_random = func_x_random + noise_level * torch.randn_like(func_x_random)
        else:
            func_x_random = self.posterior.forward_multiple(x_random, self.n_functions)

        return func_x_random

    def build_prior_gp(self, x, y, lr_gp, num_steps):
        """
        Optimizing the hyper-parameters of GP prior
        :param x: inputs of training data
        :param y: targets of training data
        :return: NULL
        """
        print('Optimizing the GP prior')
        gpr = gp.models.GPRegression(x, y, self.prior_kernel, noise=torch.tensor(.1))
        optimizer = optim.Adam(gpr.parameters(), lr=lr_gp)
        loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
        for i in range(num_steps):
            optimizer.zero_grad()
            loss = loss_fn(gpr.model, gpr.guide)
            loss.backward()
            optimizer.step()
            if i % int(num_steps / 10) == 0:
                print('>>> Epoch {:5d}/{:5d} | loss={:.5f} '.format(i,
                                                                    num_steps,
                                                                    loss.item()))

    def build_kl(self, x_batch):
        """
        Compute the KL surrogate
        Note: the surrogate might be negative as the entropy surrogate is not the entropy itself.
        """
        # generate ood data points
        x_random = self._rand_generator(self.n_oodsamples)
        x_kl = torch.cat([x_batch, x_random], axis=0)
        # estimate entropy surrogate
        func_x_random = self.build_function(x_kl, self.injected_noise)
        entropy_sur = self.stein_estimator.entropy_surrogate(func_x_random)
        # compute the analytical cross entropy
        kernel_matrix = self.prior_kernel(x_kl) + self.injected_noise ** 2 * torch.eye(x_kl.shape[0])
        prior_dist = distributions.MultivariateNormal(torch.zeros(x_kl.shape[0]), kernel_matrix)
        cross_entropy = -torch.mean(prior_dist.log_prob(func_x_random))
        self.kl_surrogate = -entropy_sur + cross_entropy

        return self.kl_surrogate

    def build_log_likelihood(self, x_batch, y_batch):
        ## TO DO: compute likelihood
        criterion = nn.MSELoss()
        self.log_likelihood = -criterion(self.posterior(x_batch), y_batch) / (2. * self.obs_var)
        return self.log_likelihood

    def init_training(self, x_train, learning_rate=0.001, batch_size=50, num_epoch=1000, coeff_ll=1., coeff_kl=1.):
        self.num_training, self.num_dim = x_train.shape
        self.learning_rate = learning_rate

        if batch_size < int(self.num_training / 10):
            self.batch_size = batch_size
        else:
            self.batch_size = self.num_training

        self.num_epoch = num_epoch
        self.coeff_ll = coeff_ll
        self.coeff_kl = coeff_kl

    def training(self, x, y, writer = None):
        posterior_parameters = set(self.posterior.parameters())
        optimizer = optim.Adam(posterior_parameters, lr=self.learning_rate, eps=1e-3)
        train_dl = DataLoader(TensorDataset(x, y), batch_size=self.batch_size, shuffle=True)
        print('Inference BNN posterior:')

        # tensorboard test
        # running_ll = 0.0

        for epoch in range(self.num_epoch):
            for x_batch, y_batch in train_dl:
                optimizer.zero_grad()
                self.posterior.train()
                # calculate the training loss
                ll = self.build_log_likelihood(x_batch, y_batch.view(-1,1))
                kl = self.build_kl(x_batch)
                self.elbo = self.coeff_ll * ll - self.coeff_kl * kl / self.batch_size
                # backpropogate the gradient
                (-self.elbo).backward()
                # optimize with SGD
                optimizer.step()

            # # tensorboard test
            # running_ll += ll.item()

            if epoch % int(self.num_epoch / 10) == 0:
                print('>>> Epoch {:5d}/{:5d} | elbo_sur={:.5f} | logLL={:.5f} | kl_sur={:.5f}'.format(epoch,
                                                                                                      self.num_epoch,
                                                                                                      self.elbo,
                                                                                                      self.log_likelihood,
                                                                                                      self.kl_surrogate))
            # # tensorboard test
            # if epoch % 10 == 0:
            #     writer.add_scalar('log likelihood', running_ll / 10, epoch)
            #     k = 1
            #     for param in self.posterior.parameters():
            #         writer.add_scalar('parameter ' + str(k), np.mean(param.detach().numpy()), epoch)
            #         writer.add_scalar('gradient' + str(k), np.mean(param.grad.numpy()), epoch)
            #         k+=1
            #     running_ll = 0