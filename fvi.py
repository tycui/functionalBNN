import torch
import torch.nn as nn
import torch.optim as optim
import pyro
import pyro.contrib.gp as gp
import torch.distributions as distributions
from torch.utils.data import TensorDataset, DataLoader


class FunctionalVI(object):
    def __init__(self, prior_kernel, posterior, rand_generator, stein_estimator, n_oodsamples=50,
                 n_functions=20, injected_noise=0.01, use_cuda = False):
        self.prior_kernel = prior_kernel
        self.posterior = posterior
        self._rand_generator = rand_generator
        self.stein_estimator = stein_estimator
        self.n_oodsamples = n_oodsamples
        self.n_functions = n_functions
        self.injected_noise = injected_noise
        if use_cuda:
            self.posterior.cuda()
        self.use_cuda = use_cuda

    def build_function(self, x_random, noise_level=None):
        if noise_level != None:
            func_x_random = self.posterior.forward_multiple(x_random, self.n_functions)
            func_x_random = func_x_random + noise_level * torch.randn_like(func_x_random).to(x_random.device)
        else:
            func_x_random = self.posterior.forward_multiple(x_random, self.n_functions)

        return func_x_random

    def build_prior_gp(self, x, y, lr_gp, num_steps):
        """
        Optimizing the hyper-parameters of GP prior
        :param x: inputs of training data
        :param y: targets of training data
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
        x_random = self._rand_generator(self.n_oodsamples).to(x_batch.device)
        x_kl = torch.cat([x_batch, x_random], axis=0)
        # estimate entropy surrogate
        func_x_random = self.build_function(x_kl, self.injected_noise)
        entropy_sur = self.stein_estimator.entropy_surrogate(func_x_random)
        # compute the analytical cross entropy
        kernel_matrix = self.prior_kernel(x_kl) + self.injected_noise ** 2 * torch.eye(x_kl.shape[0]).to(x_batch.device)
        prior_dist = distributions.MultivariateNormal(torch.zeros(x_kl.shape[0]).to(x_batch.device), kernel_matrix)
        cross_entropy = -torch.mean(prior_dist.log_prob(func_x_random))
        self.kl_surrogate = -entropy_sur + cross_entropy
        return self.kl_surrogate

    def build_log_likelihood(self, x_batch, y_batch):
        criterion = nn.MSELoss()
        self.log_likelihood = -criterion(self.posterior(x_batch), y_batch) / (2. * self.posterior.get_obs_var) - 0.5 * torch.log(self.posterior.get_obs_var)
        return self.log_likelihood

    def build_evaluation(self, x_test, y_test):
        """
        compute the log likelihood and MSE on testset
        """
        if self.use_cuda:
            x_test = x_test.cuda()
            y_test = y_test.cuda()
        y_pred = self.posterior.forward_multiple(x_test, self.n_functions)
        self.eval_rmse = torch.sqrt(torch.mean((torch.mean(y_pred, 0) - y_test.view(-1)) ** 2)).detach()
        log_likelihood_samples = -(y_pred - y_test.view(-1)) ** 2 / (2. * self.posterior.get_obs_var) - 0.5 * torch.log(self.posterior.get_obs_var)
        self.eval_ll = torch.mean(torch.logsumexp(log_likelihood_samples, 0) - torch.log(torch.tensor(self.n_functions).float())).detach()

        return self.eval_rmse, self.eval_ll

    def init_training(self, x_train, learning_rate=0.001, batch_size=50, num_epoch=1000, coeff_ll=1., coeff_kl=1.):
        """
        Initialize the training hyper-parameters
        """
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
        """
        Learning BNN posterior with stochastic variational inference
        """
        posterior_parameters = set(self.posterior.parameters())
        optimizer = optim.Adam(posterior_parameters, lr=self.learning_rate, eps=1e-3)
        train_dl = DataLoader(TensorDataset(x, y), batch_size=self.batch_size, shuffle=True)
        print('Inference BNN posterior:')

        # tensorboard test
        # running_ll = 0.0

        for epoch in range(self.num_epoch):
            for x_batch, y_batch in train_dl:
                if self.use_cuda:
                    x_batch = x_batch.cuda()
                    y_batch = y_batch.cuda()
                optimizer.zero_grad()
                self.posterior.train()
                # calculate the training loss
                ll = self.build_log_likelihood(x_batch, y_batch.view(-1,1))
                kl = self.build_kl(x_batch) + self.posterior.get_kl_prior.to(x_batch.device)
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
                                                                                                      self.elbo.item(),
                                                                                                      self.log_likelihood.item(),
                                                                                                      self.kl_surrogate))

        # Additional Info when using cuda
        if self.use_cuda:
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**2,1), 'MB')
            print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**2,1), 'MB')

            # # tensorboard test
            # if epoch % 10 == 0:
            #     writer.add_scalar('log likelihood', running_ll / 10, epoch)
            #     k = 1
            #     for param in self.posterior.parameters():
            #         writer.add_scalar('parameter ' + str(k), np.mean(param.detach().numpy()), epoch)
            #         writer.add_scalar('gradient' + str(k), np.mean(param.grad.numpy()), epoch)
            #         k+=1
            #     running_ll = 0