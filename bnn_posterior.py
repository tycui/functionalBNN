import torch
import torch.nn as nn
import numpy as np

class Meanfieldlayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Meanfieldlayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        scale = 1. * np.sqrt(6. / (input_dim + output_dim))
        # approximated posterior
        self.mu_beta = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim).uniform_(-scale, scale))
        self.rho_beta = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim).uniform_(-4, -2))

        self.mu_bias = nn.Parameter(torch.Tensor(1, self.output_dim).uniform_(-scale, scale))
        self.rho_bias = nn.Parameter(torch.Tensor(1, self.output_dim).uniform_(-4, -2))

    def forward(self, x, training):
        sigma_beta = torch.log(1 + torch.exp(self.rho_beta))
        sigma_bias = torch.log(1 + torch.exp(self.rho_bias))
        if training:
            # forward passing with stochastic via local reparametrization trick
            mean_output = torch.matmul(x, self.mu_beta) + self.mu_bias
            sigma_output = torch.sqrt(torch.mm(x ** 2, sigma_beta ** 2) + sigma_bias ** 2)

            epsilon = torch.randn(x.shape[-2], self.output_dim)
            output = mean_output + sigma_output * epsilon

            return output
        else:
            output = torch.matmul(x, self.mu_beta) + self.mu_bias

            return output

    def forward_multiple(self, x, n_functions):
        epsilon_beta = torch.randn(n_functions, self.input_dim, self.output_dim)
        epsilon_bias = torch.randn(n_functions, 1, self.output_dim)

        sigma_beta = torch.log(1 + torch.exp(self.rho_beta))
        sigma_bias = torch.log(1 + torch.exp(self.rho_bias))

        beta = self.mu_beta + sigma_beta * epsilon_beta
        bias = self.mu_bias + sigma_bias * epsilon_bias

        output = torch.matmul(x, beta) + bias
        return output

class MeanfieldNNPosterior(nn.Module):
    def __init__(self, num_feature, num_nodes, num_layers):
        super(MeanfieldNNPosterior, self).__init__()
        self.num_feature = num_feature
        self.num_layers = num_layers
        self.num_nodes = num_nodes

        list_of_layers = [Meanfieldlayer(num_feature, num_nodes)]
        list_of_layers += [Meanfieldlayer(num_nodes, num_nodes) for i in range(num_layers-1)]
        list_of_layers += [Meanfieldlayer(num_nodes, 1)]

        self.layers = nn.ModuleList(list_of_layers)
        self.activation_fn = nn.ReLU()

    def forward(self, x):
        for i, j in enumerate(self.layers):
            if i != self.num_layers:
                x = self.activation_fn(j(x, self.training))
            else:
                x = j(x, self.training)
        return x

    def forward_multiple(self, x, n_functions):
        for i, j in enumerate(self.layers):
            if i != self.num_layers:
                x = self.activation_fn(j.forward_multiple(x, n_functions))
            else:
                x = j.forward_multiple(x, n_functions)
        return torch.mean(x, -1)
