import torch
class ScoreEstimator(object):
    """
    Base class of gradient estimator
    """

    def __init__(self):
        pass

    def rbf_kernel(self, x1, x2, kernel_width):
        kernel = torch.exp(-torch.sum((x1 - x2) ** 2, dim=-1) / (2. * (kernel_width ** 2)))
        return kernel

    def gram(self, x1, x2, kernel_width):
        """
        Return the kernel matrix of x1 and x2 with RBF kernel.
        """
        x_row = x1.unsqueeze(-2)  # x_row = [n1, 1, x_dim]
        x_col = x2.unsqueeze(-3)  # x_col = [1, n2, x_dim]
        gram_matrix = self.rbf_kernel(x_row, x_col, kernel_width)
        return gram_matrix

    def grad_gram(self, x1, x2, kernel_width):
        """
        Return the kernel matrix of x1 and x2 with RBF kernel, and the gradient of kernel function w.r.t. x_1 and x_2
        """
        x_row = x1.unsqueeze(-2)  # x_row = [n1, 1, x_dim]
        x_col = x2.unsqueeze(-3)  # x_col = [1, n2, x_dim]
        G = self.rbf_kernel(x_row, x_col, kernel_width)  # G = [n_1, n_2]
        diff = (x_row - x_col) / (kernel_width ** 2)  # diff = [n_1, n_2, x_dim]
        G_expand = G.unsqueeze(-1)  # G = [n_1, n_2, 1]
        grad_x2 = G_expand * diff  # grad_x2 = [n1, n2, x_dim]
        grad_x1 = G_expand * (-diff)  # grad_x1 = [n1, n2, x_dim]
        return G, grad_x1, grad_x2

    def heuristic_kernel_width(self, x1, x2):
        """
        Return the median of pairwise distance as kernel width
        """
        x_row = x1.unsqueeze(-2)  # x_row = [n1, 1, x_dim]
        x_col = x2.unsqueeze(-3)  # x_col = [1, n2, x_dim]
        pairwise_distance = torch.sqrt(torch.sum((x_row - x_col) ** 2, dim=-1))  # list of pairwise distance
        kernel_width = torch.median(
            pairwise_distance).detach()  # set the median of pairwise distance as the kernel width
        return kernel_width

    def entropy_surrogate(self, samples):
        """
        Compute the entropy of q on samples
        """
        dlog_q = self.compute_gradients(samples)
        surrogate = torch.mean(
            torch.sum(-dlog_q.detach() * samples, -1))
        return surrogate

    def compute_gradients(self, samples, x=None):
        raise NotImplementedError()


class SteinScoreEstimator(ScoreEstimator):
    def __init__(self, eta):
        self._eta = eta
        super(SteinScoreEstimator, self).__init__()

    def compute_gradients(self, samples, x=None):
        M = samples.shape[0]
        kernel_width = self.heuristic_kernel_width(samples, samples)
        K, grad_K1, grad_K2 = self.grad_gram(samples, samples, kernel_width)
        Kinv = torch.inverse(K + self._eta * torch.eye(M))
        H_dh = torch.sum(grad_K2, dim=-2)
        grads = -torch.matmul(Kinv, H_dh)
        if x is None:
            return grads
        else:
            grads_x = []
            for x_i in x:
                x_i = x_i.unsqueeze(-2)
                Kxx = self.gram(x_i, x_i, kernel_width)
                Kxq = self.gram(x_i, samples, kernel_width)

                KxqKinv = torch.matmul(Kxq, Kinv)
                term1 = -1. / (Kxx + self._eta - torch.matmul(KxqKinv, torch.transpose(Kxq, 0, 1)))

                Kqx, grad_Kqx1, grad_Kqx2 = self.grad_gram(samples, x_i, kernel_width)

                term2 = torch.matmul(Kxq, grads) - torch.matmul(KxqKinv + 1., torch.squeeze(grad_Kqx2, -2))
                grads_x.append(torch.matmul(term1, term2))

            return torch.cat(grads_x, dim=0)


class SpectralScoreEstimator(ScoreEstimator):
    """
    Spectral Stein gradient estimator
    """

    def __init__(self, n_eigen=None, eta=None, n_eigen_threshold=None):
        self._n_eigen = n_eigen
        self._eta = eta
        self._n_eigen_threshold = n_eigen_threshold
        super(SpectralScoreEstimator, self).__init__()

    def nystrom_ext(self, samples, x, eigen_vectors, eigen_values, kernel_width):
        """
        Compute the eigenfunction of from eigenvectors and eigenvalues with Nystrom method
        """
        # samples: [M, x_dim]
        # x: [N, x_dim]
        # eigen_vectors: [M, n_eigen]
        # eigen_values: [ n_eigen]
        # return: [ N, n_eigen], by default n_eigen=M.
        M = samples.shape[0]
        # Kxq: [..., N, M]
        # grad_Kx: [..., N, M, x_dim]
        # grad_Kq: [..., N, M, x_dim]
        Kxq = self.gram(x, samples, kernel_width)
        # Kxq = tf.Print(Kxq, [tf.shape(Kxq)], message="Kxq:")
        # ret: [..., N, n_eigen]
        ret = torch.sqrt(torch.tensor(M).float()) * torch.matmul(Kxq, eigen_vectors)
        ret = ret / eigen_values
        return ret

    def compute_gradients(self, samples, x=None):
        """
        Compute the gradients of log q(x), given samples~q(x);
        If x=None, compute the gradients of log q(samples).
        """
        if x is None:
            kernel_width = self.heuristic_kernel_width(samples, samples)
            x = samples
        else:
            # _samples: [..., N + M, x_dim]
            _samples = torch.cat((samples, x), dim=-2)
            kernel_width = self.heuristic_kernel_width(_samples, _samples)

        M = samples.shape[0]
        # Kq: [..., M, M]
        # grad_K1: [..., M, M, x_dim]
        # grad_K2: [..., M, M, x_dim]
        Kq, grad_K1, grad_K2 = self.grad_gram(samples, samples, kernel_width)
        if self._eta is not None:
            Kq += self._eta * torch.eye(M)
        # eigen_vectors: [..., M, M]
        # eigen_values: [..., M]
        eigen_values, eigen_vectors = torch.symeig(Kq, eigenvectors=True)

        if (self._n_eigen is None) and (self._n_eigen_threshold is not None):
            eigen_arr = torch.mean(torch.fliplr(eigen_values.view(1, -1)), axis=0)
            eigen_arr /= torch.sum(eigen_arr)
            eigen_cum = torch.cumsum(eigen_arr, dim=0)
            self._n_eigen = torch.sum(torch.lt(eigen_cum, self._n_eigen_threshold))

        if self._n_eigen is not None:
            # eigen_values: [..., n_eigen]
            # eigen_vectors: [..., M, n_eigen]
            eigen_values = eigen_values[..., -self._n_eigen:]
            eigen_vectors = eigen_vectors[..., -self._n_eigen:]
        # eigen_ext: [..., N, n_eigen]
        eigen_ext = self.nystrom_ext(samples, x, eigen_vectors, eigen_values, kernel_width)

        grad_K1_avg = torch.mean(grad_K1, dim=-3)
        beta = -torch.sqrt(torch.tensor(M).float()) * torch.matmul(torch.transpose(eigen_vectors, 0, 1),
                                                                   grad_K1_avg) / eigen_values.view(self._n_eigen, 1)
        #         print(eigen_ext)
        #         print(beta)
        grads = torch.matmul(eigen_ext, beta)

        return grads