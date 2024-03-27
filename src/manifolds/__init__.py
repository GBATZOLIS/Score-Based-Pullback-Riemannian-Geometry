import torch

class Manifold: # TODO
    """ Base class describing a manifold of dimension `ndim` """

    def __init__(self, d):
        self.d = d

    def barycentre(self, x, tol=1e-3, max_iter=50, initialisation=None):
        """

        :param x: N x Mpoint
        :return: Mpoint
        """
        k = 0
        if initialisation is None:
            y = x[:,0]
        else:
            y = initialisation * torch.ones(x.shape[0],1)
        error = self.norm(y, torch.mean(self.log(y, x),1).unsqueeze(-2))
        rel_error = 1.
        while k <= max_iter and rel_error >= tol:
            y = self.exp(y, torch.mean(self.log(y, x),1).unsqueeze(-2)).squeeze(-2)
            k+=1
            rel_error = self.norm(y, torch.mean(self.log(y, x),1).unsqueeze(-2)) / error

        print(f"gradient descent was terminated after reaching a relative error {rel_error.item()} in {k} iterations")

        return y

    def inner(self, x, X, Y):
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def norm(self, x, X):
        """

        :param x: N x Mpoint
        :param X: N x M x Mpoint
        :return: N x M
        """
        return torch.sqrt(torch.abs(self.inner(p.unsqueeze(-2) * torch.ones((1, X.shape[-2], 1)),
                                     X.unsqueeze(-2), X.unsqueeze(-2)).squeeze(-2)))

    def distance(self, p, q):
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def log(self, x, y):
        """

        :param x: N x Mpoint
        :param y: N x M x Mpoint
        :return: N x M x Mpoint
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def exp(self, x, X):
        """

        :param x: N x Mpoint
        :param X: N x M x Mpoint
        :return: N x M x Mpoint
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def geodesic(self, x, y, t):
        """

        :param x: 1 x Mpoint
        :param y: 1 x Mpoint
        :param t: N
        :return: N x Mpoint
        """
        assert p.shape[0] == q.shape[0] == 1
        assert len(t.shape) == 1
        return self.exp(p, t.unsqueeze(0).unsqueeze(2) * self.log(p, q.unsqueeze(1)))[0]

    def parallel_transport(self, x, X, y):
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def manifold_dimension(self):
        return self.d