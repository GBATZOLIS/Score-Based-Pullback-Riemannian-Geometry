import torch

from src.manifolds.discrete_time_manifolds.sum_of_gaussian import SumOfGaussian

class SumOfDiagonal(SumOfGaussian):
    def __init__(self, d, strongly_convexs, weights):
        super().__init__(d, strongly_convexs, weights)

    
    def metric_tensor(self, x): 
        """
        :param x: N x d
        :return: N x d x d 
        """
        return torch.diag_embed(self.metric_tensor_diagonal(x))
        
    def inverse_metric_tensor(self, x):
        """
        :param x: N x d
        :return: N x d x d 
        """
        return torch.diag_embed(1 / self.metric_tensor_diagonal(x))
    
    def metric_tensor_diagonal(self, x):
        """
        :param x: N x d
        :return: N x d 
        """
        N, _ = x.shape
        psi_x = torch.zeros(N,self.m)
        for i in range(self.m):
            psi_x[:,i] = self.psi[i].forward(x)
        softmax_psi_x = (- psi_x + torch.log(self.weights[None] + 1e-8)).softmax(1)
        # return torch.sum((softmax_psi_x[:,:,None] / self.diagonals[None]**2), 1)
        return torch.sum((softmax_psi_x[:,:,None] / self.diagonals[None])[:,:,None,:] * (softmax_psi_x[:,:,None] / self.diagonals[None])[:,None,:,:],[1,2])