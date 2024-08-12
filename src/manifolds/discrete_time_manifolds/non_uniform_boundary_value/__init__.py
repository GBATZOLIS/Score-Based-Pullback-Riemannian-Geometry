import torch
import torch.optim as optim

import matplotlib.pyplot as plt

from src.manifolds.discrete_time_manifolds import DiscreteTimeManifold

class NonUniformBoundaryValueDiscreteTimeManifold(DiscreteTimeManifold): # TODO we also need to overwrite the geodesic and log
    """ Base class describing Euclidean space of dimension d under a metric with non-uniform discrete time manifold mappings """

    def __init__(self, d, L1=100, tol1=1e-2, max_iter1=20000, step_size1=1/8, L2=200, tol2=1e-4, max_iter2=200):
        super().__init__(d, L1=L1, tol1=tol1, max_iter1=max_iter1, step_size1=step_size1, L2=L2, tol2=tol2, max_iter2=max_iter2)
    
    def boundary_value_discrete_geodesic(self, x, y): # TODO allow for initialisation Z0 + consider Newton for last steps
        """
        Use Riemannian gradient descent
        :param x: d
        :param y: N x d
        :return: N x L x d
        """
        N, _ = y.shape
        tau = torch.linspace(0.,1.,self.L1+1)
        Z = ((1 - tau[None,:,None]) * x[None,None] + tau[None,:,None] * y[:,None])[:,1:-1].requires_grad_()
        
        k = 0
        validation_0 = 0.
        while k < self.max_iter1: 
            # compute loss components
            losses = torch.zeros(N,self.L1)
            losses[:,0] = self.norm(x[None], Z[None,:,0] - x[None,None])**2
            losses[:,-1] = self.norm(Z[:,-1], y[:,None] - Z[:,-1,None])[:,0]**2
            losses[:,1:-1] = self.norm(Z[:,:-1].reshape(-1,self.d), Z[:,1:].reshape(-1,1,self.d) - Z[:,:-1].reshape(-1,1,self.d)).reshape(N,self.L1-2)**2
    
            # compute loss and validation
            loss = 1/2 * torch.sum(losses)
            validation = self.L1 * torch.sum(losses,1) - torch.sum(torch.sqrt(losses),1)**2
            if k == 0:
                validation_0 = validation.clone().max()

            if validation.max() / (validation_0 + 1e-8) < self.tol1:
                break

            # compute Riemannian gradients
            loss.backward()
            gradient = Z.grad
            inverse_metric_tensor_Z = self.inverse_metric_tensor(Z.reshape(-1,self.d)).reshape((N,self.L1-1,self.d,self.d))
            Riemannian_gradient_Z = torch.einsum("NLab,NLb->NLa", inverse_metric_tensor_Z, gradient)

            # update iterates
            with torch.no_grad():  # Disable gradient tracking for parameter updates
                Z -= self.step_size1 * Riemannian_gradient_Z
            Z.grad.zero_()  # Zero gradients manually
            
            if k % 1000 == 0: # geodesic discrepancy loss
                print(f"Epoch {k}, Loss {loss.item()} | Validation: {(validation.max().item())/ (validation_0.item() + 1e-8)}")

            k += 1

        return Z.detach()
    
    def initial_value_discrete_geodesic(self, x, X):
        """
        Use Newton Raphson for inner iteration
        :param x: d
        :param X: N x d
        :return: N x d
        """
        N, _ = X.shape

        y0 = x[None] * torch.ones(N)[:,None]
        y1 = x[None] + 1/self.L2 * X

        Z = torch.zeros(N,self.L2+1,self.d)
        Z[:,0] = y0
        Z[:,1] = y1

        for l in range(self.L2-1):
            # print(f"iteration {l+2}")
            y2 = y1.clone()

            y0 = y0.detach()
            y1.requires_grad_()
            y2.requires_grad_()

            k = 0
            error_0 = 0.
            while k < self.max_iter2: 
                # compute gradient and jacobian components
                metric_tensor_y0 = self.metric_tensor(y0)
                metric_tensor_y1 = self.metric_tensor(y1)
                gradient_metric_tensor_y1 = self.gradient_metric_tensor(y1)

                # compute gradient terms
                term_1 = torch.einsum("Nab,Nb->Na", metric_tensor_y1, y1 - y2)
                term_2 = 1/2 * torch.einsum("Ncba,Nc,Nb->Na", gradient_metric_tensor_y1, y1 - y2, y1 - y2)
                term_3 = torch.einsum("Nab,Nb->Na", metric_tensor_y0, y1 - y0)

                # compute full gradients
                Fy =  term_1 + term_2 + term_3
                if k == 0:
                    error_0 = torch.norm(Fy.clone(),2,-1).max()

                error = torch.norm(Fy,2,-1).max()
                if error / error_0 < self.tol2:
                    break
                
                # compute jacobian terms
                term_1_gradient_y2 = - metric_tensor_y1
                term_2_gradient_y2 = torch.einsum("Ncba,Nb->Nab", gradient_metric_tensor_y1, y2 - y1)

                #compute full jacobian
                J = term_1_gradient_y2 + term_2_gradient_y2

                # solve linear systems
                s = torch.linalg.solve(J, -Fy)

                # update y2
                y2 = y2 + s

                k += 1
            y0 = y1
            y1 = y2
            Z[:,l+2] = y2.detach()

        # DEBUG
        plt.scatter(Z[0,:,0], Z[0,:,1])
        plt.show()
        
        return y2
    
    