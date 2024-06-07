import torch
import torch.optim as optim

import matplotlib.pyplot as plt

from src.manifolds import Manifold

class DiscreteTimeManifold(Manifold):
    """ Base class describing Euclidean space of dimension d under a metric with discrete time manifold mappings """

    def __init__(self, d):
        super().__init__(d)

    def metric_tensor(self, x):
        """
        g_ab
        :param x: N x d
        :return: N x d x d
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )
    
    def inverse_metric_tensor(self, x):
        """
        g^ab
        :param x: N x d
        :return: N x d x d
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )
    
    def gradient_metric_tensor(self, x):
        """
        g_ab;c
        :param x: N x d
        :return: N x d x d x d
        """
        def sum_metric_tensor(y):
            return torch.sum(self.metric_tensor(y),0)
        
        return torch.autograd.functional.jacobian(sum_metric_tensor, x).permute((2,0,1,3))
    
    def christoffel_symbols(self,x):
        """
        G^c_ab
        :param x: N x d
        :return: N x d x d x d 
        """
        inverse_metric_tensor_x = self.inverse_metric_tensor(x)
        gradient_metric_tensor_x = self.gradient_metric_tensor(x)

        term_1 = torch.einsum("Ncd,Ndab->Nabc",inverse_metric_tensor_x, gradient_metric_tensor_x)
        term_2 = torch.einsum("Ncd,Ndba->Nabc",inverse_metric_tensor_x, gradient_metric_tensor_x)
        term_3 = torch.einsum("Ncd,Nabd->Nabc",inverse_metric_tensor_x, gradient_metric_tensor_x)
        return 1/2 * (term_1 + term_2 - term_3)
    
    def curvature_tensor(self,x): # TODO
        """
        R^d_cab
        :param x: N x d
        :return: N x d x d x d x d
        """
        def sum_christoffel_symbol(y):
            return torch.sum(self.christoffel_symbols(y),0)
        
        christoffel_symbol_gradient_x = torch.autograd.functional.jacobian(sum_christoffel_symbol, x).permute((3,4,0,1,2))
        christoffel_symbols_x = self.christoffel_symbols(x)
        christoffel_symbol_product_x = torch.einsum("Naed,Nbce->Nabcd", christoffel_symbols_x, christoffel_symbols_x)

        term_1 = christoffel_symbol_gradient_x
        term_2 = christoffel_symbol_gradient_x.permute(0,2,1,3,4)
        term_3 = christoffel_symbol_product_x
        term_4 = christoffel_symbol_product_x.permute(0,2,1,3,4)

        return term_1 - term_2 + term_3 - term_4
    
    def ricci_tensor(self,x):
        """
        R_ab
        :param x: N x d
        :return: N x d x d 
        """
        curvature_tensor_x = self.curvature_tensor(x)
        return torch.einsum("Ncabc->Nab", curvature_tensor_x)
    
    def ricci_scalar(self,x):
        """
        R
        :param x: N x d
        :return: N 
        """
        ricci_tensor_x = self.ricci_tensor(x)
        return torch.einsum("Naa->N", ricci_tensor_x)

    def barycentre(self, x, tol=1e-2, max_iter=50, step_size=1/4): # TODO allow to recycle discrete geodesics
        """

        :param x: N x d
        :return: d
        """
        k = 0
        y = torch.mean(x,0)
        
        gradient_0 = torch.mean(self.log(y, x, tol=0.1),0)
        error = self.norm(y[None], gradient_0[None,None])
        rel_error = 1.
        while k <= max_iter and rel_error >= tol:
            gradient = torch.mean(self.log(y, x, tol=0.1),0)
            y = y + step_size * gradient
            k+=1
            rel_error = self.norm(y[None], gradient[None,None]) / error
            print(f"iteration {k} | rel_error = {rel_error.item()}")

        print(f"gradient descent was terminated after reaching a relative error {rel_error.item()} in {k} iterations")

        return y
    
    def norm(self, x, X):
        """

        :param x: N x d
        :param X: N x M x d
        :return: N x M
        """
        metric_tensor_x = self.metric_tensor(x)
        return torch.sqrt(torch.einsum("Nab,NMa,NMb->NM", metric_tensor_x, X, X) + 1e-8)
    
    def inner(self, x, X, Y):
        """

        :param x: N x d
        :param X: N x M x d
        :param Y: N x L x d
        :return: N x M x L
        """
        metric_tensor_x = self.metric_tensor(x)
        return torch.einsum("Nab,NMa,NLb->NML", metric_tensor_x, X, Y)

    
    def geodesic(self, x, y, t, L=100, tol=1e-2, max_iter=20000, step_size=1/8):
        """

        :param x: d
        :param y: d
        :param t: N
        :return: N x d
        """
        N = t.shape[0]

        # compute discrete geodesics 
        Z = self.discrete_geodesic(x, y[None], L=L, tol=tol, max_iter=max_iter, step_size=step_size)[0]

        plt.scatter(Z[:,0], Z[:,1])
        plt.show()

        # compute continuous time geodesics
        Y = torch.zeros((N,self.d))
        for i,tau in enumerate(t):
            if tau < 1/L:
                Y[i] = tau * (Z[0] - x) + x
            elif tau >= (L-1)/L:
                Y[i] = (1 - (1 - tau) * L) * (y - Z[-1]) + Z[-1]
            else:
                j = torch.floor(tau * L).int()
                Y[i] = (tau * L - j) * (Z[j] - Z[j-1]) + Z[j-1]

        return  Y

    def log(self, x, y, L=100, tol=1e-2, max_iter=20000, step_size=1/8):
        """

        :param x: d
        :param y: N x d
        :return: N x d
        """
        # compute discrete geodesics 
        Z = self.discrete_geodesic(x, y, L=L, tol=tol, max_iter=max_iter, step_size=step_size)

        # DEBUG
        N, _ = y.shape
        for i in range(N):
            plt.scatter(Z[i,:,0], Z[i,:,1])
        plt.show()

        return L * (Z[:,0] - x[None])
    
    def discrete_geodesic(self, x, y,  L=100, tol=1e-2, max_iter=20000, step_size=1/8): # TODO allow for initialisation Z0 + consider Newton for last steps
        """

        :param x: d
        :param y: N x d
        :return: N x L x d
        """
        N, _ = y.shape
        tau = torch.linspace(0.,1.,L+1)
        Z = ((1 - tau[None,:,None]) * x[None,None] + tau[None,:,None] * y[:,None])[:,1:-1].requires_grad_()
        
        k = 0
        validation_0 = 0.
        while k < max_iter: 
            # compute loss components
            losses = torch.zeros(N,L)
            losses[:,0] = self.norm(x[None], Z[None,:,0] - x[None,None])**2
            losses[:,-1] = self.norm(Z[:,-1], y[:,None] - Z[:,-1,None])[:,0]**2
            losses[:,1:-1] = self.norm(Z[:,:-1].reshape(-1,self.d), Z[:,1:].reshape(-1,1,self.d) - Z[:,:-1].reshape(-1,1,self.d)).reshape(N,L-2)**2
    
            # compute loss and validation
            loss = 1/2 * torch.sum(losses)
            validation = L * torch.sum(losses,1) - torch.sum(torch.sqrt(losses),1)**2
            if k == 0:
                validation_0 = validation.clone().max()

            if validation.max() / (validation_0 + 1e-8) < tol:
                break

            # compute Riemannian gradients
            loss.backward()
            gradient = Z.grad
            inverse_metric_tensor_Z = self.inverse_metric_tensor(Z.reshape(-1,self.d)).reshape((N,L-1,self.d,self.d))
            Riemannian_gradient_Z = torch.einsum("NLab,NLb->NLa", inverse_metric_tensor_Z, gradient)

            # update iterates
            with torch.no_grad():  # Disable gradient tracking for parameter updates
                Z -= step_size * Riemannian_gradient_Z
            Z.grad.zero_()  # Zero gradients manually
            
            if k % 1000 == 0: # geodesic discrepancy loss
                print(f"Epoch {k}, Loss {loss.item()} | Validation: {(validation.max().item())/ (validation_0.item() + 1e-8)}")

            k += 1

        return Z.detach()

    def exp(self, x, X, L=100, tol=1e-5, max_iter=50):
        """
        Use Newton Raphson for inner iteration
        :param x: d
        :param X: N x d
        :return: N x d
        """
        N, _ = X.shape
        y0 = x[None] * torch.ones(N)[:,None]
        y1 = x[None] + 1/L * X

        Z = torch.zeros(N,L+1,self.d)
        Z[:,0] = y0
        Z[:,1] = y1

        for l in range(L-1):
            # print(f"iteration {l+2}")
            y2 = y1.clone()

            y0 = y0.detach()
            y1.requires_grad_()
            y2.requires_grad_()

            k = 0
            error_0 = 0.
            while k < max_iter: 
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
                if error / error_0 < tol:
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

    
    def distance(self, x, y, L=50):
        """

        :param x: N x M x d
        :param y: N x L x d
        :return: N x M x L
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def parallel_transport(self, x, X, y, L=50):
        """

        :param x: d
        :param X: N x d
        :param y: d
        :return: N x d
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )
    
    