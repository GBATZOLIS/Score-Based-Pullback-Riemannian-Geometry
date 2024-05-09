import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

class Curve(nn.Module): # TODO evaluation function without grad
    def __init__(self, d, p):
        super().__init__()

        self.d = d  # dimension
        self.p = p  # order of the curve
        
    def forward(self, t):
        """Evaluate the curve at parameter values t
        :param t: N 
        :return: N x d
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )
    
    def differential_forward(self, t):
        """Evaluate the speed of the curve at parameter values t
        :param t: N 
        :return: N x d
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )
    
    def fit(self, loss_function, num_time_points=100, num_epochs=1000, lr=0.01, weight_decay=0.1):
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        t = torch.linspace(0.,1.,num_time_points)
        
        for epoch in range(num_epochs): # TODO we could experiment with sampling times instead of having a fixed interval
            # t = torch.rand(num_time_points).sort().values
            optimizer.zero_grad()
            gamma_t = self.forward(t)
            dot_gamma_t = self.differential_forward(t)

            losses = loss_function(gamma_t, dot_gamma_t)
            loss = torch.mean(losses) 
            # l1_reg = torch.tensor(0.)
            # for param in self.parameters():
            #     l1_reg += torch.norm(param,1)
            # loss += l1_reg

            loss.backward()
            optimizer.step()
            
            if epoch % 100 == 0:
                squared_distance = torch.mean(torch.sqrt(losses)) ** 2
                print(f"Epoch {epoch}, Loss {loss.item()} | Validation: {loss.item() - squared_distance.item()}")
                # plt.plot(t, loss_function(gamma_t, dot_gamma_t).detach().numpy())
                # plt.show()