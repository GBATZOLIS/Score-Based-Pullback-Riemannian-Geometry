import torch

import matplotlib.pyplot as plt
import seaborn as sns

# double banana example
def double_banana(x, y):
    return torch.exp(-2*(torch.sqrt(x**2 + y**2)-3)**2) * \
        (torch.exp(-2*(x-3)**2) + torch.exp(-2*(x+3)**2))

delta = 0.025
x = torch.linspace(-5.0, 5.0, 100)
y = torch.linspace(-5.0, 5.0, 100)
X, Y = torch.meshgrid(x, y)
Z_banana = double_banana(X, Y)

plt.contour(X, Y, Z_banana)
plt.show()

# diffeomorphisms

def approximate_single_banana(x,y, offset=3):
    phi_1_x = x - 1/offset**2 * y**2
    phi_1_y = y
    phi_2_x = x + 1/offset**2 * y**2
    phi_2_y = y
    return torch.exp(- (4 * (phi_1_x + offset) **2 + 1/6 * phi_1_y **2)) + torch.exp(- (4 * (phi_2_x - offset) **2 + 1/6 * phi_2_y **2))

Z_single_banana = approximate_single_banana(X, Y)

plt.contour(X, Y, Z_single_banana)
plt.show()


# TODO create class for target distribution