from src.manifolds.deformed_gaussian_pullback_manifold import DeformedGaussianPullbackManifold
from src.riemannian_autoencoder.deformed_gaussian_riemannian_autoencoder import DeformedGaussianRiemannianAutoencoder
from src.unimodal import Unimodal
import torch
import matplotlib.pyplot as plt


def check_manifold_properties(phi, psi):
    distribution = Unimodal(diffeomorphism=phi, strongly_convex=psi)
    manifold = DeformedGaussianPullbackManifold(distribution)

    #check the density
    xx = torch.linspace(-6.0, 6.0, 500)
    yy = torch.linspace(-6.0, 6.0, 500)
    x_grid, y_grid = torch.meshgrid(xx, yy)

    xy_grid = torch.zeros((*x_grid.shape,2))
    xy_grid[:,:,0] = x_grid
    xy_grid[:,:,1] = y_grid

    density = torch.exp(manifold.dg.log_density(xy_grid.reshape(-1,2)).reshape(x_grid.shape))
    plt.contour(x_grid, y_grid, density)
    plt.show()

    #Check barycenter

    ## special points
    x0 = torch.tensor([2.,4.])
    x1 = torch.tensor([2.,-4.])
    x = torch.zeros((2,2))
    x[0] = x0
    x[1] = x1
    barycentre = manifold.barycentre(x)

    plt.contour(x_grid, y_grid, density)
    plt.scatter(torch.tensor([x0[0], x1[0]]), torch.tensor([x0[1], x1[1]]))
    plt.scatter(barycentre[0], barycentre[1], color="orange")
    plt.savefig(".logs/figures/barycentre.eps")
    plt.show()

    # Test inner
    X = torch.eye(2)

    inner_0 = manifold.inner(torch.zeros(2)[None], X[None], X[None])
    inner_x0 = manifold.inner(x0[None], X[None], X[None])

    print(inner_0)
    print(inner_x0)

    # Test interpolation
    t = torch.linspace(0.,1.,100)

    geodesic = manifold.geodesic(x0,x1,t)

    plt.contour(x_grid, y_grid, density)
    plt.plot(geodesic[:,0], geodesic[:,1], color="orange")
    plt.scatter(torch.tensor([x0[0], x1[0]]), torch.tensor([x0[1], x1[1]]))
    plt.savefig(".logs/figures/geodesic.eps")
    plt.show()

    # test logarithmic mapping
    logarithmic = manifold.log(x0,x1[None])[0]

    plt.contour(x_grid, y_grid, density)
    plt.arrow(x0[0], x0[1], logarithmic[0], logarithmic[1], head_width=0.2, color="orange")
    plt.scatter(torch.tensor([x0[0], x1[0]]), torch.tensor([x0[1], x1[1]]))
    plt.savefig(".logs/figures/logarithmic.eps")
    plt.show()

    # test exponential mapping
    exponential = manifold.exp(x0,logarithmic[None])[0]

    plt.contour(x_grid, y_grid, density)
    plt.scatter(x0[0], x0[1])
    plt.arrow(x0[0], x0[1], logarithmic[0], logarithmic[1], head_width=0.2)
    plt.scatter(exponential[0], exponential[1], color="orange")
    plt.savefig(".logs/figures/exponential.eps")
    plt.show()
    print(f"The error between exp_x0(log_x0 (x1)) and x1 is {torch.norm(exponential - x1)}")

    # test distance
    l2_distance = torch.norm(x0 - x1)
    distance = manifold.distance(x0[None,None], x1[None,None])[0,0,0]
    print(l2_distance)
    print(distance)

    # test parallel transport
    parallel_transport = manifold.parallel_transport(x0, logarithmic[None], x1)[0]

    plt.contour(x_grid, y_grid, density)
    plt.scatter(torch.tensor([x0[0], x1[0]]), torch.tensor([x0[1], x1[1]]))
    plt.arrow(x0[0], x0[1], logarithmic[0], logarithmic[1], head_width=0.2)
    plt.arrow(x1[0], x1[1], parallel_transport[0], parallel_transport[1], head_width=0.2, color="orange")
    plt.savefig(".logs/figures/parallel-transport.eps")
    plt.show()

    # Riemannian autoencoder
    epsilon = 0.1
    banana_rae = DeformedGaussianRiemannianAutoencoder(manifold, epsilon)

    p = torch.linspace(-6, 6, 100)[:,None]
    rae_decode_p = banana_rae.decode(p)

    plt.contour(x_grid, y_grid, density)
    plt.plot(rae_decode_p[:,0], rae_decode_p[:,1], color="orange")
    plt.savefig(".logs/figures/rae_manifold.eps")
    plt.show()