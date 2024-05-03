from src.manifolds.deformed_gaussian_pullback_manifold import DeformedGaussianPullbackManifold
from src.riemannian_autoencoder.deformed_gaussian_riemannian_autoencoder import DeformedGaussianRiemannianAutoencoder
from src.unimodal import Unimodal
import torch
import matplotlib.pyplot as plt


def check_manifold_properties(phi, psi, writer, epoch):
    distribution = Unimodal(diffeomorphism=phi, strongly_convex=psi)
    manifold = DeformedGaussianPullbackManifold(distribution)

    #check the density
    xx = torch.linspace(-6.0, 6.0, 500)
    yy = torch.linspace(-6.0, 6.0, 500)
    x_grid, y_grid = torch.meshgrid(xx, yy)

    xy_grid = torch.zeros((*x_grid.shape,2))
    xy_grid[:,:,0] = x_grid
    xy_grid[:,:,1] = y_grid

    density = torch.exp(manifold.dg.log_density(xy_grid.reshape(-1,2)).reshape(x_grid.shape)).detach()
    fig, ax = plt.subplots()
    ax.contour(x_grid, y_grid, density)
    writer.add_figure("Density", fig, epoch)
    plt.close(fig)

    #Check barycenter

    ## special points
    x0 = torch.tensor([2.,4.])
    x1 = torch.tensor([2.,-4.])
    x = torch.zeros((2,2))
    x[0] = x0
    x[1] = x1
    barycentre = manifold.barycentre(x).detach()

    fig, ax = plt.subplots()
    ax.contour(x_grid, y_grid, density)
    ax.scatter(torch.tensor([x0[0], x1[0]]), torch.tensor([x0[1], x1[1]]))
    ax.scatter(barycentre[0], barycentre[1], color="orange")
    writer.add_figure("Barycentre", fig, epoch)
    plt.close(fig)

    # Test interpolation
    t = torch.linspace(0.,1.,100)
    geodesic = manifold.geodesic(x0,x1,t).detach()

    fig, ax = plt.subplots()
    ax.contour(x_grid, y_grid, density)
    ax.plot(geodesic[:, 0], geodesic[:, 1], color="orange")
    ax.scatter(torch.tensor([x0[0], x1[0]]), torch.tensor([x0[1], x1[1]]))
    writer.add_figure("Geodesic", fig, epoch)
    plt.close(fig)

    # Test logarithmic mapping
    logarithmic = manifold.log(x0, x1[None])[0].detach()
    fig, ax = plt.subplots()
    ax.contour(x_grid, y_grid, density)
    ax.arrow(x0[0], x0[1], logarithmic[0], logarithmic[1], head_width=0.2, color="orange")
    ax.scatter(torch.tensor([x0[0], x1[0]]), torch.tensor([x0[1], x1[1]]))
    writer.add_figure("Logarithmic Mapping", fig, epoch)
    plt.close(fig)

    # Test exponential mapping
    exponential = manifold.exp(x0, logarithmic[None])[0].detach()
    fig, ax = plt.subplots()
    ax.contour(x_grid, y_grid, density)
    ax.scatter(x0[0], x0[1])
    ax.arrow(x0[0], x0[1], logarithmic[0], logarithmic[1], head_width=0.2)
    ax.scatter(exponential[0], exponential[1], color="orange")
    writer.add_figure("Exponential Mapping", fig, epoch)
    plt.close(fig)

    # Log the error between exp_x0(log_x0 (x1)) and x1 to TensorBoard
    error = torch.norm(exponential - x1)
    writer.add_scalar("Error exp(log) vs x1", error.item(), epoch)

    # Test distance
    l2_distance = torch.norm(x0 - x1)
    distance = manifold.distance(x0[None, None], x1[None, None])[0, 0, 0]
    writer.add_text("Distance", f"L2: {l2_distance}, Manifold: {distance}", epoch)

    # Test parallel transport
    parallel_transport = manifold.parallel_transport(x0, logarithmic[None], x1)[0].detach()

    fig, ax = plt.subplots()
    ax.contour(x_grid, y_grid, density)
    ax.scatter(torch.tensor([x0[0], x1[0]]), torch.tensor([x0[1], x1[1]]))
    ax.arrow(x0[0], x0[1], logarithmic[0], logarithmic[1], head_width=0.2)
    ax.arrow(x1[0], x1[1], parallel_transport[0], parallel_transport[1], head_width=0.2, color="orange")
    writer.add_figure("Parallel Transport", fig, epoch)
    plt.close(fig)

    # Riemannian autoencoder
    epsilon = 0.1
    banana_rae = DeformedGaussianRiemannianAutoencoder(manifold, epsilon)
    p = torch.linspace(-6, 6, 100)[:, None]
    rae_decode_p = banana_rae.decode(p).detach()

    fig, ax = plt.subplots()
    ax.contour(x_grid, y_grid, density)
    ax.plot(rae_decode_p[:, 0], rae_decode_p[:, 1], color="orange")
    writer.add_figure("Riemannian Autoencoder", fig, epoch)
    plt.close(fig)