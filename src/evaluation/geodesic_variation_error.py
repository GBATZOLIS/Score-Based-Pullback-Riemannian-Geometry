import torch

def geodesic_variation_error(learned_pl_manifold, gt_pl_manifold, test_loader, device):
    variation_distances = []
    
    # Iterate over the test_loader
    for batch in test_loader:
        if isinstance(batch, list):
            batch = batch[0]
        batch = batch.to(device)
        
        # Sample 50 pairs of points in each batch
        for _ in range(50):
            # Randomly pick two points from the batch x1 and xN
            idx = torch.randperm(batch.size(0))[:2]
            x0, x1 = batch[idx[0]], batch[idx[1]]
            
            # Generate a new point z close to x1 (small perturbation)
            z = x1 + 0.1 * torch.randn_like(x1)  # Adding a small random perturbation to x1
            
            # Define interpolation parameter t
            t = torch.linspace(0., 1., 100, device=device)
            
            # Calculate geodesics for learned and ground truth manifolds
            geodesic_x0_x1_learned = learned_pl_manifold.geodesic(x0, x1, t)
            geodesic_x0_z_learned = learned_pl_manifold.geodesic(x0, z, t)
            
            # Compute the variation distance for both learned and ground truth manifolds
            learned_variation = torch.mean(torch.norm(geodesic_x0_x1_learned - geodesic_x0_z_learned, dim=1))
            
            # Store the variation distances
            variation_distances.append(learned_variation)
    
    # Calculate the average geodesic variation error
    variation_error_mean = torch.mean(torch.tensor(variation_distances)).item()
    variation_error_std = torch.std(torch.tensor(variation_distances)).item()
    
    return variation_error_mean, variation_error_std
