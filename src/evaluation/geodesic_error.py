import torch

def geodesic_error(learned_pl_manifold, gt_pl_manifold, test_loader, device):
    distances = []
    
    # Iterate over the test_loader
    for batch in test_loader:
        if isinstance(batch, list):
            batch = batch[0]
        batch = batch.to(device)

        # Sample 50 pairs of points in each batch
        for _ in range(50):
            # Randomly pick two different points from the batch
            idx = torch.randperm(batch.size(0))[:2]
            x0, x1 = batch[idx[0]], batch[idx[1]]
            
            # Define interpolation parameter t
            t = torch.linspace(0., 1., 100, device=device)
            
            # Calculate geodesics for both learned and ground truth manifolds
            geodesic_learned = learned_pl_manifold.geodesic(x0, x1, t)
            geodesic_gt = gt_pl_manifold.geodesic(x0, x1, t)
            
            # Compute the distance between the two geodesics
            distance = torch.mean(torch.norm(geodesic_learned - geodesic_gt, dim=1))
            distances.append(distance)
    
    # Calculate the average distance (geodesic error)
    geodesic_error_mean = torch.mean(torch.tensor(distances)).item()
    geodesic_error_std = torch.std(torch.tensor(distances)).item()

    print(len(distances))
    
    return geodesic_error_mean, geodesic_error_std
