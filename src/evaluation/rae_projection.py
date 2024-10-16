import os
import torch
import plotly.graph_objects as go
import numpy as np
from .utils import get_learned_pullback_manifold
from src.riemannian_autoencoder.deformed_gaussian_riemannian_autoencoder import DeformedGaussianRiemannianAutoencoder
import random
import itertools

def rae_projection(psi, phi, tensorboard_dir, device, test_loader):
    """
    Evaluates the Riemannian autoencoder and saves interactive 3D plots.
    Overlays the test dataset on the plots.
    """
    
    epsilon = 0.1
    manifold = get_learned_pullback_manifold(phi, psi)
    rae = DeformedGaussianRiemannianAutoencoder(manifold, epsilon)

    # Create the 'plots' directory if it doesn't exist
    plots_dir = os.path.join(tensorboard_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Print the number of latent dimensions chosen and their corresponding variances
    print(f"Chosen d_eps (latent dimensions): {rae.d_eps}")
    corresponding_variances = psi.diagonal[rae.idx_eps].detach().cpu().numpy()
    print(f"Corresponding variances: {corresponding_variances}")

    # Sample the values in the latent dimensions using the corresponding variance for each dimension
    p_values = []
    num_points = 100  # Number of points per dimension
    for idx in range(rae.d_eps):
        # Use the variance for the current dimension
        std = np.sqrt(corresponding_variances[idx])
        p = torch.linspace(-3 * std, 3 * std, num_points, device=device)
        p_values.append(p.cpu().numpy())  # Convert to numpy for meshgrid

    # Create the meshgrid for 1D or 2D latent spaces
    if len(p_values) == 1:
        latent_samples = p_values[0][:, None]  # Shape: (num_points, 1)
        latent_samples = torch.tensor(latent_samples, device=device).float()
        rae_decode_p = rae.decode(latent_samples).detach().cpu().numpy()

    elif len(p_values) == 2:
        P1, P2 = np.meshgrid(*p_values)  # Shape: (num_points, num_points)
        latent_samples = np.stack([P1.ravel(), P2.ravel()], axis=-1)  # Shape: (num_points*num_points, 2)
        latent_samples = torch.tensor(latent_samples, device=device).float()
        rae_decode_p = rae.decode(latent_samples).detach().cpu().numpy()
    else:
        print("Visualization supports only 1D or 2D latent spaces.")
        return

    # Collect test data points from test_loader
    data_points = []
    for data in test_loader:
        if isinstance(data, (list, tuple)):
            x = data[0]
        else:
            x = data
        data_points.append(x)

    if not data_points:
        print("No data points found in test_loader.")
        return

    data_points = torch.cat(data_points, dim=0)
    data_points_np = data_points.detach().cpu().numpy()

    ambient_dim = rae_decode_p.shape[1]

    if ambient_dim >= 9:
        # Generate all unique combinations of (i, j, k) without considering permutations
        random.seed(42)
        combinations = list(itertools.combinations(range(ambient_dim), 3))
        selected_combinations = random.sample(combinations, 15)

        for combo_idx, (i, j, k) in enumerate(selected_combinations):
            # Sorting the dimensions to ensure increasing order
            dims = sorted([i, j, k])

            x_line = rae_decode_p[:, dims[0]]
            y_line = rae_decode_p[:, dims[1]]
            z_line = rae_decode_p[:, dims[2]]

            x_data = data_points_np[:7500, dims[0]]
            y_data = data_points_np[:7500, dims[1]]
            z_data = data_points_np[:7500, dims[2]]

            # Create the 3D plot using Plotly
            fig = go.Figure()

            if len(p_values) == 1:
                fig.add_trace(go.Scatter3d(
                    x=x_line,
                    y=y_line,
                    z=z_line,
                    mode='lines',
                    line=dict(color='orange', width=7),
                    opacity=1,
                    name='RAE Decoded'
                ))
            elif len(p_values) == 2:
                num_points = int(np.sqrt(len(x_line)))
                X = x_line.reshape(num_points, num_points)
                Y = y_line.reshape(num_points, num_points)
                Z = z_line.reshape(num_points, num_points)
                fig.add_trace(go.Surface(
                    x=X,
                    y=Y,
                    z=Z,
                    surfacecolor=np.zeros_like(X),
                    cmin=0,
                    cmax=1,
                    colorscale=[[0, 'orange'], [1, 'orange']],
                    showscale=False,
                    opacity=1,
                    name='RAE Decoded Surface'
                ))

            fig.add_trace(go.Scatter3d(
                x=x_data,
                y=y_data,
                z=z_data,
                mode='markers',
                marker=dict(
                    size=3,
                    color='blue',
                    opacity=0.17
                ),
                name='Test Data Points'
            ))

            # Remove background and axis information
            fig.update_layout(
                scene=dict(
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    zaxis=dict(visible=False),
                    bgcolor='white',  # White background
                ),
                paper_bgcolor='white',
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=False,
                width=700, height=700
            )

            # Save the figure as an HTML file with the dimensions in the name
            html_plot_path = os.path.join(plots_dir, f'riemannian_autoencoder_plot_dims_{dims[0]}_{dims[1]}_{dims[2]}.html')
            fig.write_html(html_plot_path)
            print(f"Interactive 3D plot saved at: {html_plot_path}")

    else:
        if ambient_dim < 3:
            print("Ambient dimension is less than 3, cannot create 3D plots.")
            return

        x_line = rae_decode_p[:, 0]
        y_line = rae_decode_p[:, 1]
        z_line = rae_decode_p[:, 2]

        x_data = data_points_np[:, 0]
        y_data = data_points_np[:, 1]
        z_data = data_points_np[:, 2]

        fig = go.Figure()

        if len(p_values) == 1:
            fig.add_trace(go.Scatter3d(
                x=x_line,
                y=y_line,
                z=z_line,
                mode='lines',
                line=dict(color='orange', width=6),
                opacity=1,
                name='RAE Decoded'
            ))
        elif len(p_values) == 2:
            num_points = int(np.sqrt(len(x_line)))
            X = x_line.reshape(num_points, num_points)
            Y = y_line.reshape(num_points, num_points)
            Z = z_line.reshape(num_points, num_points)
            fig.add_trace(go.Surface(
                x=X,
                y=Y,
                z=Z,
                surfacecolor=np.zeros_like(X),
                cmin=0,
                cmax=1,
                colorscale=[[0, 'orange'], [1, 'orange']],
                showscale=False,
                opacity=1,
                name='RAE Decoded Surface'
            ))

        fig.add_trace(go.Scatter3d(
            x=x_data,
            y=y_data,
            z=z_data,
            mode='markers',
            marker=dict(
                size=4,
                color='blue',
                opacity=0.4
            ),
            name='Test Data Points'
        ))

        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                bgcolor='white',  # White background
            ),
            paper_bgcolor='white',
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
            width=700, height=700
        )

        html_plot_path = os.path.join(plots_dir, 'riemannian_autoencoder.html')
        fig.write_html(html_plot_path)
        print(f'Interactive 3D plot saved at: {html_plot_path}')




def rae_projection_old(psi, phi, tensorboard_dir, device, test_loader):
    """
    Evaluates the Riemannian autoencoder and saves an interactive 3D plot.
    Overlays the test dataset on the plot.
    """
    
    epsilon = 0.1
    manifold = get_learned_pullback_manifold(phi, psi)
    rae = DeformedGaussianRiemannianAutoencoder(manifold, epsilon)

    # Create the 'plots' directory if it doesn't exist
    plots_dir = os.path.join(tensorboard_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Print the number of latent dimensions chosen and their corresponding variances
    print(f"Chosen d_eps (latent dimensions): {rae.d_eps}")
    corresponding_variances = psi.diagonal[rae.idx_eps].detach().cpu().numpy()
    print(f"Corresponding variances: {corresponding_variances}")

    # Sample the values in the latent dimensions using the corresponding variance for each dimension
    p_values = []
    num_points = 100  # Number of points per dimension
    for idx in range(rae.d_eps):
        # Use the variance for the current dimension
        std = np.sqrt(corresponding_variances[idx])
        p = torch.linspace(-3 * std, 3 * std, num_points, device=device)
        p_values.append(p.cpu().numpy())  # Convert to numpy for meshgrid

    # Create the meshgrid for 1D or 2D latent spaces
    if len(p_values) == 1:
        # For 1D latent space
        latent_samples = p_values[0][:, None]  # Shape: (num_points, 1)
        latent_samples = torch.tensor(latent_samples, device=device).float()
        # Decode the latent samples
        rae_decode_p = rae.decode(latent_samples).detach().cpu().numpy()

        # Prepare data for plotting
        x_line = rae_decode_p[:, 0]
        y_line = rae_decode_p[:, 1]
        z_line = rae_decode_p[:, 2]

        # Create the 3D plot using Plotly
        fig = go.Figure()
        # Plot the decoded points in 3D space as a line
        fig.add_trace(go.Scatter3d(
        x=x_line,
        y=y_line,
        z=z_line,
        mode='lines',
        line=dict(color='orange', width=6),  # You can also adjust the line width if needed
        opacity=1,  # Full opacity for the entire trace
        name='RAE Decoded'
    ))


    elif len(p_values) == 2:
        # For 2D latent space
        P1, P2 = np.meshgrid(*p_values)  # Shape: (num_points, num_points)
        latent_samples = np.stack([P1.ravel(), P2.ravel()], axis=-1)  # Shape: (num_points*num_points, 2)
        latent_samples = torch.tensor(latent_samples, device=device).float()
        # Decode the latent samples
        rae_decode_p = rae.decode(latent_samples).detach().cpu().numpy()
        # Reshape the decoded samples to grid shape
        X = rae_decode_p[:, 0].reshape(num_points, num_points)
        Y = rae_decode_p[:, 1].reshape(num_points, num_points)
        Z = rae_decode_p[:, 2].reshape(num_points, num_points)

        # Create the 3D plot using Plotly
        fig = go.Figure()

        # Use Surface to connect neighboring points in the grid
        # Use 'orange' color without any colormap
        fig.add_trace(go.Surface(
            x=X,
            y=Y,
            z=Z,
            surfacecolor=np.zeros_like(X),  # Dummy values for surfacecolor
            cmin=0,
            cmax=1,
            colorscale=[[0, 'orange'], [1, 'orange']],  # Single color 'orange'
            showscale=False,
            opacity=1,
            name='RAE Decoded Surface'
        ))
    else:
        # For higher dimensions, we exit as per your instruction
        print("Visualization supports only 1D or 2D latent spaces.")
        return

    # Collect test data points from test_loader
    data_points = []
    for data in test_loader:
        if isinstance(data, (list, tuple)):
            x = data[0]
        else:
            x = data
        data_points.append(x)

    if not data_points:
        # No data to plot
        print("No data points found in test_loader.")
        return

    data_points = torch.cat(data_points, dim=0)
    data_points_np = data_points.detach().cpu().numpy()

    # Overlay the test data points as scatter points with alpha=0.5
    fig.add_trace(go.Scatter3d(
        x=data_points_np[:, 0],
        y=data_points_np[:, 1],
        z=data_points_np[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color='blue',
            opacity=0.17
        ),
        name='Test Data Points'
    ))

    # Update layout to configure the size and labels
    fig.update_layout(
        title="Riemannian Autoencoder 3D Visualization",
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis'
        ),
        width=700, height=700
    )

    # Save the figure as an HTML file so it can be interacted with
    html_plot_path = os.path.join(plots_dir, 'riemannian_autoencoder.html')
    fig.write_html(html_plot_path)

    print(f"Interactive 3D plot saved at: {html_plot_path}")
