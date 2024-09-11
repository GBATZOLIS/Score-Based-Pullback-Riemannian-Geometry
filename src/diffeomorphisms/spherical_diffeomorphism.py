import torch
import torch.nn as nn
import numpy as np

from src.diffeomorphisms import Diffeomorphism

class spherical_diffeomorphism(Diffeomorphism):
    def __init__(self) -> None:
        super().__init__(3)  # We are in 3D space
    
    def forward(self, x):
        """
        Computes the forward transformation from Cartesian to spherical coordinates with a shift.
        :param x: Tensor of shape (N, 3), where N is the batch size.
        :return: Transformed tensor of shape (N, 3), where the first component is radius, 
                 and the second and third components are shifted azimuthal and polar angles.
        """
        y = x.clone()
        r = torch.sqrt(x[:, 0]**2 + x[:, 1]**2 + x[:, 2]**2)  # Radius
        theta = torch.atan2(x[:, 1], x[:, 0])  # Azimuthal angle (longitude)
        phi = torch.acos(x[:, 2] / r)  # Polar angle (latitude)
        
        # Apply the shifts
        r_shifted = r - 1
        theta_shifted = theta - (torch.pi / 4)
        phi_shifted = phi - (torch.pi / 4)
        
        return torch.stack([r_shifted, theta_shifted, phi_shifted], dim=1)

    def inverse(self, y):
        """
        Computes the inverse transformation from spherical coordinates (with shift) back to Cartesian coordinates.
        :param y: Tensor of shape (N, 3), where N is the batch size.
        :return: Inverted tensor of shape (N, 3), back to Cartesian coordinates.
        """
        r_shifted, theta_shifted, phi_shifted = y[:, 0], y[:, 1], y[:, 2]
        
        # Recover the unshifted spherical coordinates
        r = r_shifted + 1
        theta = theta_shifted + (torch.pi / 4)
        phi = phi_shifted + (torch.pi / 4)
        
        # Convert back to Cartesian coordinates
        x1 = r * torch.sin(phi) * torch.cos(theta)
        x2 = r * torch.sin(phi) * torch.sin(theta)
        x3 = r * torch.cos(phi)
        
        return torch.stack([x1, x2, x3], dim=1)

    def differential_forward(self, x, X):
        """
        Computes the differential of the forward transformation (Jacobian).
        :param x: Tensor of shape (N, 3), inputs in Cartesian coordinates.
        :param X: Tensor of shape (N, 3), differentials to transform.
        :return: Transformed differential tensor of shape (N, 3), in spherical coordinates.
        """
        r = torch.sqrt(x[:, 0]**2 + x[:, 1]**2 + x[:, 2]**2)  # Radius
        theta = torch.atan2(x[:, 1], x[:, 0])  # Azimuthal angle
        phi = torch.acos(x[:, 2] / r)  # Polar angle
        
        # Compute the Jacobian components for the forward transformation
        dr_dx1 = x[:, 0] / r
        dr_dx2 = x[:, 1] / r
        dr_dx3 = x[:, 2] / r
        
        dtheta_dx1 = -x[:, 1] / (x[:, 0]**2 + x[:, 1]**2)
        dtheta_dx2 = x[:, 0] / (x[:, 0]**2 + x[:, 1]**2)
        dtheta_dx3 = torch.zeros_like(x[:, 2])
        
        dphi_dx1 = -x[:, 0] * x[:, 2] / (r**2 * torch.sqrt(x[:, 0]**2 + x[:, 1]**2))
        dphi_dx2 = -x[:, 1] * x[:, 2] / (r**2 * torch.sqrt(x[:, 0]**2 + x[:, 1]**2))
        dphi_dx3 = torch.sqrt(x[:, 0]**2 + x[:, 1]**2) / r**2
        
        # Build the full Jacobian matrix
        D_x = torch.zeros_like(X)
        D_x[:, 0] = X[:, 0] * dr_dx1 + X[:, 1] * dr_dx2 + X[:, 2] * dr_dx3
        D_x[:, 1] = X[:, 0] * dtheta_dx1 + X[:, 1] * dtheta_dx2 + X[:, 2] * dtheta_dx3
        D_x[:, 2] = X[:, 0] * dphi_dx1 + X[:, 1] * dphi_dx2 + X[:, 2] * dphi_dx3
        
        return D_x

    def differential_inverse(self, y, Y):
        """
        Computes the differential of the inverse transformation (Jacobian).
        :param y: Tensor of shape (N, 3), inputs in spherical coordinates (with shift).
        :param Y: Tensor of shape (N, 3), differentials to invert.
        :return: Inverted differential tensor of shape (N, 3), in Cartesian coordinates.
        """
        r_shifted, theta_shifted, phi_shifted = y[:, 0], y[:, 1], y[:, 2]
        
        # Recover the unshifted spherical coordinates
        r = r_shifted + 1
        theta = theta_shifted + (torch.pi / 4)
        phi = phi_shifted + (torch.pi / 4)
        
        # Compute the partial derivatives for the inverse transformation
        dx1_dr = torch.sin(phi) * torch.cos(theta)
        dx2_dr = torch.sin(phi) * torch.sin(theta)
        dx3_dr = torch.cos(phi)
        
        dx1_dtheta = -r * torch.sin(phi) * torch.sin(theta)
        dx2_dtheta = r * torch.sin(phi) * torch.cos(theta)
        dx3_dtheta = torch.zeros_like(r)
        
        dx1_dphi = r * torch.cos(phi) * torch.cos(theta)
        dx2_dphi = r * torch.cos(phi) * torch.sin(theta)
        dx3_dphi = -r * torch.sin(phi)
        
        # Build the full Jacobian matrix for the inverse transformation
        D_y = torch.zeros_like(Y)
        D_y[:, 0] = Y[:, 0] * dx1_dr + Y[:, 1] * dx1_dtheta + Y[:, 2] * dx1_dphi
        D_y[:, 1] = Y[:, 0] * dx2_dr + Y[:, 1] * dx2_dtheta + Y[:, 2] * dx2_dphi
        D_y[:, 2] = Y[:, 0] * dx3_dr + Y[:, 1] * dx3_dtheta + Y[:, 2] * dx3_dphi
        
        return D_y
