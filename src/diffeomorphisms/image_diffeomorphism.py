import torch
from torch import nn
from torch.autograd.functional import jvp

from src.diffeomorphisms import Diffeomorphism
from src.diffeomorphisms import transforms
from src.diffeomorphisms import nets as nn_
from src.diffeomorphisms import utils


class Conv2dSameSize(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size):
        same_padding = kernel_size // 2  # Padding that would keep the spatial dims the same
        super().__init__(in_channels, out_channels, kernel_size,
                         padding=same_padding)

class ConvNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.net = nn.Sequential(
            Conv2dSameSize(in_channels, hidden_channels, kernel_size=3),
            nn.ReLU(),
            Conv2dSameSize(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            Conv2dSameSize(hidden_channels, out_channels, kernel_size=3),
        )

    def forward(self, inputs, context=None):
        return self.net.forward(inputs)

def create_transform_step(num_channels, hidden_channels, actnorm, coupling_layer_type, spline_params, use_resnet, num_res_blocks, resnet_batchnorm, dropout_prob):
    if use_resnet:
        def create_convnet(in_channels, out_channels):
            net = nn_.ConvResidualNet(in_channels=in_channels,
                                      out_channels=out_channels,
                                      hidden_channels=hidden_channels,
                                      num_blocks=num_res_blocks,
                                      use_batch_norm=resnet_batchnorm,
                                      dropout_probability=dropout_prob)
            return net
    else:
        if dropout_prob != 0.:
            raise ValueError()
        def create_convnet(in_channels, out_channels):
            return ConvNet(in_channels, hidden_channels, out_channels)

    mask = utils.create_mid_split_binary_mask(num_channels)

    if coupling_layer_type == 'cubic_spline':
        coupling_layer = transforms.PiecewiseCubicCouplingTransform(
            mask=mask,
            transform_net_create_fn=create_convnet,
            tails='linear',
            tail_bound=spline_params['tail_bound'],
            num_bins=spline_params['num_bins'],
            apply_unconditional_transform=spline_params['apply_unconditional_transform'],
            min_bin_width=spline_params['min_bin_width'],
            min_bin_height=spline_params['min_bin_height']
        )
    elif coupling_layer_type == 'quadratic_spline':
        coupling_layer = transforms.PiecewiseQuadraticCouplingTransform(
            mask=mask,
            transform_net_create_fn=create_convnet,
            tails='linear',
            tail_bound=spline_params['tail_bound'],
            num_bins=spline_params['num_bins'],
            apply_unconditional_transform=spline_params['apply_unconditional_transform'],
            min_bin_width=spline_params['min_bin_width'],
            min_bin_height=spline_params['min_bin_height']
        )
    elif coupling_layer_type == 'rational_quadratic_spline':
        coupling_layer = transforms.PiecewiseRationalQuadraticCouplingTransform(
            mask=mask,
            transform_net_create_fn=create_convnet,
            tails='linear',
            tail_bound=spline_params['tail_bound'],
            num_bins=spline_params['num_bins'],
            apply_unconditional_transform=spline_params['apply_unconditional_transform'],
            min_bin_width=spline_params['min_bin_width'],
            min_bin_height=spline_params['min_bin_height'],
            min_derivative=spline_params['min_derivative']
        )
    elif coupling_layer_type == 'affine':
        coupling_layer = transforms.AffineCouplingTransform(
            mask=mask,
            transform_net_create_fn=create_convnet
        )
    elif coupling_layer_type == 'additive':
        coupling_layer = transforms.AdditiveCouplingTransform(
            mask=mask,
            transform_net_create_fn=create_convnet
        )
    else:
        raise RuntimeError('Unknown coupling_layer_type')

    step_transforms = []

    if actnorm:
        step_transforms.append(transforms.ActNorm(num_channels))

    step_transforms.extend([
        transforms.OneByOneConvolution(num_channels),
        coupling_layer
    ])

    return transforms.CompositeTransform(step_transforms)

def create_transform(c, h, w, levels, hidden_channels, steps_per_level, alpha, num_bits, preprocessing, multi_scale, coupling_layer_type, spline_params, use_resnet, num_res_blocks, resnet_batchnorm, dropout_prob, actnorm, U, mean):
    if not isinstance(hidden_channels, list):
        hidden_channels = [hidden_channels] * levels

    if multi_scale:
        mct = transforms.MultiscaleCompositeTransform(num_transforms=levels)
        for level, level_hidden_channels in zip(range(levels), hidden_channels):
            squeeze_transform = transforms.SqueezeTransform()
            c, h, w = squeeze_transform.get_output_shape(c, h, w)

            transform_level = transforms.CompositeTransform(
                [squeeze_transform]
                + [create_transform_step(c, level_hidden_channels, actnorm, coupling_layer_type, spline_params, use_resnet, num_res_blocks, resnet_batchnorm, dropout_prob) for _ in range(steps_per_level)]
                + [transforms.OneByOneConvolution(c)]
            )

            new_shape = mct.add_transform(transform_level, (c, h, w))
            if new_shape:
                c, h, w = new_shape
    else:
        all_transforms = []
        all_transforms.append(transforms.PrincipalRotationTransform(U=U, mean=mean))

        for level, level_hidden_channels in zip(range(levels), hidden_channels):
            squeeze_transform = transforms.SqueezeTransform()
            c, h, w = squeeze_transform.get_output_shape(c, h, w)

            transform_level = transforms.CompositeTransform(
                [squeeze_transform]
                + [create_transform_step(c, level_hidden_channels, actnorm, coupling_layer_type, spline_params, use_resnet, num_res_blocks, resnet_batchnorm, dropout_prob) for _ in range(steps_per_level)]
                + [transforms.OneByOneConvolution(c)]
            )
            all_transforms.append(transform_level)

        all_transforms.append(transforms.ReshapeTransform(
            input_shape=(c,h,w),
            output_shape=(c*h*w,)
        ))
        mct = transforms.CompositeTransform(all_transforms)

    if preprocessing == 'glow':
        preprocess_transform = transforms.AffineScalarTransform(scale=(1. / 2 ** num_bits),
                                                                shift=-0.5)
    elif preprocessing == 'realnvp':
        preprocess_transform = transforms.CompositeTransform([
            transforms.AffineScalarTransform(scale=(1. / 2 ** num_bits)),
            transforms.AffineScalarTransform(shift=alpha, scale=(1 - alpha)),
            transforms.Logit()
        ])
    elif preprocessing == 'realnvp_2alpha':
        preprocess_transform = transforms.CompositeTransform([
            transforms.AffineScalarTransform(scale=(1. / 2 ** num_bits)),
            transforms.AffineScalarTransform(shift=alpha, scale=(1 - 2. * alpha)),
            transforms.Logit()
        ])
    elif preprocessing is None:
        preprocess_transform = None
    else:
        raise RuntimeError('Unknown preprocessing type: {}'.format(preprocessing))

    if preprocess_transform:
        return transforms.CompositeTransform([preprocess_transform, mct])
    else:
        return transforms.CompositeTransform([mct])


class image_diffeomorphism(Diffeomorphism):
    def __init__(self, args, U=None, mean=None) -> None:
        super().__init__(args.d)
        self.args = args
        self._transform = create_transform(
            c=args.c, h=args.h, w=args.w,
            levels=args.levels,
            hidden_channels=args.hidden_channels,
            steps_per_level=args.steps_per_level,
            alpha=args.alpha,
            num_bits=args.num_bits,
            preprocessing=args.preprocessing,
            multi_scale=args.multi_scale,
            coupling_layer_type=args.coupling_layer_type,
            spline_params=args.spline_params,
            use_resnet=args.use_resnet,
            num_res_blocks=args.num_res_blocks,
            resnet_batchnorm=args.resnet_batchnorm,
            dropout_prob=args.dropout_prob,
            actnorm=args.actnorm, 
            U=U,
            mean=mean
        )

    def forward(self, x, detach_logdet=False):
        """
        Forward pass through the image diffeomorphism.
        :param x: Input tensor, shape (B, C, H, W).
        :param detach_logdet: Whether to detach the logabsdetjacobian from the computation graph.
        :return: Transformed tensor and logabsdetjacobian.
        """
        out, logabsdetjacobian = self._transform(x, context=None, detach_logdet=detach_logdet)
        return out

    def inverse(self, y, detach_logdet=False):
        """
        Inverse pass through the image diffeomorphism.
        :param y: Input tensor, shape (B, C, H, W).
        :param detach_logdet: Whether to detach the logabsdetjacobian from the computation graph.
        :return: Inverse-transformed tensor and logabsdetjacobian.
        """
        out, logabsdetjacobian = self._transform.inverse(y, context=None, detach_logdet=detach_logdet)
        return out

    def differential_forward(self, x, X):
        """
        Compute the differential map of phi at x for a vector X.
        
        :param x: A batch of points, N x C x H x W.
        :param X: A batch of tangent vectors, N x C x H x W.
        :return: A batch of transformed tangent vectors, N x C x H x W.
        """
        _, jvp_result = jvp(lambda x: self._transform(x, context=None)[0], (x,), (X,))
        return jvp_result

    def differential_inverse(self, y, Y):
        """
        Compute the differential map of the inverse of phi at y for a vector Y.
        
        :param y: A batch of points, N x C x H x W.
        :param Y: A batch of tangent vectors, N x C x H x W.
        :return: A batch of transformed tangent vectors, N x C x H x W.
        """
        _, jvp_result = jvp(lambda y: self._transform.inverse(y, context=None)[0], (y,), (Y,))
        return jvp_result
