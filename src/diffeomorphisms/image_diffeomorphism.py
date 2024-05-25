import torch
from torch import nn
from torch.autograd.functional import jvp

from src.diffeomorphisms import Diffeomorphism
from src.diffeomorphisms import transforms
from src.diffeomorphisms import nets as nn_
from src.diffeomorphisms import utils


class ConvNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.net = nn.Sequential(
            nn_.Conv2dSameSize(in_channels, hidden_channels, kernel_size=3),
            nn.ReLU(),
            nn_.Conv2dSameSize(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn_.Conv2dSameSize(hidden_channels, out_channels, kernel_size=3),
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

def create_transform(c, h, w, levels, hidden_channels, steps_per_level, alpha, num_bits, preprocessing, multi_scale, coupling_layer_type, spline_params, use_resnet, num_res_blocks, resnet_batchnorm, dropout_prob, actnorm):
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
    else:
        raise RuntimeError('Unknown preprocessing type: {}'.format(preprocessing))

    return transforms.CompositeTransform([preprocess_transform, mct])

class core_image_diffeomorphism(Diffeomorphism):

    def __init__(self, args) -> None:
        super().__init__(args.d)
        self.args = args
        self.transform = create_transform(
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
            actnorm=args.actnorm
        )

    def forward(self, x):
        out, logabsdetjacobian = self.transform(x, context=None)
        return out

    def inverse(self, y):
        out, logabsdetjacobian = self.transform.inverse(y, context=None)
        return out

    def differential_forward(self, x, X):
        _, jvp_result = jvp(lambda x: self.transform(x, context=None)[0], (x,), (X,))
        return jvp_result

    def differential_inverse(self, y, Y):
        _, jvp_result = jvp(lambda y: self.transform.inverse(y, context=None)[0], (y,), (Y,))
        return jvp_result

class image_diffeomorphism(core_image_diffeomorphism):

    def __init__(self, args) -> None:
        super().__init__(args)
        self.c, self.h, self.w = args.c, args.h, args.w  # Store the original shape

    def flatten_image(self, image_tensor):
        # Flatten the image tensor to shape (batch_size, c*h*w)
        return image_tensor.view(image_tensor.size(0), -1)

    def unflatten_image(self, flattened_tensor):
        # Unflatten the tensor to shape (batch_size, c, h, w)
        return flattened_tensor.view(flattened_tensor.size(0), self.c, self.h, self.w)

    def flatten_unflatten_decorator(func):
        def wrapper(self, *args):
            # Check if the input tensors are flattened
            input_was_flattened = [arg.dim() == 2 for arg in args]
            # Unflatten if necessary
            unflattened_args = [self.unflatten_image(arg) if is_flattened else arg for arg, is_flattened in zip(args, input_was_flattened)]
            # Call the original method
            result = func(self, *unflattened_args)
            # Flatten the output tensor if the inputs were flattened
            if isinstance(result, tuple):
                result = tuple(self.flatten_image(res) if is_flattened else res for res, is_flattened in zip(result, input_was_flattened))
            else:
                result = self.flatten_image(result) if any(input_was_flattened) else result
            return result
        return wrapper

    @flatten_unflatten_decorator
    def forward(self, x):
        return super().forward(x)

    @flatten_unflatten_decorator
    def inverse(self, y):
        return super().inverse(y)

    @flatten_unflatten_decorator
    def differential_forward(self, x, X):
        return super().differential_forward(x, X)

    @flatten_unflatten_decorator
    def differential_inverse(self, y, Y):
        return super().differential_inverse(y, Y)