import torch

from src.diffeomorphisms import Diffeomorphism
from src.diffeomorphisms import transforms
from src.diffeomorphisms import nets as nn_
from src.diffeomorphisms import utils

from torch.autograd.functional import jvp

def get_base_transform_fn(args):
    dim = args.d
    def create_base_transform(i):
        if args.base_transform_type == 'rq':
            return transforms.PiecewiseRationalQuadraticCouplingTransform(
                mask=utils.create_alternating_binary_mask(
                    features=dim,
                    even=(i % 2 == 0)
                ),
                transform_net_create_fn=lambda in_features, out_features:
                nn_.ResidualNet(
                    in_features=in_features,
                    out_features=out_features,
                    hidden_features=args.hidden_features,
                    num_blocks=args.num_transform_blocks,
                    dropout_probability=args.dropout_probability,
                    use_batch_norm=args.use_batch_norm
                ),
                num_bins=args.num_bins,
                apply_unconditional_transform=False,
            )
        elif args.base_transform_type == 'affine':
            return transforms.AffineCouplingTransform(
                mask=utils.create_alternating_binary_mask(
                    features=dim,
                    even=(i % 2 == 0)
                ),
                transform_net_create_fn=lambda in_features, out_features:
                nn_.ResidualNet(
                    in_features=in_features,
                    out_features=out_features,
                    hidden_features=args.hidden_features,
                    num_blocks=args.num_transform_blocks,
                    dropout_probability=args.dropout_probability,
                    use_batch_norm=args.use_batch_norm
                )
            )
        else:
            raise ValueError
    
    return create_base_transform
        
class EuclideanDiffeomorphism(Diffeomorphism):

    def __init__(self, args) -> None:
        super().__init__(args.d)

        self.args = args
        self._transform = transforms.CompositeTransform([
                        get_base_transform_fn(args)(i) for i in range(args.num_flow_steps)
                    ])

    def forward(self, x):
        """
        :param x: N x 2
        :return: N x 2
        """
        out, logabsdetjacobian = self._transform(x, context=None)
        return out

    def inverse(self, y):
        """
        :param y: N x 2
        :return: N x 2
        """
        out, logabsdetjacobian = self._transform.inverse(y, context=None)
        return out

    def differential_forward(self, x, X):
        """
        Compute the differential map of phi at x for a vector X.
        
        :param x: A batch of points, N x 2.
        :param X: A batch of tangent vectors, N x 2.
        :return: A batch of transformed tangent vectors, N x 2.
        """

        #jvp: is a fast pytorch implementation of the jacobian vector product.
        _, jvp_result = jvp(lambda x: self._transform(x, context=None)[0], (x,), (X,))
        return jvp_result

    def differential_inverse(self, y, Y):
        """
        Compute the differential map of the inverse of phi at y for a vector Y.
        
        :param y: A batch of points, N x 2.
        :param Y: A batch of tangent vectors, N x 2.
        :return: A batch of transformed tangent vectors, N x 2.
        """
        _, jvp_result = jvp(lambda y: self._transform.inverse(y, context=None)[0], (y,), (Y,))
        return jvp_result
    
