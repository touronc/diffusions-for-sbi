import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from nse import NSE

from torch import Tensor, Size
from torch.distributions import Distribution
from tqdm import tqdm
from typing import *

from zuko.distributions import DiagNormal, NormalizingFlow
from zuko.nn import MLP
from zuko.transforms import FreeFormJacobianTransform
from zuko.utils import broadcast


class PF_NPSE(NSE):
    r"""Creates a partially fatorized neural posterior score estimatior (PF-NPSE).

    Arguments:
        theta_dim: The dimensionality :math:`D` of the parameter space.
        x_dim: The dimensionality :math:`L` of the observation space.
        freqs: The number of time embedding frequencies.
        n_max:
        build_net: The network constructor. It takes the
            number of input and output features as positional arguments.
        embedding_nn_theta: The embedding network for the parameters :math:`\theta`.
        embedding_nn_x: The embedding network for the observations :math:`x`.
        kwargs: Keyword arguments passed to the network constructor `build_net`.
    """

    def __init__(
        self,
        theta_dim: int,
        x_dim: int,
        freqs: int = 3,
        n_max: int = 1,
        build_net: Callable[[int, int], nn.Module] = MLP,
        embedding_nn_theta: nn.Module = nn.Identity(),
        embedding_nn_x: nn.Module = nn.Identity(),
        **kwargs
    ):
        super().__init__(
            theta_dim,
            x_dim,
            freqs,
            build_net,
            embedding_nn_theta,
            embedding_nn_x,
            **kwargs
        )

        self.n_max = n_max

        self.net = build_net(
            self.theta_emb_dim + self.x_emb_dim + 2 * freqs + n_max, theta_dim, **kwargs
        )

    def create_mask(self, n, batch_dim):
        shape = (self.n_max,)
        if batch_dim != 0:
            shape = (batch_dim,) + shape

        if n is not None:
            mask = ((torch.arange(self.n_max).expand(shape).to(n) < n) * 1).unsqueeze(
                -1
            )
        else:
            # Only ones for fixed set sizes
            mask = torch.ones((shape)).unsqueeze(-1)
        return mask  # (*, n_max, 1)

    def set_mask(self, mask):
        """Sets the masks in context embedding net."""
        if hasattr(self.embedding_nn_x, "set_mask"):
            self.embedding_nn_x.set_mask(mask)
        elif hasattr(self.embedding_nn_x, "__iter__"):
            for layer in self.embedding_nn_x:
                if hasattr(layer, "set_mask"):
                    layer.set_mask(mask)

    def aggregate(self, x, n):
        r"""Aggregates context sets of size :math:`n` in :math:`x` by computing
        the mean over set elements :math:`x^j`:

            .. math: `x_agg = \frac{1}{n}\sum_{j=1}^{n} x^j`

        Arguments:
            x: (masked) context variables, with shape :math:`(n_max, *, L)`
            n: sizes of the context sets, with shape :math:`(*,)`

        Returns:
            x_agg: aggregated context variable, with shape :math:`(*, L)`
        """
        if x.ndim > 2:  # batch dimensiojn exists
            x = x.permute(1, 0, 2)  # reshape to (n_max, *, x_emb_dim)
        if n is not None and n.sum() != 0:
            # variable context sizes
            x_agg = torch.sum(x, dim=0) / n
        else:
            # fixed context size
            x_agg = torch.mean(x, dim=0)
        return x_agg

    def forward(self, theta, x, t, n=None):
        r"""
        Arguments:
            theta: The parameters :math:`\theta`, with shape :math:`(*, D)`.
            x: The observations :math:`x`, with shape :math:`(*, n_max, L)`.
            t: The time :math:`t`, with shape :math:`(*,)`.
            n: The sizes of the context sets, with shape :math:`(*,)`.
                Values either range between 1 and :math:`n_max`, or be all 0 (no contex = prior).
                If None, observation sets must be of fixed size :math:`n=n_max`.

        Returns:
            The estimated noise :math:`\epsilon_\phi(\theta, x, t)`, with shape :math:`(*, D)`.
        """

        # Define masks to allow variable set sizes
        mask = self.create_mask(n, batch_dim=(theta.ndim > 1) * theta.shape[0])
        # Mask observations
        x = mask * x
        # Set masks for embedding net
        self.set_mask(mask)

        # Positional embedding for the time variable
        t = self.freqs * t[..., None]
        t = torch.cat((t.cos(), t.sin()), dim=-1)

        # Embeddings for parameters and observations
        theta = self.embedding_nn_theta(theta)
        x = self.embedding_nn_x(x)

        # Aggregate observation sets
        x = mask * x  # necessary if x isn't masked within the embedding net
        x = self.aggregate(x, n)

        if n is not None:
            # One-hot encode the set sizes n between 0 and n_max
            if n[0] > 0:
                n = F.one_hot(n.long().squeeze() - 1, num_classes=self.n_max)
            else:
                n = torch.zeros((n.shape[0], self.n_max))
            # Compute the score
            theta, x, t, n = broadcast(theta, x, t, n, ignore=1)
            return self.net(torch.cat((theta, x, t, n), dim=-1))
        else:
            # Compute the score
            return self.net(torch.cat((theta, x, t), dim=-1))

