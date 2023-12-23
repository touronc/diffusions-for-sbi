import math
import torch
import torch.nn as nn

from torch import Tensor, Size
from torch.distributions import Distribution
from tqdm import tqdm
from typing import *

from zuko.distributions import DiagNormal, NormalizingFlow
from zuko.nn import MLP
from zuko.transforms import FreeFormJacobianTransform
from zuko.utils import broadcast


class NSE(nn.Module):
    r"""Creates a neural score estimation (NSE) network.

    Arguments:
        theta_dim: The dimensionality :math:`D` of the parameter space.
        x_dim: The dimensionality :math:`L` of the observation space.
        freqs: The number of time embedding frequencies.
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
        build_net: Callable[[int, int], nn.Module] = MLP,
        embedding_nn_theta: nn.Module = nn.Identity(),
        embedding_nn_x: nn.Module = nn.Identity(),
        **kwargs,
    ):
        super().__init__()

        self.embedding_nn_theta = embedding_nn_theta
        self.embedding_nn_x = embedding_nn_x
        self.theta_emb_dim, self.x_emb_dim = self.get_theta_x_embedding_dim(
            theta_dim, x_dim
        )

        self.net = build_net(
            self.theta_emb_dim + self.x_emb_dim + 2 * freqs, theta_dim, **kwargs
        )

        self.register_buffer("freqs", torch.arange(1, freqs + 1) * math.pi)
        self.register_buffer("zeros", torch.zeros(theta_dim))
        self.register_buffer("ones", torch.ones(theta_dim))

    def get_theta_x_embedding_dim(self, theta_dim, x_dim) -> int:
        r"""Returns the dimensionality of the embeddings for :math:`\theta` and :math:`x`."""
        theta, x = torch.ones((1, theta_dim)), torch.ones((1, x_dim))
        return (
            self.embedding_nn_theta(theta).shape[-1],
            self.embedding_nn_x(x).shape[-1],
        )

    def forward(self, theta: Tensor, x: Tensor, t: Tensor) -> Tensor:
        r"""
        Arguments:
            theta: The parameters :math:`\theta`, with shape :math:`(*, D)`.
            x: The observation :math:`x`, with shape :math:`(*, L)`.
            t: The time :math:`t`, with shape :math:`(*,).`

        Returns:
            The estimated noise :math:`\epsilon_\phi(\theta, x, t)`, with shape :math:`(*, D)`.
        """

        t = self.freqs * t[..., None]
        t = torch.cat((t.cos(), t.sin()), dim=-1)

        theta = self.embedding_nn_theta(theta)
        x = self.embedding_nn_x(x)

        theta, x, t = broadcast(theta, x, t, ignore=1)

        return self.net(torch.cat((theta, x, t), dim=-1))

    # The following function define the VP SDE with linear noise schedule beta(t):
    # dtheta = f(t) theta dt + g(t) dW = -0.5 * beta(t) theta dt + sqrt(beta(t)) dW

    def beta(self, t: Tensor) -> Tensor:
        r"""Linear noise schedule of the VP SDE:
        .. math:: \beta(t) = 32 t .
        """
        return 32 * t

    def f(self, t: Tensor) -> Tensor:
        """Drift of the VP SDE:
        .. math:: f(t) = -0.5 * \beta(t) .
        """
        return -0.5 * self.beta(t)

    def g(self, t: Tensor) -> Tensor:
        """
        .. math:: g(t) = \sqrt{\beta(t)} .
        """
        return torch.sqrt(self.beta(t))

    def alpha(self, t: Tensor) -> Tensor:
        r"""Mean of the transition kernel of the VP SDE:
        .. math: `alpha(t) = \exp ( -0.5 \int_0^t \beta(s)ds)`.
        """
        return torch.exp(-8 * t**2)

    def sigma(self, t: Tensor) -> Tensor:
        r"""Standard deviation of the transition kernel of the VP SDE:
        .. math:: \sigma^2(t) = 1 - \exp( - \int_0^t \beta(s)ds) + C
        where C is such that :math: `\sigma^2(1) = 1, \sigma^2(0)  = \epsilon \approx 1e-4`.
        """
        return torch.sqrt(1 - self.alpha(t) ** 2 + math.exp(-16))

    def ode(self, theta: Tensor, x: Tensor, t: Tensor, **kwargs) -> Tensor:
        return self.f(t) * theta + self.g(t) ** 2 / 2 * self(
            theta, x, t, **kwargs
        ) / self.sigma(t)

    def flow(self, x: Tensor, **kwargs) -> Distribution:
        r"""
        Arguments:
            x: The observation :math:`x`, with shape :math:`(*, L)`.
            kwargs: additional args for the forward method.

        Returns:
            The normalizing flow :math:`p_\phi(\theta | x)` induced by the
            probability flow ODE.
        """

        return NormalizingFlow(
            transform=FreeFormJacobianTransform(
                f=lambda t, theta: self.ode(theta, x, t, **kwargs),
                t0=x.new_tensor(0.0),
                t1=x.new_tensor(1.0),
                phi=(x, *self.parameters()),
            ),
            base=DiagNormal(self.zeros, self.ones).expand(x.shape[:-1]),
        )

    def ddim(
        self, shape: Size, x: Tensor, steps: int = 64, verbose: bool = False, **kwargs
    ):
        time = torch.linspace(1, 0, steps + 1).to(x)
        dt = 1 / steps

        theta = DiagNormal(self.zeros, self.ones).sample(shape)

        for t in tqdm(time[:-1], disable=not verbose):
            ratio = self.alpha(t - dt) / self.alpha(t)
            theta = ratio * theta + (self.sigma(t - dt) - ratio * self.sigma(t)) * self(
                theta, x, t, **kwargs
            )

        return theta

    def euler(
        self, shape: Size, x: Tensor, steps: int = 64, verbose: bool = False, **kwargs
    ):
        time = torch.linspace(1, 0, steps + 1).to(x)
        dt = 1 / steps

        theta = DiagNormal(self.zeros, self.ones).sample(shape)

        for t in tqdm(time[:-1], disable=not verbose):
            z = torch.randn_like(theta)

            score = self(theta, x, t, **kwargs) / (-self.sigma(t))

            drift = self.f(t) * theta - self.g(t) ** 2 * score
            diffusion = self.g(t)

            theta = theta + drift * (-dt) + diffusion * z * dt**0.5

        return theta

    def annealed_langevin(
        self,
        shape: Size,
        x: Tensor,
        steps: int = 64,
        lsteps: int = 1000,
        tau: float = 1,
        verbose: bool = False,
        **kwargs,
    ):
        time = torch.linspace(1, 0, steps + 1).to(x)

        theta = DiagNormal(self.zeros, self.ones).sample(shape)

        for t in tqdm(time[:-1], disable=not verbose):
            for _ in range(lsteps):
                z = torch.randn_like(theta)
                score = self(theta, x, t, **kwargs) / -self.sigma(t)
                # delta = tau * self.alpha(t) / score.square().mean()
                delta = (
                    tau
                    * self.alpha(t)
                    * min(self.sigma(t) ** 2, 1 / score.square().mean())
                )
                theta = theta + delta * score + torch.sqrt(2 * delta) * z

        return theta


class NSELoss(nn.Module):
    r"""Calculates the *noise parametrized* denoising score matching (DSM) loss for NSE.
    Minimizing this loss estimates the noise :math: `\eplison_phi`, from which the score function
    can be calculated as

        .. math: `s_\phi(\theta, x, t) = - \sigma(t) * \epsilon_\phi(\theta, x, t)`.

    Given a batch of :math:`N` pairs :math:`(\theta_i, x_i)`, the module returns

    .. math:: l = \frac{1}{N} \sum_{i = 1}^N\|
            \epsilon_\phi(\alpha(t_i) \theta_i + \sigma(t_i) \epsilon_i, x_i, t_i)
            - \epsilon_i
        \|_2^2

    where :math:`t_i \sim \mathcal{U}(0, 1)` and :math:`\epsilon_i \sim \mathcal{N}(0, I)`.

    Arguments:
        estimator: A regression network :math:`\epsilon_\phi(\theta, x, t)`.
    """

    def __init__(self, estimator: NSE):
        super().__init__()

        self.estimator = estimator

    def forward(self, theta: Tensor, x: Tensor, **kwargs) -> Tensor:
        r"""
        Arguments:
            theta: The parameters :math:`\theta`, with shape :math:`(N, D)`.
            x: The observation :math:`x`, with shape :math:`(N, L)`.
            kwargs: Additional args for the forward method of the estimator.

        Returns:
            The scalar loss :math:`l`.
        """

        t = torch.rand(theta.shape[0], dtype=theta.dtype, device=theta.device)

        alpha = self.estimator.alpha(t)
        sigma = self.estimator.sigma(t)

        eps = torch.randn_like(theta)
        theta_t = alpha[:, None] * theta + sigma[:, None] * eps

        return (self.estimator(theta_t, x, t, **kwargs) - eps).square().mean()
