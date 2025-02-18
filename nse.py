import math
from functools import partial
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import Size, Tensor
from torch.distributions import Distribution
from tqdm import tqdm
from zuko.distributions import DiagNormal, NormalizingFlow
from zuko.nn import MLP
from zuko.transforms import FreeFormJacobianTransform
from zuko.utils import broadcast
from torch.func import vmap


from embedding_nets import FNet, GaussianNet, GaussianNetAlpha
from tall_posterior_sampler import (
    prec_matrix_backward,
    tweedies_approximation,
    tweedies_approximation_prior,
)


class NSE(nn.Module):
    def __init__(
        self,
        theta_dim: int,
        x_dim: int,
        freqs: int = 3,
        build_net: Callable[[int, int], nn.Module] = MLP,
        net_type: str = "default",
        embedding_nn_theta: nn.Module = nn.Identity(),
        embedding_nn_x: nn.Module = nn.Identity(),
        **kwargs: dict,
    ) -> None:
        r"""Creates a neural score estimation (NSE) network.
        This class implements the neural score estimation (NSE) network parametrized
        by the VP SDE and a linear noise schedule [1]. It takes as input the parameters
        `theta`, observations `x`, and the time variable `t`, and outputs the estimated
        noise `epsilon(theta, x, t)`, that is related to the score function via
        `score(theta, x, t) = -sigma(t) * epsilon(theta, x, t)`, where `sigma(t)` is the
        standard deviation of the transition kernel of the VP SDE.

        The network consists of two optional embedding networks for the parameters `theta` and
        observations `x`, respectively, and a positional encoding of the time variable `t`.
        The embeddings are concatenated with the positional encoding and passed through a
        final score network that outputs the estimated noise.

        `NSE` also implements different sampling algorithms to infer the posterior distribution
        in the classical and in the tall data setting (for multiple context observations `x`):
        - Denoising Diffusion Implicit Models (DDIM) sampler [2],
        - Annealed Langevin dynamics as implemented in [3],
        - Predictor-Corrector (PC) sampler (generalization of DDIM and Langevin dynamics [1]
            with tamed ULA [4] for the langevin corrector step),

        References:
            [1] Song et al. (2020). Score-Based Generative Modeling through SDEs,
                https://arxiv.org/abs/2011.13456.
            [2] Song et al. (2021). Denoising Diffusion Implicit Models,
                https://arxiv.org/abs/2010.02502.
            [3] Geffner et al. (2023). Compositional Score Modeling for Simulation-Based Inference,
                https://arxiv.org/abs/2209.14249.
            [4] Brosse et al. (2017). The Tamed Unadjusted Langevin Algorithm,
                https://inria.hal.science/hal-01648667/document.
        Args:
            theta_dim: The dimensionality `m` of the parameter space.
            x_dim: The dimensionality `d` of the observation space.
            freqs: The number of time embedding frequencies, default is 3.
            build_net: The network constructor. It takes the number of input and
            output features as positional arguments. Default is a simple MLP.
            net_type: The type of final score network. Can be 'default' or 'fnet'.
            embedding_nn_theta: The embedding network for the parameters `theta`.
                Default is the identity function.
            embedding_nn_x: The embedding network for the observations `x`.
                Default is the identity function.
            kwargs: Keyword arguments passed to the network constructor `build_net`.
        """
        super().__init__()

        self.embedding_nn_theta = embedding_nn_theta
        self.embedding_nn_x = embedding_nn_x
        self.theta_emb_dim, self.x_emb_dim = self.get_theta_x_embedding_dim(
            theta_dim, x_dim
        )
        self.net_type = net_type
        self.n_max = 1  # used in PF_NSE

        if net_type == "default":
            self.net = build_net(
                self.theta_emb_dim + self.x_emb_dim + 2 * freqs, theta_dim, **kwargs
            )
            
        elif net_type == "fnet":
            self.net = FNet(
                dim_input=theta_dim, dim_cond=x_dim, dim_embedding=128, n_layers=1
            )

        elif net_type == "gaussian" or net_type == "analytical":
            self.net = GaussianNet(
                hidden_dim=16, output_dim=theta_dim
            )

        elif net_type == "gaussian_alpha":
            self.net = GaussianNetAlpha(hidden_dim=10)

        else:
            raise NotImplementedError("Unknown net_type")

        self.tweedies_approximator = tweedies_approximation
        self.register_buffer("freqs", torch.arange(1, freqs + 1) * math.pi) #return [1,2,3]*pi
        #self.freqs=freqs
        
        self.register_buffer("zeros", torch.zeros(theta_dim))
        self.register_buffer("ones", torch.ones(theta_dim))

    def get_theta_x_embedding_dim(self, theta_dim: int, x_dim: int) -> int:
        r"""Returns the dimensionality of the embeddings for `\theta` and `x`."""
        theta, x = torch.ones((1, theta_dim)), torch.ones((1, x_dim))
        return (
            self.embedding_nn_theta(theta).shape[-1],
            self.embedding_nn_x(x).shape[-1],
        )

    def forward(self, theta: Tensor, x: Tensor, t: Tensor) -> Tensor:
        r"""
        Args:
            theta: The parameters `\theta`, of shape `(*, m)`.
            x: The observation `x`, of shape `(*, d)`.
            t: The time `t`, of shape `(*,).`

        Returns:
           : The estimated noise `epsilon(theta, x, t)`, of shape `(*, m)`.
        """

        if self.net_type == "default":
            if 0:
                # compute sinusoidal positional encoding for `t`
                # sin(t * exp(-log(10000)/dim * i)) where i=[0,1...,dim]
                # cos(t * exp(-log(10000)/dim * i)) where i=[0,1...,dim]
                # magic number 10000 is from transformers
                max_positions=10000
                half_dim = self.freqs #nb of freqs to compute the sin and cos
                emb = math.log(max_positions) / (half_dim - 1)
                emb = torch.exp(-torch.arange(half_dim) * emb)
                #print("forward t ", t[..., None].size())
                #print(t.ndim)
                #print("freqs", emb[None, :].size())
                # if t.ndim==0:
                #     t = t.unsqueeze(0)
                #emb = t[:, None] * emb[None, :]
                emb = emb * t[..., None]
                t = torch.cat((emb.cos(), emb.sin()), dim=-1)
                #print("t", t.size())
            else:
                #print("freqs",self.freqs.size())
                t = self.freqs * t[..., None]
                t = torch.cat((t.cos(), t.sin()), dim=-1)
           
            # compute embeddings for `theta` and `x`
            theta = self.embedding_nn_theta(theta)
            x = self.embedding_nn_x(x)

            # broadcast `theta`, `x`, and `t` to the same shape
            theta, x, t = broadcast(theta, x, t, ignore=1)
            #print("size t",t.size())
            # concatenate variables and output the estimated noise
            return self.net(torch.cat((theta, x, t), dim=-1))

        elif self.net_type == "fnet":
            return self.net(theta, x, t)
        
        elif self.net_type == "gaussian" or self.net_type == "analytical" or self.net_type == "gaussian_alpha":
            return self.net(theta,x,t)
        
    # The following function define the VP SDE with linear noise schedule beta(t):
    # dtheta = f(t) theta dt + g(t) dW = -0.5 * beta(t) theta dt + sqrt(beta(t)) dW

    def score(self, theta: Tensor, x: Tensor, t: Tensor, **kwargs) -> Tensor:
        #the score is related to the noise estimator
        return -self(theta, x, t) / self.sigma(t)
    

    def beta(self, t: Tensor) -> Tensor:
        r"""Linear noise schedule of the VP SDE: `beta(t) = 32*t`."""
        return 32 * t #beta_min=0 et beta_max=32

    def f(self, t: Tensor) -> Tensor:
        """Drift of the VP SDE: `f(t) = -0.5 * beta(t)`."""
        return -0.5 * self.beta(t)

    def g(self, t: Tensor) -> Tensor:
        """`g(t) = sqrt(beta(t))`."""
        return torch.sqrt(self.beta(t))

    def alpha(self, t: Tensor) -> Tensor: #see SDE paper eq 29 p15 (should be multiplied by theta_0)
        r"""! Square root ! of Mean of the transition kernel of the VP SDE:
        `alpha(t) = \exp ( -0.5 \int_0^t beta(s)ds)`.
        """
        return torch.exp(-16 * t**2) #should be -8*t**2

    def sigma(self, t: Tensor) -> Tensor:
        r"""Standard deviation of the transition kernel of the VP SDE:
        `sigma^2(t) = 1 - \exp( - \int_0^t beta(s)ds) + C,
        where C is such that  `sigma^2(1) = 1, sigma^2(0)  = epsilon \approx 1e-4`.
        """
        return torch.sqrt(1 - self.alpha(t) + math.exp(-16))

    def ode(self, theta: Tensor, x: Tensor, t: Tensor, **kwargs) -> Tensor:
        r"""The probability flow ODE corresponding to the VP SDE."""
        return self.f(t) * theta + self.g(t) ** 2 / 2 * self(
            theta, x, t, **kwargs
        ) / self.sigma(t)

    def flow(self, x: Tensor, **kwargs) -> Distribution:
        r"""
        Args:
            x: observation `x`, of shape `(*, d)`.
            kwargs: additional args for the forward method.

        Returns:
            (zuko.distributions.Distribution): The normalizing flow
                `p_\phi(\theta | x)` induced by the probability flow ODE.
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

    def mean_pred(
        # corresponds to the denoised image f_theta
        self, theta: Tensor, score: Tensor, alpha_t: Tensor, **kwargs
    ) -> Tensor:
        """Mean predictor of the backward kernel
        (used in DDIM sampler and gaussian approximation).
        """

        upsilon = 1 - alpha_t
        mean = (alpha_t ** (-0.5)) * (theta + upsilon * score)
        return mean

    def bridge_mean( 
        #compute the mean of the reverse process
        #est_noise corresponds to epsilon_theta
        self,
        alpha_t: Tensor,
        alpha_t_1: Tensor,
        theta_t: Tensor,
        theta_0: Tensor,
        bridge_std: float,
    ) -> Tensor:
        """Bridge mean for the DDIM sampler."""
        est_noise = (theta_t - (alpha_t**0.5) * theta_0) / ((1 - alpha_t) ** 0.5)
        return (alpha_t_1**0.5) * theta_0 + (
            (1 - alpha_t_1 - bridge_std**2) ** 0.5
        ) * est_noise

    def ddim(
        self,
        shape: Size,
        x: Tensor,
        steps: int = 1000,
        eta: float = 1.0,
        verbose: bool = False,
        theta_clipping_range=(None, None),
        **kwargs,
    ) -> Tensor:
        r"""Sampler from Denoising Diffusion Implicit Models [1],  but adapted to
        the tall data setting with `n` context observations `x`.

        References:
            [1] Song et al. (2021). Denoising Diffusion Implicit Models,
                https://arxiv.org/abs/2010.02502.

        Args:
            shape (torch.Size): The shape of the samples.
            x: The conditioning variable for the score network, of shape `(n, m)`.
            steps: The number of steps in the diffusion process, default is 1000.
            eta: The noise level for the bridge process, default is 1.0.
            verbose (bool): If True, displays a progress bar, default is False.
            theta_clipping_range (Tuple[float, float]): The range for clipping the samples.
                Default is `(None, None)`.
            kwargs: Additional args for the score function.

        Returns:
           : The samples from the diffusion process, of shape `(shape[0], m)`.
        """
        if x.shape[0] == shape[0] or len(x.shape) == 1 or x.shape[0] == 1:
            score_fun = partial(self.score, **kwargs)
        else: #call the factorized score function if nb of obs >1
            score_fun = partial(self.factorized_score, **kwargs)
        eps_s=1e-2
        time = eps_s+(1-eps_s)*torch.linspace(1, 0, steps + 1).to(x)
        dt = 1 / steps
        
        #on part de q_T=N(0,I) puis on fait le reverse process jusqu'à t=0
        theta = DiagNormal(self.zeros, self.ones).sample(shape)

        for t in tqdm(time[:-1], disable=not verbose):
            #theta will follow p_t at each iteration
            theta = self.ddim_step(theta, x, t, score_fun, dt, eta)
            if theta_clipping_range[0] is not None:
                theta = theta.clip(*theta_clipping_range)
        return theta

    def ddim_step(
        self,
        theta: Tensor,
        x: Tensor,
        t: Tensor,
        score_fun: Callable[[Tensor, Tensor, Tensor], Tensor],
        dt: float,
        eta: float,
        **kwargs,
    ):
        r"""One step of the DDIM sampler."""
        # size (num_samples, dim theta)
        
        score = score_fun(theta, x, t).detach() #compute the score of tall posterior
        alpha_t = self.alpha(t)
        alpha_t_1 = self.alpha(t - dt)
        bridge_std = eta * (
            (((1 - alpha_t_1) / (1 - alpha_t)) * (1 - alpha_t / alpha_t_1)) ** 0.5
        ) #corresponds to sigma_t, std of the reverse transition kernels
        # if eta=1 DDPM reverse process, if eta=0 DDIM reverse process

        #predicts the denoised theta from theta_t ie f(theta_t)=hat(theta_0)
        # size (num_samples, dim theta)
        pred_theta_0 = self.mean_pred(theta=theta, score=score, alpha_t=alpha_t)

        theta_mean = self.bridge_mean( #compute the mean of the reverse transition kernel q_sigma(t-1|t)
            alpha_t=alpha_t,
            alpha_t_1=alpha_t_1,
            theta_0=pred_theta_0,
            theta_t=theta,
            bridge_std=bridge_std,
        )# size (num_samples, dim theta)
        #theta ~ N(theta_mean,bridge_std**2I)=q_sigma(t-1|t)

        theta = theta_mean + torch.randn_like(theta_mean) * bridge_std
        return theta

    def langevin_corrector(
        self,
        theta: Tensor,
        x: Tensor,
        t: Tensor,
        score_fun: Callable[[Tensor, Tensor, Tensor], Tensor],
        lsteps: int,
        r: float,
        **kwargs,
    ) -> Tensor:
        r"""Langevin corrector for the Predictor-Corrector (PC) sampler.
        It implements one step of the tamed ULA algorithm [1].

        References:
            [1] Brosse et al. (2017). The Tamed Unadjusted Langevin Algorithm,
                https://inria.hal.science/hal-01648667/document.

        Args:
            theta: The current state of the diffusion process, of shape `(n, m)`.
            x: The conditioning variable for the score network, of shape `(n, m)`.
            t: The time `t`, of shape `(*,)`.
            score_fun: The score function.
            lsteps: The number of Langevin steps.
            r: The step size (or signal-to-noise ratio) for the Langevin dynamics.

        Returns:
           : The corrected sampple state of the diffusion process, of shape `(n, m)`.
        """

        for _ in range(lsteps):
            z = torch.randn_like(theta)
            g = score_fun(theta, x, t).detach()
            # Step size as defined in [Song et al, 2021]
            # (Alg 5 in https://arxiv.org/pdf/2011.13456.pdf with sigma^2 replacing 1/ ||g||^2)
            eps = r * (self.alpha(t) ** 0.5) * (self.sigma(t) ** 2) # pq manque-t-il 2?
            tamed_eps = (eps / (1 + eps * torch.linalg.norm(g, axis=-1)))[..., None]
            theta = theta + tamed_eps * g + ((2 * eps) ** 0.5) * z
        return theta

    def predictor_corrector(
        self,
        shape: Size,
        x: Tensor,
        steps: int = 1000,
        verbose: bool = False,
        predictor_type="ddim",
        corrector_type="langevin",
        theta_clipping_range=(None, None),
        **kwargs,
    ) -> Tensor:
        r"""Predictor-Corrector (PC) sampling algorithm [1], but adapted to
        the tall data setting with `n` context observations `x`.

        The PC sampler is a generalization of the
        - Langevin Dynamics: predictor_type = 'id', corrector_type = 'langevin'
        - DDIM sampler: predictor_type = 'ddim', corrector_type = 'id'

        References:
            [1] Song et al. (2020). Score-Based Generative Modeling through SDEs,
                https://arxiv.org/abs/2011.13456.

        Args:
            shape (torch.Size): The shape of the samples.
            x: The conditioning variable for the score network, of shape `(n, m)`.
            steps: The number of steps in the diffusion process, default is 1000.
            verbose (bool): If True, displays a progress bar, default is False.
            predictor_type: The type of predictor. Can be 'ddim' or 'id', default is 'ddim'.
            corrector_type: The type of corrector. Can be 'langevin' or 'id', default is 'langevin'.
            theta_clipping_range (Tuple[float, float]): The range for clipping the samples.
                Default is `(None, None)`.
            kwargs: Additional args for the score function, the predictor and corrector.

        Returns:
           : The samples from the diffusion process, of shape `(shape[0], m)`.
        """

        # get simple or tall data score function
        if x.shape[0] == shape[0] or len(x.shape) == 1 or x.shape[0] == 1:
            score_fun = partial(self.score, **kwargs)
        else:
            if corrector_type == "langevin":
                score_fun = partial(self.factorized_score_geffner, **kwargs)
            else:
                score_fun = partial(self.factorized_score, **kwargs)

        # get predictor and corrector functions
        if predictor_type == "ddim":
            predictor_fun = partial(self.ddim_step, **kwargs)
        elif predictor_type == "id":
            predictor_fun = lambda theta, x, t, score_fun, dt: theta
        else:
            raise NotImplemented("")
        if corrector_type == "langevin":
            corrector_fun = partial(self.langevin_corrector, **kwargs)
        elif corrector_type == "id":
            corrector_fun = lambda theta, x, t, score_fun: theta
        else:
            raise NotImplemented("")

        # run the PC sampler
        time = torch.linspace(1, 0, steps + 1).to(x)
        dt = 1 / steps
        # start from theta_T ~ N(0,I)
        theta = DiagNormal(self.zeros, self.ones).sample(shape)

        for t in tqdm(time[:-1], disable=not verbose):
            # predictor step
            theta_pred = predictor_fun( #on fait un DDIM step to get an approx sample from p_(t-dt)
                theta=theta, x=x, t=t, score_fun=score_fun, dt=dt
            )
            # corrector step
            theta = corrector_fun(theta=theta_pred, x=x, t=t - dt, score_fun=score_fun)
            #on corrige le sample to be sure it is a true sample from p_(t-dt)
            if theta_clipping_range[0] is not None:
                theta = theta.clip(*theta_clipping_range)
        return theta

    def annealed_langevin_geffner(
        self,
        shape: Size,
        x: Tensor,
        prior_score_fn: Callable[[torch.tensor, torch.tensor], torch.tensor],
        prior_type: Optional[str] = None,
        steps: int = 400,
        lsteps: int = 5,
        tau: float = 0.5,
        theta_clipping_range=(None, None),
        verbose: bool = False,
        **kwargs,
    ):
        """Annealed Langevin dynamics with optional use of the factorized posterior score
        from [1] for tall data setting.

        References:
            [1] Geffner et al. (2023). Compositional Score Modeling for Simulation-Based Inference,
                https://arxiv.org/abs/2209.14249.

        Args:
            shape: The shape of the samples.
            x: The conditioning variable for the score network, of shape `(n, m)`.
            prior_score_fn: The prior score function.
            prior_type: The type of prior for the factorized score, default is None.
            steps: The number of steps in the diffusion process, default is 400.
            lsteps: The number of Langevin steps, default is 5.
            tau: Scale of the langevin step size, default is 0.5.
            theta_clipping_range: The range for clipping the samples, default is `(None, None)`.
            verbose: If True, displays a progress bar, default is False.
            kwargs: Additional args for the score function.
        """
        eps_s=1e-2
        time = eps_s+(1-eps_s)*torch.linspace(1, 0, steps + 1).to(x)
        # start from a normal standard variable
        # t runs reversely (from 1 to 0)
        theta = DiagNormal(self.zeros, self.ones).sample(shape)
        for i, t in enumerate(tqdm(time[:-1], disable=not verbose)):
            #i is the indice of the steps and t is the corresponding t_i
            
            if i < steps - 1:
                gamma_t = self.alpha(t) / self.alpha(t - time[-2]) #correspond to t-dt
            else: #for t_0
                gamma_t = self.alpha(t)
            #tau should be 0.3 in the paper but 0.5 in julia's paper
            delta = tau * (1 - gamma_t) / (gamma_t**0.5)
            for _ in range(lsteps):#at the end of the loop we get a sample from p_t
                z = torch.randn_like(theta) #sample from normal standard
                
                if len(x.shape) == 1 or x.shape[0] == 1:
                    score = self.score(theta, x, t, **kwargs).detach()
                else:
                    score = self.factorized_score_geffner(#if we have several cond obs)
                        theta, x, t, prior_score_fn, prior_type=prior_type, **kwargs
                    ).detach().squeeze(0)
                theta = theta + delta * score + ((2 * delta) ** 0.5) * z
            if theta_clipping_range[0] is not None:
                theta = theta.clip(*theta_clipping_range)

        return theta

    # The following functions are used for samplers for the tall data setting
    # with n context observations x: they implement the factorized posterior score.

    def factorized_score_geffner(
        self,
        theta: Tensor,
        x: Tensor,
        t: Tensor, # a real value 
        prior_score_fun: Callable[[Tensor, Tensor], Tensor] = None,
        clf_free_guidance: bool = False,
        **kwargs,
    ):
        r"""Factorized score function for the tall data setting with n context observations x.
        Corresponds to the `F-NPSE` method from Geffner et al. (2023) [1]. Optionally, it can
        use the prior score learned via the classifier-free guidance approach [2].

        References:
            [1] Geffner et al. (2023). Compositional Score Modeling for Simulation-Based Inference,
                https://arxiv.org/abs/2209.14249.
            [2] Ho et al. (2022). Classifier-Free Diffusion Guidance,
                https://arxiv.org/abs/2207.12598.

        Args:
            theta: The posterior samples `theta`, of shape `(n, m)`.
            x: The observation `x`, of shape `(n, d)`.
            t: The time `t`, of shape `(*,)`.
            prior_score_fun: The analytical prior score function.
            clf_free_guidance: If True, we use the prior score learned via the classifier-free guidance
            approach [1], default is False.
            kwargs: Additional args for the score function.

        Returns:
            : The factorized score, of shape `(n, m)` for time t "ONLY".
        """
        # Defining variables
        n_observations = x.shape[0] if len(x.shape) > 1 else 1
        n_samples = theta.shape[0]
        # Calculating m, Sigma and scores for the posteriors
        # approx the noise espsilon and relate to the score via score_t=-espilon_t/sigma_t
        if self.net_type == "fnet":
            scores = self(
                theta[:, None, :]
                .repeat(1, n_observations, 1)
                .reshape(n_observations * n_samples, -1),
                x[None, :, :]
                .repeat(n_samples, 1, 1)
                .reshape(n_observations * n_samples, -1),
                t[None, None].repeat(n_samples * n_observations, 1),
                **kwargs,
            ) / -self.sigma(t)
            scores = scores.reshape(n_samples, n_observations, -1)
        
        elif self.net_type == "analytical":

            #scores = self.score(theta[:, None], x[None, :], t).detach() 
            
            scores = vmap(
                    lambda theta: vmap(
                        partial(self.score, t=t),
                        in_dims=(None, 0),
                        randomness="different",
                    )(theta, x),
                    randomness="different",
                )(theta)
            
        elif self.net_type == "gaussian" or self.net_type == "gaussian_alpha":
            scores = self.score(theta[:, None], x[None, :], t).detach()  
            
        else:
            scores = self.score(theta[:, None], x[None, :], t, **kwargs).detach() 
            #size : (n_samples, n_observations, dim theta)
            # if torch.isnan(scores[0,0,0]):
            #     print(t)
                #print("post scores",scores[:5,:])
        if clf_free_guidance:
            x_ = torch.zeros_like(x[0])
            if "n" in kwargs:
                kwargs_score_prior = {"n": torch.zeros_like(kwargs["n"][0][None, :])}
            prior_score = self.score(
                theta[:, None], x_[None, :], t, **kwargs_score_prior
            ).detach()[:, 0, :]
        else:
            prior_score = prior_score_fun(theta[None], t)#[0]
            # size (n-samples, dim theta)
            # if torch.isnan(prior_score[0,0]):
            #     print(t)
                #print("facct score prior",prior_score[:5,:])
        aggregated_score = (1 - n_observations) * prior_score + scores.sum(axis=1) #why no (T-t) power as in Geffner ?
        # ATTENTION change here for the score
        #aggregated_score = (1 - n_observations) * (1-t) * prior_score + scores.sum(axis=1) #why no (T-t) power as in Geffner ?
        return aggregated_score

    def factorized_score(
        self,
        theta,
        x_obs,
        t,
        prior_score_fn,
        prior,
        dist_cov_est=None,
        dist_cov_est_prior=None,
        cov_mode="GAUSS",
        prior_type="gaussian",
        clf_free_guidance=False,
        **kwargs,
    ):
        r"""Factorized score function for the tall data setting with n context observations x.
        Corresponds to our proposition ("GAUSS" and "JAC"). Optionally, it can use the prior score
        learned via the classifier-free guidance approach [1].

        References:
            [1] Ho et al. (2022). Classifier-Free Diffusion Guidance,
                https://arxiv.org/abs/2207.12598.

        Args:
            theta: The posterior samples `theta`, of shape `(n, m)`.
            x_obs: The observation `x`, of shape `(n, d)`.
            t: The time `t`, of shape `(*,)`.
            prior_score_fn: The analytical prior score function.
            prior: The prior distribution.
            dist_cov_est: The "backward covariance matrix" of the posterior scores,
                used for tweedies approximation, default is None.
            dist_cov_est_prior: The "bakckward covariance matrix" of the prior scores
                used for tweedies approximation, default is None.
            cov_mode: The mode for the covariance estimation, can be 'GAUSS' or 'JAC',
                default is 'GAUSS'.
            prior_type: The type of prior for the factorized score, default is 'gaussian'.
            clf_free_guidance: If True, we use the prior score learned via the classifier-free guidance
            approach [1], default is False.
            kwargs: Additional args for the score function.

        Returns:
            : The factorized score, of shape `(n, m)`.
        """
        # device
        n_obs = x_obs.shape[0]
        prec_0_t, _, scores = self.tweedies_approximator( #get Sigma_t,j^-1 and s_j for all j
            x=x_obs,
            theta=theta,
            nse=self,
            t=t,
            score_fn=partial(self.score, **kwargs),
            dist_cov_est=dist_cov_est,
            mode=cov_mode,
        )
        # scores of size (n_samples, n_obs, dim theta)
        # Sigma_j,t of size (n_samples, n_obs, dim theta x dim theta)

        if clf_free_guidance:
            x_ = torch.zeros_like(x_obs[0][None, :])
            if "n" in kwargs:
                if kwargs["n"] is not None:
                    kwargs_score_prior = {
                        "n": torch.zeros_like(kwargs["n"][0][None, :])
                    }
                else:
                    kwargs_score_prior = {"n": None}
            prec_prior_0_t_cfg, _, prior_score_cfg = self.tweedies_approximator(
                x=x_,
                theta=theta,
                nse=self,
                t=t,
                score_fn=partial(self.score, **kwargs_score_prior),
                dist_cov_est=dist_cov_est_prior,
                mode=cov_mode,
            )
            prec_score_prior_cfg = (prec_prior_0_t_cfg @ prior_score_cfg[..., None])[
                ..., 0
            ][:, 0, :]
            prec_score_post = (prec_0_t @ scores[..., None])[..., 0]
            lda_cfg = prec_prior_0_t_cfg[:, 0, :] * (1 - n_obs) + prec_0_t.sum(dim=1)
            weighted_scores_cfg = prec_score_prior_cfg + (
                prec_score_post - prec_score_prior_cfg[:, None]
            ).sum(dim=1)

            total_score = torch.linalg.solve(A=lda_cfg, B=weighted_scores_cfg)

        else:
            if prior_type == "gaussian":
                prior_score = prior_score_fn(theta, t)
                prec_prior_0_t = prec_matrix_backward(t, prior.covariance_matrix, self)
                prec_score_prior = (prec_prior_0_t @ prior_score[..., None])[..., 0]
                prec_score_post = (prec_0_t @ scores[..., None])[..., 0]
                lda = prec_prior_0_t * (1 - n_obs) + prec_0_t.sum(dim=1)
                weighted_scores = prec_score_prior + (
                    prec_score_post - prec_score_prior[:, None]
                ).sum(dim=1)
                
                total_score = torch.linalg.solve(A=lda, B=weighted_scores)

            else: #case when the prior is not gaussian
                prior_score = prior_score_fn(theta, t) # size (n-samples, dim theta)
                total_score = (1 - n_obs) * prior_score + scores.sum(dim=1)
                if (self.alpha(t) ** 0.5 > 0.5) and (n_obs > 1):
                    prec_prior_0_t, _, _ = tweedies_approximation_prior(# compute Sigma_lambda,t with JAC mode
                        theta, t, prior_score_fn, nse=self
                    )# size (n_samples, dim theta x dim theta)
                    prec_score_prior = (prec_prior_0_t @ prior_score[..., None])[..., 0]
                    prec_score_post = (prec_0_t @ scores[..., None])[..., 0]
                    lda = prec_prior_0_t * (1 - n_obs) + prec_0_t.sum(dim=1)
                    weighted_scores = prec_score_prior + (
                        prec_score_post - prec_score_prior[:, None]
                    ).sum(dim=1)
            
                    total_score = torch.linalg.solve(A=lda, B=weighted_scores)# size (n_samples, dim theta)
        return total_score  # / (1 + (1/n_obs)*torch.abs(total_score))


class NSELoss(nn.Module):
    r"""Calculates the *noise parametrized* denoising score matching (DSM) loss for NSE.
    Minimizing this loss estimates the noise  `eplison`, from which the score function
    can be calculated as

        `score(\theta, x, t) = - \sigma(t) * epsilon(\theta, x, t)`.

    Given a batch of `N` pairs `(theta_i, x_i)`, the module returns `
        mean(
            ||epsilon(alpha(t_i) theta_i + sigma(t_i) epsilon_i, x_i, t_i)- \epsilon_i||^2
        )

    where `t_i ~ U(0, 1)` and `epsilon_i ~ N(0, I)`.

    Args:
        estimator (NSE): A regression network `epsilon(\theta, x, t)`.
    """

    def __init__(self, estimator: NSE):
        super().__init__()

        self.estimator = estimator

    def forward(self, theta: Tensor, x: Tensor, **kwargs) -> Tensor:
        r"""
        Args:
            theta: The parameters `theta`, of shape `(N, m)`.
            x: The observation `x`, of shape `(N, d)`.
            kwargs: Additional args for the forward method of the estimator.

        Returns:
           : The noise parametrized scalar DSM loss `l`.
        """
        eps_t=1e-5
        t = torch.rand(theta.shape[0], dtype=theta.dtype, device=theta.device)#uniform t in eps_t,1        
        scaling = self.estimator.alpha(t) ** 0.5 #mean of the transition kernel from 0 to t
        sigma = self.estimator.sigma(t) #std of the transition kernel from 0 to t

        eps = torch.randn_like(theta)
        theta_t = scaling[:, None] * theta + sigma[:, None] * eps # corresp to the diffused theta_0 at time t

        return (self.estimator(theta_t, x, t, **kwargs) - eps).square().mean() #loss of the denoiser SM
        #return (((1-scaling**2)**0.5)[...,None]*self.estimator(theta_t, x, t, **kwargs) + eps).square().mean() #loss of the denoiser SM


class ExplicitLoss(nn.Module):
    
    def __init__(self, inv_cov_prior, mu_prior, inv_cov_lik, cov_post, estimator: NSE):
        super().__init__()

        self.estimator = estimator
        self.cov_post=cov_post
        self.inv_cov_lik=inv_cov_lik
        self.inv_cov_prior = inv_cov_prior
        self.mu_prior = mu_prior

    def forward(self, theta: Tensor, x: Tensor, **kwargs) -> Tensor:
        r"""
        Args:
            theta: The parameters `theta`, of shape `(N, m)`.
            x: The observation `x`, of shape `(N, d)`.
            kwargs: Additional args for the forward method of the estimator.

        Returns:
           : The noise parametrized scalar DSM loss `l`.
        """
        eps_t=1e-3
        t = eps_t+(1-eps_t)*torch.rand(theta.shape[0], dtype=theta.dtype, device=theta.device)#uniform t in eps_t,1
        scaling = self.estimator.alpha(t) ** 0.5 #mean of the transition kernel from 0 to t
        sigma = self.estimator.sigma(t) #std of the transition kernel from 0 to t
        eps = torch.randn_like(theta)
        sigma_2 = sigma**2
        scaling_2 = scaling**2
        cov_diff = sigma_2[...,None,None]*torch.eye(theta.shape[1]).repeat(theta.shape[0],1,1)
        cov_diff+= scaling_2[...,None,None]*self.cov_post.repeat(theta.shape[0],1,1)
        #size (batch size,2,2)
        inv_cov_diff = torch.linalg.inv(cov_diff)

        theta_t = scaling[:, None] * theta + sigma[:, None] * eps # corresp to the diffused theta_0 at time t
        mu_post = self.cov_post@(self.inv_cov_lik@x.reshape(2,x.size(0)) + self.inv_cov_prior@self.mu_prior.reshape(2,1))
        #size (2,batch size)
        rescaling = theta_t-scaling[...,None]*torch.transpose(mu_post,0,1)
        tmp = (self.estimator(theta_t, x, t, **kwargs) + (inv_cov_diff@rescaling[...,None]).squeeze()).square()
        #return (self.estimator(theta_t, x, t, **kwargs) + (inv_cov_diff@rescaling[...,None]).squeeze()).square().mean() #loss of the denoiser SM
        return (sigma_2[...,None]*tmp).mean() #loss of the denoiser SM

if __name__ == "__main__":
    theta = torch.randn(128, 2)
    x = torch.randn(10, 2)
    t = torch.rand(1)
    nse = NSE(2, 2)

    nse.predictor_corrector(
        (128,),
        x=x,
        steps=2,
        prior_score_fun=lambda theta, t: torch.ones_like(theta),
        eta=0.01,
        lsteps=2,
        r=0.5,
        predictor_type="ddim",
        verbose=True,
    ).cpu()
