from functools import partial

import torch
from torch.func import jacrev, vmap
from tqdm import tqdm
import numpy as np

from vp_diffused_priors import get_vpdiff_gaussian_score


def mean_backward(theta, t, score_fn, nse, **kwargs): #compute the mean of the backward kernel
    alpha_t = nse.alpha(t)
    sigma_t = nse.sigma(t) #1-alpha_t

    return (
        1
        / (alpha_t**0.5)
        * (theta + sigma_t**2 * score_fn(theta=theta, t=t, **kwargs))
    )


def sigma_backward(t, dist_cov, nse):
    #return the cov of the backward kernel
    #dist_cov is the inverse approximate posterior cov at t=0
    alpha_t = nse.alpha(t)
    sigma_t = nse.sigma(t)
    eye = torch.eye(dist_cov.shape[-1]).to(alpha_t.device)
    # Same as the other but using woodberry. (eq 53 or https://arxiv.org/pdf/2310.06721.pdf)
    if (alpha_t**0.5) < 0.5:
        return torch.linalg.inv(
            torch.linalg.inv(dist_cov) + (alpha_t / sigma_t**2) * eye
        ) #if q_data is a gaussian distribution
    return ((sigma_t**2) / alpha_t) * eye - (
        ((sigma_t**2) ** 2 / alpha_t)
    ) * torch.linalg.inv(alpha_t * (dist_cov.to(alpha_t.device) - eye) + eye)


def prec_matrix_backward(t, dist_cov, nse):
    #dist_cov is the inverse cov of the t=0 individual posterior
    #return the inv of diffused posterior cov at time t if qdata is gaussian (GAUSS)
    alpha_t = nse.alpha(t)
    sigma_t = nse.sigma(t) #corresponds to 1-alpha_t
    eye = torch.eye(dist_cov.shape[-1]).to(alpha_t.device)
    # Same as the other but using woodberry. (eq 53 or https://arxiv.org/pdf/2310.06721.pdf)
    # return torch.linalg.inv(dist_cov * alpha_t + (sigma_t**2) * eye)
    return (
        torch.linalg.inv(dist_cov.to(alpha_t.device)) + (alpha_t / sigma_t**2) * eye
    )


def tweedies_approximation(
        #return the backward mean, backward cov for GAUSS and JAC and the score
    x,
    theta,
    t,
    score_fn,
    nse,
    dist_cov_est=None,
    mode="JAC",
    clip_mean_bounds=(None, None),
    partial_factorization=False,
):
    alpha_t = nse.alpha(t)
    sigma_t = nse.sigma(t) #correspond to v_t**0,5 or (1-alpha_t)**0,5

    if mode == "JAC":
        if nse.net_type == "fnet":

            def score_jac(theta, x):
                score = score_fn(theta=theta[None, :], t=t[None, None], x=x[None, ...])[
                    0
                ]
                return score, score

            jac_score, score = vmap(
                lambda theta: vmap(jacrev(score_jac, has_aux=True))(
                    theta[None].repeat(x.shape[0], 1), x
                )
            )(theta) #compute the jacobian of the score
        
        else:

            def score_jac(theta, x):
                score = score_fn(theta=theta, t=t, x=x)
                return score, score

            if not partial_factorization:
                jac_score, score = vmap(
                    lambda theta: vmap(
                        jacrev(score_jac, has_aux=True), in_dims=(None, 0)
                    )(theta, x)
                )(theta)
            else:
                jac_score, score = vmap(
                    jacrev(score_jac, has_aux=True), in_dims=(None, 0)
                )(theta, x)
        
        if sigma_t**2==1:
            sigma_t-=1e-6

        cov = (sigma_t**2 / alpha_t) * (
            torch.eye(theta.shape[-1], device=theta.device) + (sigma_t**2) * jac_score
        )
        #compute the backward cov Sigma_t,j^-1 for the indiv posterior p(theta|xj) JAC algo
        prec = torch.linalg.inv(cov) 
    elif mode == "GAUSS":

        if nse.net_type == "fnet":
            
            def score_jac(theta, x):
                
                n_theta = theta.shape[0]
                n_x = x.shape[0]
                score = score_fn(
                    theta=theta[:, None, :]
                    .repeat(1, n_x, 1)
                    .reshape(n_theta * n_x, -1),
                    t=t[None, None].repeat(n_theta * n_x, 1).squeeze(),
                    x=x[None, ...].repeat(n_theta, 1, 1).reshape(n_theta * n_x, -1),
                )
                score = score.reshape(n_theta, n_x, -1)
                return score, score
            
            score = score_jac(theta, x)[0]

        else:
            if not partial_factorization:  
                score = vmap(
                    lambda theta: vmap(
                        partial(score_fn, t=t),
                        in_dims=(None, 0),
                        randomness="different",
                    )(theta, x),
                    randomness="different",
                )(theta)

            else:
                score = score_fn(theta[:, None], x[None, :], t)
        #compute Sigma_t,j^-1 when initial data is gaussian and cov matrix is estimated
        prec = prec_matrix_backward(t=t, dist_cov=dist_cov_est, nse=nse)
        # # Same as the other but using woodberry. (eq 53 or https://arxiv.org/pdf/2310.06721.pdf)
        # cov = torch.linalg.inv(torch.linalg.inv(dist_cov_est) + (alpha_t / sigma_t ** 2) * eye)
        prec = prec[None].repeat(theta.shape[0], 1, 1, 1) #return the Sigmat,j^-1 (GAUSS)

    else:
        raise NotImplemented("Available methods are GAUSS, PSEUDO, JAC")
    # tweedie approx for the mean of the backward kernels
    mean = 1 / (alpha_t**0.5) * (theta[:, None] + sigma_t**2 * score)
    if clip_mean_bounds[0]:
        mean = mean.clip(*clip_mean_bounds)
    return prec, mean, score


def tweedies_approximation_prior(theta, t, score_fn, nse, mode="vmap"):
    #use jac approximation here, don't suppose the prior is gaussian
    #return Sigma_lambda,t^-1, mean of backward kernel for the prior and the score s_lambda
    alpha_t = nse.alpha(t)
    sigma_t = nse.sigma(t)
    if mode == "vmap":

        def score_jac(theta):
            score = score_fn(theta=theta, t=t)
            return score, score

        jac_score, score = vmap(jacrev(score_jac, has_aux=True))(theta) #compute the jacobian of the score
    else:
        raise NotImplemented
    mean = 1 / (alpha_t**0.5) * (theta + sigma_t**2 * score)
    cov = (sigma_t**2 / alpha_t) * (
        torch.eye(theta.shape[-1], device=theta.device) + (sigma_t**2) * jac_score
    )
    prec = torch.linalg.inv(cov)
    return prec, mean, score


def diffused_tall_posterior_score(
    theta,
    t,
    prior,
    prior_score_fn,
    x_obs,
    nse,
    score_fn=None,
    dist_cov_est=None,
    cov_mode="JAC",
    prior_type="gaussian",
):
    n_obs = x_obs.shape[0]

    # Tweedies approx for p_{0|t}
    prec_0_t, _, scores = tweedies_approximation( #get the scores sj and Sigma,t,j for all j
        x=x_obs,
        theta=theta,
        nse=nse,
        t=t,
        score_fn=nse.score if score_fn is None else score_fn,
        dist_cov_est=dist_cov_est, #this estimate is for GAUSS algo only, corresp to the cov of t=0 indiv posterior
        mode=cov_mode,
    )
    prior_score = prior_score_fn(theta, t)

    if prior_type == "gaussian":
    
        prec_prior_0_t = prec_matrix_backward(t, prior.covariance_matrix, nse).repeat(#use the formula for a gaussian initial distribution
            theta.shape[0], 1, 1
        ) #return diff inv cov at time t (Sigma_t,lambda^-1) if initial data is gaussian

        prec_score_prior = (prec_prior_0_t @ prior_score[..., None])[..., 0] #Sigma_lambda,t^-1 @ s_lambda
        
        prec_score_post = (prec_0_t @ scores[..., None])[..., 0] # Sigma_t,j^-1 @ s_j for all j

        lda = prec_prior_0_t * (1 - n_obs) + prec_0_t.sum(dim=1) #big lambda
        weighted_scores = prec_score_prior + (
            prec_score_post - prec_score_prior[:, None] #corresp to tilde s (algo GAUSS)
        ).sum(dim=1)
        total_score = torch.linalg.solve(A=lda, B=weighted_scores)

    else:
        total_score = (1 - n_obs) * prior_score + scores.sum(dim=1) #why ?
        if (nse.alpha(t) ** 0.5 > 0.5) and (n_obs > 1):
            prec_prior_0_t, _, _ = tweedies_approximation_prior( #get Sigma_lambda,t^-1 (JAC algo)
                theta, t, prior_score_fn, nse
            )
            prec_score_prior = (prec_prior_0_t @ prior_score[..., None])[..., 0]#Sigma_lambda,t^-1 @ s_lambda
            prec_score_post = (prec_0_t @ scores[..., None])[..., 0]#  Sigma_t,j^-1 @ s_j for all j
            lda = prec_prior_0_t * (1 - n_obs) + prec_0_t.sum(dim=1) #big lambda
            weighted_scores = prec_score_prior + (
                prec_score_post - prec_score_prior[:, None]
            ).sum(dim=1) #corresp to tilde_s
            
            total_score = torch.linalg.solve(A=lda, B=weighted_scores)

    return total_score  # / (1 + (1/n_obs)*torch.abs(total_score))


def euler_sde_sampler(
    score_fn, #compute the score of the tall diffused posterior at times t
    nsamples,
    dim_theta,
    beta,
    device="cpu",
    debug=False,
    theta_clipping_range=(None, None),
):
    """
    Sample from the true tall posterior with reverse SDE and euler step
    """
    #start with a gaussian sample from p_T(theta|x)
    eps_s=1e-2
    theta_t = torch.randn((nsamples, dim_theta)).to(device)  # (nsamples, 2)
    time_pts = eps_s + (1-eps_s)*torch.linspace(1, 0, 1000).to(device)  # (ntime_pts,)
    theta_list = [theta_t]
    gradlogL_list = []
    lda_list = []
    posterior_scores_list = []
    means_posterior_backward_list = []
    sigma_posterior_backward_list = []
    for i in tqdm(range(len(time_pts) - 1)):
        t = time_pts[i]
        dt = time_pts[i + 1] - t
        # calculate the drift and diffusion terms
        f = -0.5 * beta(t) * theta_t
        g = beta(t) ** 0.5
        if debug:
            (
                score,
                gradlogL,
                lda,
                posterior_scores,
                means_posterior_backward,
                sigma_posterior_backward,
            ) = score_fn(theta_t, t, debug=True)
        else:
            score = score_fn(theta=theta_t, t=t)
        score = score.detach()
        drift = f - g * g * score #reverse SDE eq (6) p.4
        diffusion = g

        # euler-maruyama step
        # theta_t = theta_t-1 + dt*drift + diffusion*Z with Z from N(0,dtI)
        theta_t = (
            theta_t.detach()
            + drift * dt
            + diffusion * torch.randn_like(theta_t) * torch.abs(dt) ** 0.5
        )
        if theta_clipping_range[0] is not None:
            theta_t = theta_t.clip(*theta_clipping_range)
        theta_list.append(theta_t.detach().cpu())
        if debug:
            gradlogL_list.append(gradlogL)
            lda_list.append(lda)
            posterior_scores_list.append(posterior_scores)
            means_posterior_backward_list.append(means_posterior_backward)
            sigma_posterior_backward_list.append(sigma_posterior_backward)

    theta_list[0] = theta_list[0].detach().cpu()
    if debug:
        return (
            theta_t,
            torch.stack(theta_list),
            torch.stack(gradlogL_list),
            torch.stack(lda_list),
            torch.stack(posterior_scores_list),
            torch.stack(means_posterior_backward_list),
            torch.stack(sigma_posterior_backward_list),
        )
    else:
        return theta_t, theta_list
    
def heun_ode_sampler(
    score_fn, #compute the score of the tall diffused posterior at times t
    nsamples,
    dim_theta,
    beta,
    device="cpu",
    theta_clipping_range=(None, None),
):
    """
    Sample from the true tall posterior with reverse ODE (probability flow) and 2nd order scheme
    """
    #start with a gaussian sample from p_T(theta|x)
    eps_s=1e-2
    theta_t = torch.randn((nsamples, dim_theta)).to(device)  # (nsamples, 2)
    time_pts = eps_s+(1-eps_s)*torch.linspace(1, 0, 1000).to(device)  # (ntime_pts,)
    theta_list = [theta_t]
    
    for i in tqdm(range(len(time_pts) - 1)):
        # compute current time step t_i
        t = time_pts[i]
        dt = time_pts[i + 1] - t

        # calculate the drift and diffusion terms at t_i
        f = -0.5 * beta(t) * theta_t
        g = beta(t) ** 0.5
        # estimated score at t_i
        score = score_fn(theta=theta_t, t=t)
        score = score.detach()
        # evaluate dx/dt at t_i
        d_i = f - 0.5 * g * g * score 
        # euler step from t_i to t_i+1
        tmp_theta_tp1 = theta_t.detach() + dt * d_i 
        if i < len(time_pts) - 2:
            # corretion step
            # take next time step t_i+1
            t = time_pts[i+1]
            #compute drift and diffusion at t_i+1 and tmp_x_i+1
            f = -0.5 * beta(t) * tmp_theta_tp1
            g = beta(t) ** 0.5
            score = score_fn(theta=tmp_theta_tp1, t=t).detach()
            # evaluate dx/dt at t_i+1
            d_i_prime = f - 0.5 * g * g * score
            theta_t = theta_t.detach() + dt*(d_i/2 + d_i_prime/2)
        if theta_clipping_range[0] is not None:
            theta_t = theta_t.clip(*theta_clipping_range)
        theta_list.append(theta_t.detach().cpu())
        
    theta_list[0] = theta_list[0].detach().cpu()
    
    return theta_t, theta_list

def ablation_sampler(
    net, nsamples, dim_theta, x, num_steps=18, sigma_min=None, sigma_max=None,
    solver='heun', discretization='vp', schedule='vp', scaling='vp',
    epsilon_s=1e-3
):
    """
    Deterministic sampling (reverse ODE) with a 2nd order scheme using 
    the learnt denoiser rather than the score
    """

    # Helper functions for VP & VE noise level schedules.
    vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    vp_sigma_deriv = lambda beta_d, beta_min: lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
    vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d

    # Select default noise level range based on the specified time step discretization.
    if sigma_min is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=epsilon_s) #associated with time t=eps_s
        sigma_min = vp_def
    if sigma_max is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=1) #associated with time t=1
        sigma_max = vp_def
    
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    print("hey", sigma_min, sigma_max)
    # Compute corresponding betas for VP.
    vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s - np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
    vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d
    print(vp_beta_d,vp_beta_min)
    # Define time steps in terms of noise level.
    step_indices = torch.arange(num_steps, dtype=torch.float64)#uniform steps indices between 0, N-1
    if discretization == 'vp':
        orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1) #t is between eps_s and 1 (in the sampler)
        print(orig_t_steps)
        sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
   
    # Define noise level schedule.
    if schedule == 'vp':
        sigma = vp_sigma(vp_beta_d, vp_beta_min)
        sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
        sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)

    # Define scaling schedule.
    if scaling == 'vp':
        s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)

    # Compute final time steps based on the corresponding noise levels.
    t_steps = sigma_inv(net.round_sigma(sigma_steps))
    print("tsteps",t_steps)
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0
    # Main sampling loop.
    t_next = t_steps[0]
    x_next = torch.randn((nsamples,dim_theta)).to(torch.float64) * (sigma(t_next) * s(t_next))
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next
        
        # Euler step.
        h = t_next - t_cur
        denoised = net(x_cur / s(t_cur), x, sigma(t_cur)).to(torch.float64)
        d_cur = (sigma_deriv(t_cur) / sigma(t_cur) + s_deriv(t_cur) / s(t_cur)) * x_cur - sigma_deriv(t_cur) * s(t_cur) / sigma(t_cur) * denoised
        x_prime = x_cur + h * d_cur
        t_prime = t_next

        # Apply 2nd order correction.
        if solver == 'euler' or i == num_steps - 1:
            x_next = x_prime
        else:
            assert solver == 'heun'
            denoised = net(x_prime / s(t_prime), x, sigma(t_prime)).to(torch.float64)
            d_prime = (sigma_deriv(t_prime) / sigma(t_prime) + s_deriv(t_prime) / s(t_prime)) * x_prime - sigma_deriv(t_prime) * s(t_prime) / sigma(t_prime) * denoised
            x_next = x_cur + h * 0.5 * (d_cur +  d_prime)

    return x_next

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from nse import NSE, NSELoss
    from sm_utils import train
    from tasks.toy_examples.data_generators import SBIGaussian2d

    torch.manual_seed(1)
    N_TRAIN = 10_000
    N_SAMPLES = 4096
    TYPE_COV_EST = "DDIM"

    # Task
    task = SBIGaussian2d(prior_type="gaussian")
    # Prior and Simulator
    prior = task.prior
    simulator = task.simulator

    # Observations
    theta_true = torch.FloatTensor([-5, 150])  # true parameters, size (dim theta)
    x_obs_100 = torch.cat( #100 extra obs, size(nextra, dim x)
        [simulator(theta_true).reshape(1, -1) for _ in range(100)], dim=0
    )

    # Train data
    theta_train = task.prior.sample((N_TRAIN,)) #size(NTRAIN, dim theta)
    x_train = simulator(theta_train)# size(NTRAIN, dim x)
    
    # normalize theta
    theta_train_ = (theta_train - theta_train.mean(axis=0)) / theta_train.std(axis=0)

    # normalize x
    x_train_ = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0)
    x_obs_100_ = (x_obs_100 - x_train.mean(axis=0)) / x_train.std(axis=0)

    # # train score network
    dataset = torch.utils.data.TensorDataset(theta_train_.cuda(), x_train_.cuda())
    score_net = NSE(theta_dim=2, x_dim=2, hidden_features=[128, 256, 128]).cuda() #no embedding net, simple MLP with 3 layers
    avg_score_net = train(
        model=score_net,
        dataset=dataset,
        loss_fn=NSELoss(score_net),
        n_epochs=200,
        lr=1e-3,
        batch_size=256,
        #prior_score=False, # learn the prior score via the classifier-free guidance approach
    )
    score_net = avg_score_net.module #copy the trained module of the averaged model
    #torch.save(score_net, "score_net.pkl")

    # load score network
    # score_net = torch.load("score_net.pkl").cuda()

    # normalize prior
    #loc is almost 0
    loc_ = (prior.prior.loc - theta_train.mean(axis=0)) / theta_train.std(axis=0)
    #cov is almost identity
    cov_ = (
        torch.diag(1 / theta_train.std(axis=0))
        @ prior.prior.covariance_matrix
        @ torch.diag(1 / theta_train.std(axis=0))
    )
    prior_ = torch.distributions.MultivariateNormal(
        loc=loc_.cuda(), covariance_matrix=cov_.cuda()
    )
    prior_score_fn_ = get_vpdiff_gaussian_score(
        prior_.loc.cuda(), prior_.covariance_matrix.cuda(), score_net
    )

    for N_OBS in [2, 30, 40, 50, 60, 70, 80, 90]:
        t = torch.linspace(0, 1, 1000) #time discretization

        true_posterior = task.true_tall_posterior(x_obs_100[:N_OBS])

        # Normalize posterior
        loc_ = (true_posterior.loc - theta_train.mean(axis=0)) / theta_train.std(axis=0)
        cov_ = (
            torch.diag(1 / theta_train.std(axis=0))
            @ true_posterior.covariance_matrix
            @ torch.diag(1 / theta_train.std(axis=0))
        )
        true_posterior_ = torch.distributions.MultivariateNormal(
            loc=loc_.cuda(), covariance_matrix=cov_.cuda()
        )

        score_fn_true_posterior_ = get_vpdiff_gaussian_score(
            true_posterior_.loc, true_posterior_.covariance_matrix, score_net
        )

        # score_fn_true_posterior_ = get_vpdiff_gaussian_score(posterior_mean_0_, posterior_cov_0_, score_net)
        posterior_cov_0 = true_posterior.covariance_matrix.cuda()
        posterior_mean_0 = true_posterior.loc.cuda()
        posterior_cov_diffused = (
            posterior_cov_0[None] * score_net.alpha(t)[:, None, None].cuda()
            + (1 - score_net.alpha(t).cuda())[:, None, None]
            * torch.eye(posterior_cov_0.shape[0])[None].cuda()
        )
        posterior_mean_diffused = (
            posterior_mean_0[None] * score_net.alpha(t)[:, None].cuda()
        )

        posterior_cov_diffused_ = (
            cov_[None].cuda() * score_net.alpha(t)[:, None, None].cuda()
            + (1 - score_net.alpha(t).cuda())[:, None, None]
            * torch.eye(cov_.shape[0])[None].cuda()
        )
        posterior_mean_diffused_ = (
            loc_[None].cuda() * score_net.alpha(t)[:, None].cuda()
        )

        cov_est = vmap(
            lambda x: score_net.ddim(shape=(1000,), x=x, steps=100, eta=0.5),
            randomness="different",
        )(x_obs_100_[:N_OBS].cuda())
        cov_est = vmap(lambda x: torch.cov(x.mT))(cov_est)
        score_fn_gauss = partial(
            diffused_tall_posterior_score,
            prior=prior_,  # normalized prior# analytical posterior
            prior_score_fn=prior_score_fn_,
            x_obs=x_obs_100_[:N_OBS].cuda(),  # observations
            nse=score_net,  # trained score network
            dist_cov_est=cov_est,
            cov_mode="GAUSS",
        )

        score_fn_jac = partial(
            diffused_tall_posterior_score,
            prior=prior_,  # normalized prior# analytical posterior
            prior_score_fn=prior_score_fn_,
            x_obs=x_obs_100_[:N_OBS].cuda(),  # observations
            nse=score_net,  # trained score network
            cov_mode="JAC",
        )
        mse_scores = {"GAUSS": [], "JAC": []}
        for mu_, cov_, t in zip(
            posterior_mean_diffused_,
            posterior_cov_diffused_,
            torch.linspace(0, 1, 1000),
        ):
            dist = torch.distributions.MultivariateNormal(
                loc=mu_, covariance_matrix=cov_
            )
            ref_samples = dist.sample((1000,))
            ref_samples.requires_grad_(True)
            dist.log_prob(ref_samples).sum().backward()
            real_score = ref_samples.grad.clone()
            ref_samples.grad = None
            ref_samples.requires_grad_(False)

            approx_score_gauss = score_fn_gauss(ref_samples.cuda(), t.cuda())
            approx_score_jac = score_fn_jac(ref_samples.cuda(), t.cuda())
            error_gauss = (
                torch.linalg.norm(approx_score_gauss - real_score, axis=-1) ** 2
            )
            error_jac = torch.linalg.norm(approx_score_jac - real_score, axis=-1) ** 2
            error_mean = (
                torch.linalg.norm(real_score - real_score.mean(dim=0)[None], dim=-1)
                ** 2
            )
            r2_score_gauss = 1 - (error_gauss.sum() / error_mean.sum())
            r2_score_jac = 1 - (error_jac.sum() / error_mean.sum())
            mse_scores["GAUSS"].append(r2_score_gauss.item())
            mse_scores["JAC"].append(r2_score_jac.item())
        fig = plt.figure(figsize=(5, 5))
        plt.plot(
            torch.linspace(0, 1, 1000), mse_scores["GAUSS"], label="GAUSS", alpha=0.8
        )
        plt.plot(torch.linspace(0, 1, 1000), mse_scores["JAC"], label="JAC", alpha=0.8)
        plt.suptitle(N_OBS)
        plt.legend()
        plt.ylim(-1.1, 1.1)
        # plt.yscale('log')
        plt.ylabel("R2 score")
        plt.xlabel("Diffusion time")
        # plt.show()
        plt.savefig(f"results/gaussian/r2_score_{TYPE_COV_EST}_n_obs_{N_OBS}.png")
        plt.clf()
        # compute results for learned and analytical score during sampling
        # where each euler step is updated with the learned tall posterior score
        samples_per_alg = {}
        for name, score_fun in [
            ("NN / Gaussian cov", score_fn_gauss),
            ("JAC", score_fn_jac),
            ("Analytical", score_fn_true_posterior_),
        ]:
            samples_, _ = euler_sde_sampler(
                score_fun,
                N_SAMPLES,
                dim_theta=2,
                beta=score_net.beta,
                device="cuda:0",
                # theta_clipping_range=(-3, 3),
            )
            samples_per_alg[name] = (
                samples_.detach().cpu() * theta_train.std(axis=0)[None, None]
                + theta_train.mean(axis=0)[None, None]
            )

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"N={N_OBS}")
        ref_samples = true_posterior.sample((500,)).cpu()
        ind_samples = torch.randint(low=0, high=N_SAMPLES, size=(1000,))
        for ax, (name, all_thetas) in zip(axes.flatten(), samples_per_alg.items()):
            ax.scatter(*ref_samples.T, label="Ground truth")
            ax.scatter(*all_thetas[-1, ind_samples].T, label=name, alpha=0.8)
            # ax.set_ylim(140, 160)
            # ax.set_xlim(0, -10)
            ax.set_title(name)
            leg = ax.legend()
            for lh in leg.legendHandles:
                lh.set_alpha(1)
        plt.savefig(f"results/gaussian/samples_{TYPE_COV_EST}_n_obs_{N_OBS}.png")
        plt.clf()
