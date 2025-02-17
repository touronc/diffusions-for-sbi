# Script for the Guassian toy example. Reproduces experiemnts from section 4.1 of the paper.

import os
import sys
import time

import torch
from torch.func import vmap
from tqdm import tqdm

from embedding_nets import EpsilonNet, FakeFNet
from nse_toy import NSE
from tall_posterior_sampler import mean_backward, prec_matrix_backward
from tasks.toy_examples.data_generators import Gaussian_Gaussian_mD
from vp_diffused_priors import get_vpdiff_gaussian_score

if __name__ == "__main__":
    path_to_save = sys.argv[1]
    os.makedirs(path_to_save, exist_ok=True)

    torch.manual_seed(1)

    all_exps = []
    for DIM in [2, 4, 8, 10, 16, 32, 64]:
        for eps in [0, 1e-3, 1e-2, 1e-1]:
            for seed in tqdm(range(5), desc=f"Dim {DIM} eps {eps}"):
                # Simulator and observations
                torch.manual_seed(seed)
                means = torch.rand(DIM) * 20 - 10  # between -10 and 10
                stds = torch.rand(DIM) * 25 + 0.1  # between 0.1 and 25.1
                task = Gaussian_Gaussian_mD(dim=DIM, means=means, stds=stds)
                prior = task.prior #normal(means,diag(stds))
                simulator = task.simulator #normal(theta,rho Id)
                theta_true = prior.sample(sample_shape=(1,))  # true parameters
                #size (1,DIM)
                x_obs_100 = torch.cat(
                    [simulator(theta_true).reshape(1, -1) for _ in range(100)], dim=0
                ) #size (100,DIM)

                # True posterior score / epsilon network
                score_net = NSE(theta_dim=DIM, x_dim=DIM, net_type="fnet")
                
                beta_min = 0.1
                beta_max = 40
                beta_d = beta_max - beta_min

                def alpha(t): #mean of the forward transition kernel qt|0
                    log_alpha = 0.5 * beta_d * (t**2) + beta_min * t
                    return torch.exp(-0.5 * log_alpha)

                score_net.alpha = alpha

                idm = torch.eye(DIM).cuda()
                inv_lik = torch.linalg.inv(task.simulator_cov).cuda() #likelihood cov, big sigma in the paper
                inv_prior = torch.linalg.inv(prior.covariance_matrix).cuda()#prior cov, big sigma_lambda
                posterior_cov = torch.linalg.inv(inv_lik + inv_prior) #true cov of the individual posterior dist

                inv_prior_prior = inv_prior @ prior.loc.cuda() #prior_cov-1 @ prior mean

                def real_eps(theta, x, t):
                    posterior_cov_diff = ( #cov of the intermediate distr p_t(theta|x)
                        posterior_cov[None] * score_net.alpha(t)[..., None]
                        + (1 - score_net.alpha(t))[..., None] * idm[None]
                    )
                    posterior_mean_0 = ( #mean of the individual posteriors of the true data (t=0), mu_post(x star)
                        posterior_cov @ (inv_prior_prior[:, None] + inv_lik @ x.mT)
                    ).mT
                    posterior_mean_diff = (score_net.alpha(t) ** 0.5) * posterior_mean_0 #mean of the intermediate distr p_t
                    score = -(
                        torch.linalg.inv(posterior_cov_diff)
                        @ (theta - posterior_mean_diff)[..., None]
                    )[..., 0] #true score analytical
                    return -score_net.sigma(t) * score #true noise epsilon eq (26)

                # Perturbed score network
                score_net.net = FakeFNet(
                    real_eps_fun=real_eps, eps_net=EpsilonNet(DIM), eps_net_max=eps
                )
                score_net.cuda()

                # Prior score function
                t = torch.linspace(0, 1, 1000)
                prior_score_fn = get_vpdiff_gaussian_score(
                    prior.loc.cuda(), prior.covariance_matrix.cuda(), score_net
                ) #function to compute true prior scores at time t
                # when prior is gaussian

                def prior_score(theta, t):
                    mean_prior_0_t = mean_backward(theta, t, prior_score_fn, score_net)
                    prec_prior_0_t = prec_matrix_backward(
                        t, prior.covariance_matrix.cuda(), score_net
                    ).repeat(theta.shape[0], 1, 1)
                    prior_score = prior_score_fn(theta, t)
                    return (
                        prior_score,
                        mean_prior_0_t,
                        prec_prior_0_t,
                    )

                # Sampling
                for N_OBS in [2, 4, 8, 16, 32, 64, 90]:
                    true_posterior_cov = torch.linalg.inv(inv_lik * N_OBS + inv_prior) #true posterior cov for n obs
                    true_posterior_mean = true_posterior_cov @ ( #true posterior mean with n obs
                        inv_prior_prior + inv_lik @ x_obs_100[:N_OBS].sum(dim=0).cuda()
                    )

                   # True posterior samples
                    ref_samples = (
                        torch.distributions.MultivariateNormal(
                            loc=true_posterior_mean,
                            covariance_matrix=true_posterior_cov,
                        )
                        .sample((1000,))
                        .cpu()
                    )
                    print("ref samples", ref_samples.size())

                    infos = {
                        "true_posterior_mean": true_posterior_mean,
                        "true_posterior_cov": true_posterior_cov,
                        "true_theta": theta_true,
                        "N_OBS": N_OBS,
                        "seed": seed,
                        "dim": DIM,
                        "eps": eps,
                        "exps": {"Langevin": [], "GAUSS": [], "JAC": []},
                    }

                    # Approximate posterior samples
                    for sampling_steps, eta in zip( #eta is for the std of DDIM reverse process
                        [50, 150, 400, 1000], [0.2, 0.5, 0.8, 1]
                    ):
                        tstart_gauss = time.time()
                        # Estimate Gaussian covariance
                        samples_ddim = (
                            score_net.ddim(
                                shape=(1000 * N_OBS,), #return 1000 samples per extra obs
                                x=x_obs_100[None, :N_OBS]
                                .repeat(1000, 1, 1)
                                .reshape(1000 * N_OBS, -1)
                                .cuda(),
                                steps=100,
                                eta=0.5,
                            )
                            .detach()
                            .reshape(1000, N_OBS, -1) #reshape such that there 1000 samples per x_j obs
                            .cpu()
                        ) #sample with ddim from the true tall posterior
                        
                        cov_est = vmap(lambda x: torch.cov(x.mT))(
                            samples_ddim.permute(1, 0, 2)
                        )#compute an estimate of the cov for each indiv true posterior p(theta|x_j)
                       
                        # Sample with GAUSS
                        samples_gauss = score_net.ddim(
                            shape=(1000,), #we want 1000 samples of tall posterior
                            x=x_obs_100[:N_OBS].cuda(), #cond variable
                            eta=eta, #for the std of reverse process
                            steps=sampling_steps,
                            prior_score_fn=prior_score_fn,
                            prior=prior,
                            dist_cov_est=cov_est.cuda(), #used in score function with GAUSS algo
                            cov_mode="GAUSS",
                        ).cpu()

                        tstart_jac = time.time()
                       # Sample with JAC
                        samples_jac = score_net.ddim(
                            shape=(1000,),
                            x=x_obs_100[:N_OBS].cuda(),
                            eta=eta,
                            steps=sampling_steps,
                            prior_score_fn=prior_score,
                            # prior=prior,
                            cov_mode="JAC",
                        ).cpu()

                        tstart_lang = time.time()
                    #     #Sample with Langevin
                    #     with torch.no_grad():
                    #         lang_samples = score_net.annealed_langevin_geffner(
                    #             shape=(1000,),
                    #             x=x_obs_100[:N_OBS].cuda(),
                    #             prior_score_fn=prior_score_fn,
                    #             lsteps=5,
                    #             steps=sampling_steps,
                    #         )
                    #     t_end_lang = time.time()
                        dt_gauss = tstart_jac - tstart_gauss
                        dt_jac = tstart_lang - tstart_jac
                    #     dt_lang = t_end_lang - tstart_lang

                    #     infos["exps"]["Langevin"].append(
                    #         {
                    #             "dt": dt_lang,
                    #             "samples": lang_samples,
                    #             "n_steps": sampling_steps,
                    #         }
                    #     )
                        infos["exps"]["GAUSS"].append(
                            {
                                "dt": dt_gauss,
                                "samples": samples_gauss,
                                "n_steps": sampling_steps,
                            }
                        )
                        infos["exps"]["JAC"].append(
                            {
                                "dt": dt_jac,
                                "samples": samples_jac,
                                "n_steps": sampling_steps,
                            }
                        )
                        if "DDIM" not in infos:
                            infos["DDIM"] = {
                                "samples": samples_ddim.cpu(),
                                "steps": 100,
                                "eta": 0.5,
                            }
                    all_exps.append(infos)
                    print(N_OBS, eps)
                    torch.save(all_exps, os.path.join(path_to_save, "gaussian_exp.pt"))

    # file=torch.load("/home/ctouron/codedev/diffusions-for-sbi/results/gaussian_exp.pt")
    # print(file[27]["DDIM"]["samples"].size())
    # print(file[27].keys())
    # print("dim", file[27]["dim"])
    # print("obs",file[27]["N_OBS"])
    # print(file[27]["exps"]["GAUSS"][0]["samples"].size())


    