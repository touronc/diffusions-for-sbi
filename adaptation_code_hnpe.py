import torch
import numpy as np
import numpyro
from sbi.utils import BoxUniform
import seaborn as sns
import matplotlib.pyplot as plt
from tasks.toy_examples import simulator
from tasks.toy_examples import c2st

import os
#os.environ["JAX_PLATFORMS"] = "cpu"

from tasks.toy_examples import get_true_samples_nextra_obs as true
from torch.func import vmap
from functools import partial

from nse import NSE, NSELoss
from sm_utils import train_with_validation
from tall_posterior_sampler import diffused_tall_posterior_score, euler_sde_sampler, tweedies_approximation, heun_ode_sampler
from vp_diffused_priors import get_vpdiff_gaussian_score, get_vpdiff_uniform_score
#from sbibm_posterior_estimation import run_train_sgm, run_sample_sgm

numpyro.set_host_device_count(4)
torch.manual_seed(42)
n_extra_obs=20
noise=0.001
total_budget_train=2000
num_train = total_budget_train
n_epochs=2000
batch_size=500
n_samples=1000
training = False
sampling = not training
langevin = False
corrector = False
ddim_jac = True
ddim_gauss = True
euler = True
if euler:
    method="euler"
elif langevin:
    method="langevin"
elif corrector:
    method="corrector"
elif ddim_gauss:
    method="ddim_gauss"
elif ddim_jac:
    method="ddim_jac"
if (ddim_gauss and ddim_jac and euler) or (ddim_jac and euler):
    method="all"

prior_beta_low=torch.tensor([0.0])
prior_beta_high=torch.tensor([1.0])
prior=simulator.prior_ToyModel(low=torch.tensor([0.0, 0.0]),
                    high=torch.tensor([1.0, 1.0]))
prior_beta = BoxUniform(low=prior_beta_low, high=prior_beta_high)
theta_true = torch.tensor([0.5,0.5]).unsqueeze(0)

x_obs = simulator.simulator_ToyModel(theta_true,n_extra_obs,sigma=noise,p_alpha=prior).squeeze(1)
print(x_obs)
print("size true theta",theta_true.size())
print("size true obs",x_obs.size())

# compute samples from the true tall posterior
#_,true_samples=true.get_posterior_samples(n_extra_obs,theta_true.squeeze(0),x_obs,n_samples)
#create training datset
theta_train=prior.sample((num_train,)) # get different beta, get pairs (alpha,beta)
#beta_train=theta_train[:,1].repeat_interleave(n_extra_obs+1).unsqueeze(1) #beta is the second parameter, repeat it for different obs x
beta_train=theta_train[:,1].unsqueeze(1) #beta is the second parameter, repeat it for different obs x

#x_train=simulator.simulator_ToyModel(theta=theta_train, n_extra=n_extra_obs, p_alpha=prior).squeeze(1).reshape((-1,x_obs.size(0)))
x_train=simulator.simulator_ToyModel(theta=theta_train, sigma=noise,p_alpha=prior).squeeze(1)

print("Training data:", beta_train.shape, x_train.shape)
score_network = NSE(theta_dim=beta_train.size(1), 
                    x_dim=x_train.size(1),
                    hidden_features=[64,128,128,64],
                    #activation = torch.nn.SiLU,
                    freqs=1)#, net_type="fnet")

beta_min = 0.1
beta_max = 40
beta_d = beta_max - beta_min

def beta(t):
    return beta_min + beta_d*t

def alpha(t): #mean of the forward transition kernel q_t|0
    log_alpha = 0.5 * beta_d * (t**2) + beta_min * t
    return torch.exp(- log_alpha)

def sigma(t):
    return torch.sqrt(1-score_network.alpha(t))#+1e-5

score_network.beta = beta
score_network.alpha = alpha
score_network.sigma = sigma

#to normalize the data before training

beta_train_mean, beta_train_std = beta_train.mean(dim=0), beta_train.std(dim=0)
x_train_mean, x_train_std = x_train.mean(dim=0), x_train.std(dim=0)
# normalize theta
beta_train_norm = (beta_train - beta_train_mean) / beta_train_std
# normalize x
x_train_norm = (x_train - x_train_mean) / x_train_std
# dataset for dataloader
data_train = torch.utils.data.TensorDataset(beta_train_norm, x_train_norm)

if beta_train.shape[0] > 10000:
    # min_nb_epochs = n_epochs * 0.8 # 4000
    min_nb_epochs = 2000
else:
    min_nb_epochs = 100
if training :
# Train Score Network
    avg_score_net, train_losses, val_losses, best_epoch = train_with_validation(
        score_network,
        dataset=data_train,
        loss_fn=NSELoss(score_network),
        n_epochs=n_epochs,
        lr=1e-4,
        batch_size=batch_size,
        validation_split=0.2,
        early_stopping=True,
        min_nb_epochs=min_nb_epochs,
    )
    score_network = avg_score_net.module
    torch.save(score_network.state_dict(), "trained_network.pkl")

    # plot training and validation losses
    plt.figure()
    plt.plot(torch.arange(n_epochs),train_losses, label="training loss")
    plt.plot(torch.arange(n_epochs),val_losses, label="validation loss")
    plt.legend()
    plt.show()
    print("End of training")

score_network.tweedies_approximator = tweedies_approximation

if sampling :
    file = "trained_network_64_128_128_64_freqs_1.pkl"
    score_network.load_state_dict(torch.load(file, weights_only=True))
    score_network.eval()
    # normalize the true observations
    x_obs_norm = (x_obs - x_train_mean) / x_train_std
    # normalize the prior function before computing its score
    low_norm = (prior_beta_low - beta_train_mean) / beta_train_std * 2
    high_norm = (prior_beta_high - beta_train_mean) / beta_train_std * 2
    prior_norm = torch.distributions.Uniform(low_norm, high_norm)
    # compute the prior score
    prior_score_fn = get_vpdiff_uniform_score(low_norm, high_norm, score_network)


    # Sample with annealed langevin from Geffner
    #attention il faut reshape l'obs x_0
    if langevin :
        samples_langevin_geffner = score_network.annealed_langevin_geffner(
                        shape=(n_samples,),
                        x=x_obs_norm.reshape(n_extra_obs+1,-1),
                        prior_score_fn=prior_score_fn,
                        clf_free_guidance=False,
                        steps=400,
                        lsteps=5,
                        tau=0.5,
                        theta_clipping_range=(prior_beta_low,prior_beta_high),
                        verbose=True,
                    )
        # renormalize the samples with the previous means and stds
        samples = samples_langevin_geffner.detach() * beta_train_std + beta_train_mean
        print(samples_langevin_geffner.size())

    # sample with predictor corrector algo
    elif corrector:
        samples_pred_correct = score_network.predictor_corrector(
                    shape=(n_samples,),
                    x=x_obs_norm.reshape(n_extra_obs+1,-1),
                    steps=400,
                    prior_score_fun=prior_score_fn,
                    lsteps=5,
                    r=0.5,
                    predictor_type="id",
                    verbose=True,
                    theta_clipping_range=(prior_beta_low,prior_beta_high),
                )
        samples = samples_pred_correct.detach() * beta_train_std + beta_train_mean
        print(samples_pred_correct.size())

    # sample with ddim and jac mode to compute the backward transition kernel
    if ddim_jac:
        print("########### DDIM JAC ###############")
        steps = 1000
        samples_ddim_jac = score_network.ddim(
                    shape=(n_samples,),
                    x=x_obs_norm.reshape(n_extra_obs+1,-1),
                    eta=1
                    if steps == 1000
                    else 0.8
                    if steps == 400
                    else 0.5,  # corresponds to the equivalent time setting from section 4.1
                    steps=steps,
                    theta_clipping_range=(prior_beta_low,prior_beta_high),
                    prior=prior_norm,
                    prior_type="uniform",
                    prior_score_fn=prior_score_fn,
                    clf_free_guidance=False,
                    dist_cov_est=None, #if GAUSS, use estimated cov at time 0 to compute Sigma_t,j^-1
                    dist_cov_est_prior=None,
                    cov_mode="JAC",
                    verbose=True,
                )
        print("avant",samples_ddim_jac[:2,:])

        samples = samples_ddim_jac.detach() * beta_train_std + beta_train_mean
        samples_ddim_jac = samples_ddim_jac.detach() * beta_train_std + beta_train_mean
        print("après",samples_ddim_jac[:2,:])
        
        print(samples_ddim_jac.size())

    if ddim_gauss:
        print("########### DDIM GAUSS ###############")

        cov_est = vmap(
                    lambda x: score_network.ddim(
                        shape=(n_samples,), x=x, steps=100, eta=0.5
                    ),randomness="different",)(x_obs_norm.reshape((n_extra_obs+1,-1)))
        # size (n_obs, n_samples, dim theta)
        # .mT transpose the last 2 dims
        cov_est = vmap(lambda x: torch.cov(x.mT).reshape((1,1)))(cov_est)
        #size (n_extra+1,dim theta x dim theta)
        steps = 400
        samples_ddim_gauss = score_network.ddim(
                    shape=(n_samples,),
                    x=x_obs_norm.reshape((n_extra_obs+1,-1)),
                    eta=1
                    if steps == 1000
                    else 0.8
                    if steps == 400
                    else 0.5,  # corresponds to the equivalent time setting from section 4.1
                    steps=steps,
                    theta_clipping_range=(prior_beta_low,prior_beta_high),
                    prior=prior_norm,
                    prior_type="uniform",
                    prior_score_fn=prior_score_fn,
                    clf_free_guidance=False,
                    dist_cov_est=cov_est, #if GAUSS, use estimated cov at time 0 to compute Sigma_t,j^-1
                    dist_cov_est_prior=None,
                    cov_mode="GAUSS",
                    verbose=True,
                )
        samples = samples_ddim_gauss.detach() * beta_train_std + beta_train_mean
        samples_ddim_gauss = samples_ddim_gauss.detach() * beta_train_std + beta_train_mean

    if euler:
        print("########### EULER ###############")
        cov_est = vmap(
                    lambda x: score_network.ddim(
                        shape=(n_samples,), x=x, steps=100, eta=0.5
                    ),randomness="different",)(x_obs_norm.reshape((n_extra_obs+1,-1)))
        # size (n_obs, n_samples, dim theta)
        # .mT transpose the last 2 dims
        cov_est = vmap(lambda x: torch.cov(x.mT).reshape((1,1)))(cov_est)
        #size (n_extra+1,dim theta x dim theta)
        score_fn = partial(
                    diffused_tall_posterior_score,
                    prior=prior_norm,  # normalized prior
                    prior_type="uniform",
                    prior_score_fn=prior_score_fn,  # analytical prior score function
                    x_obs=x_obs_norm.reshape((n_extra_obs+1,-1)),  # observations
                    nse=score_network,  # trained score network
                    dist_cov_est=cov_est,
                    cov_mode="GAUSS",
                )

        # sample from tall posterior with reverse sde
        samples_euler,list_samples = euler_sde_sampler( #sample with reverse SDE and the score at all times t
            score_fn,
            n_samples,
            dim_theta=beta_train_mean.shape[-1],
            beta=score_network.beta,
            debug=False,
            theta_clipping_range=(prior_beta_low,prior_beta_high),
        )
        samples = samples_euler.detach() * beta_train_std + beta_train_mean
        samples_euler = samples_euler.detach() * beta_train_std + beta_train_mean
        
        # samples_heun,_ = heun_ode_sampler( #sample with probability ODE and the score at all times t
        #     score_fn,
        #     n_samples,
        #     dim_theta=beta_train_mean.shape[-1],
        #     beta=score_network.beta,
        #     theta_clipping_range=(prior_beta_low,prior_beta_high),
        # )
        # samples_heun = samples_heun.detach() * beta_train_std + beta_train_mean
        # print(samples_heun[:5,:])
        print("euler",samples_euler.size())

    # plt.figure()
    # for j in range(len((list_samples))):
    #     if j%100==0:
    #         plt.subplot(5,2,j//100+1)
    #         sns.kdeplot(list_samples[j], label=f"{1-j/1000}")
    #         plt.legend()
    # plt.show()
    
    param, density_beta = true.true_marginal_beta_nextra_obs(x_obs)
    plt.figure()
    plt.subplot(121)
    #axes[1][1].plot(param, density_beta, color="red")
    sns.kdeplot(samples_euler.squeeze(1), label="euler", color="blue")
    plt.plot(param, density_beta, color="red", label="true")
    #sns.kdeplot(samples_heun.squeeze(1), label="heun", color="orange")
    #sns.kdeplot(true_samples[:,1], label="mcmc", color="black")
    plt.title(r"$p(\beta|x_0,...,x_n)$")
    plt.legend()

    plt.subplot(122)
    plt.title(r"$p(\beta|x_0,...,x_n)$")
    plt.plot(param, density_beta, color="red", label="true")
    sns.kdeplot(samples_ddim_gauss.squeeze(1), label="ddim gauss", color="green")
    sns.kdeplot(samples_ddim_jac.squeeze(1), label="ddim jac", color="grey")
    plt.legend()

    plt.savefig(file.split(".")[0]+f"_density_{method}_noise_{noise}_{n_extra_obs+1}_obs.pdf",format="pdf")
    plt.savefig(file.split(".")[0]+f"_density_{method}_noise_{noise}_{n_extra_obs+1}_obs.svg",format="svg")

    plt.show()
    # acc = c2st.c2st(samples_euler,torch.tensor(true_samples[:,1]).unsqueeze(1))
    # print("EULER accuracy : ", acc)
    # acc = c2st.c2st(samples_ddim_gauss,torch.tensor(true_samples[:,1]).unsqueeze(1))
    # print("DDIM GAUSS accuracy : ", acc)
    # acc = c2st.c2st(samples_ddim_jac,torch.tensor(true_samples[:,1]).unsqueeze(1))
    # print("DDIM JAC accuracy : ", acc)
if 0:
    beta_0 = prior_beta.sample((n_samples,))
    #samples=[beta_0]
    z_0 = torch.randn(beta_0.size())

    time = torch.linspace(1e-3,1.0,19)
    i=2
    plt.figure()
    plt.subplot(10,2,1)
    sns.kdeplot(beta_0.squeeze(1), label="0")
    sns.kdeplot(z_0.squeeze(1), label="ref", color="orange")

    plt.legend()
    for t in time :
        z = torch.randn(beta_0.size())
        alpha_t = score_network.alpha(t)
        sigma_t=score_network.sigma(t)
        beta_0 = alpha_t**0.5*beta_0+sigma_t*z
        #samples.append(beta_0)
        plt.subplot(10,2,i)
        sns.kdeplot(beta_0.squeeze(1), label=f"{round(t.item(),3)}")
        
        sns.kdeplot(z_0.squeeze(1), label="ref", color="orange")
        plt.legend()
        i+=1
    plt.subplots_adjust(hspace=0.5)
    # plt.savefig("evolution of diffusion process.svg", format="svg")
    # plt.savefig("evolution of diffusion process.pdf", format="pdf")
    plt.show()


