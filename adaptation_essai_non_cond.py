import torch
import numpy as np
from sbi.utils import BoxUniform
import seaborn as sns
import matplotlib.pyplot as plt
#from tasks.toy_examples import get_true_samples_nextra_obs as true
from torch.func import vmap
from functools import partial

from nse import NSE, NSELoss, ExplicitLoss,FNet
from sm_utils import train_with_validation
from tall_posterior_sampler import diffused_tall_posterior_score, euler_sde_sampler, tweedies_approximation, heun_ode_sampler
from vp_diffused_priors import get_vpdiff_gaussian_score
#from sbibm_posterior_estimation import run_train_sgm, run_sample_sgm

torch.manual_seed(10)
total_budget_train=5000
num_train = total_budget_train
n_epochs=3000
batch_size=500
n_samples=1000
type_net="noncond"
cov_mode="GAUSS"
training = False
sampling = not training
loss_type="denoising"

mu_prior = torch.ones(2)
cov_prior = torch.eye(2)*3
prior_beta = torch.distributions.MultivariateNormal(mu_prior, cov_prior)

def true_score(theta):
    "analytical score of the true individual posterior"
    return -torch.linalg.inv(cov_prior)@(theta.reshape(2,1)-mu_prior)

def true_diff_score(theta,t, score_net):
    "analytical score of the true diffused posterior p_t(beta|x)"
    alpha_t = score_net.alpha(t).item()
    return -torch.linalg.inv((1-alpha_t)*torch.eye(2)+alpha_t*cov_prior)@((theta-alpha_t**0.5*mu_prior).reshape(2,1))

def wasserstein_dist(mu_1,mu_2,cov_1,cov_2):
    """Compute the Wasserstein distance between 2 Gaussians"""
    # G.Peyré & M. Cuturi (2020), Computational Optimal Transport, eq 2.41 
    bures = ((cov_1.sqrt()-cov_2.sqrt())**2).sum()
    return ((mu_1-mu_2)**2).sum() + bures

#create training datset
beta_train=prior_beta.sample((num_train,)) # get different beta, get pairs (alpha,beta)
x_fake = beta_train
data_train = torch.utils.data.TensorDataset(beta_train,x_fake)
print("########################################")
print("Training data:", beta_train.shape)
print("########################################")

score_network = NSE(theta_dim=beta_train.size(1), 
                    x_dim=x_fake.size(1),
                    net_type=type_net,
                    freqs=1,
                    hidden_features=[32,32])#, net_type="fnet")
print(score_network)

#define losses (explicit form or denoising approximation)
if loss_type=="denoising":
    loss=NSELoss(score_network)

#define noise schedule
beta_min = 0.1
beta_max = 20
beta_d = beta_max - beta_min

def beta(t):
    return beta_min+beta_d*t
score_network.beta=beta

def alpha(t): #squared mean of the forward transition kernel q_t|0
    log_alpha = 0.5 * beta_d * (t**2) + beta_min * t
    return torch.exp(- log_alpha)

score_network.alpha = alpha

def sigma(t): #std of the transition kernel
    return torch.sqrt(1-score_network.alpha(t))#+1e-5# + beta_min)
    #return ((0.5 * beta_d * (t ** 2) + beta_min * t).exp() - 1+1e-5).sqrt()
    
score_network.sigma=sigma

def score_fnet(theta,x,t): #score estimated by the neural network
    sigma=score_network.sigma(t)
    if t.dim()>0:
        sigma=sigma.unsqueeze(1) 
    return -score_network(theta,x,t)/sigma

def analytical_score_gaussian(theta,x,t):# analytical score for individual posteriors
    if theta.ndim>1:
        theta_tmp=theta.reshape(theta.shape[1],theta.shape[0])
    else:
        theta_tmp = theta.unsqueeze(1) # add a dimension for computation
    alpha_t = alpha(t) #same size as t
    tmp_score= -torch.linalg.inv((1-alpha_t)*torch.eye(2)+alpha_t*cov_post)@(theta_tmp-alpha_t**0.5*mu_post(x))
    return tmp_score.reshape(theta.size())

def score_gaussian(theta,x,t):
    return score_network(theta,x,t)

if type_net=="fnet":
    score_network.score=score_fnet

if type_net=="analytical":
    score_network.score=analytical_score_gaussian

# if type_net=="gaussian":
#     score_network.score=score_gaussian

score_network.tweedies_approximator = tweedies_approximation

if beta_train.shape[0] > 10000:
    # min_nb_epochs = n_epochs * 0.8 # 4000
    min_nb_epochs = 2000
else:
    min_nb_epochs = 100
if training :
    avg_score_net, train_losses, val_losses, best_epoch = train_with_validation(
        score_network,
        dataset=data_train,
        loss_fn=loss,
        n_epochs=n_epochs,
        lr=1e-4,
        batch_size=batch_size,
        validation_split=0.2,
        early_stopping=True,
        min_nb_epochs=min_nb_epochs,
    )
    score_network = avg_score_net.module
    print("End of training")
    plt.figure()
    plt.plot(torch.arange(n_epochs),train_losses, label="training loss")
    plt.plot(torch.arange(n_epochs),val_losses, label="validation loss")
    plt.legend()
    plt.show()
    torch.save(score_network.state_dict(), "trained_network_gauss.pkl")

if sampling:
    file = "trained_network_gauss.pkl"
    score_network.load_state_dict(torch.load(file, weights_only=True))
    score_network.eval()
    
    # Sample from the true posterior
    tmp_beta= prior_beta.sample((1,))
    tmp_x=tmp_beta
    time = torch.linspace(1e-3,1.0,1000)
    true_score_noncond=true_diff_score(tmp_beta,time[0],score_network).reshape(1,2)
    est_score=score_network.score(theta=tmp_beta,x=tmp_x,t=time[0].item()*torch.ones(1)).detach()
    for i in range(1,time.size(0)):
        t= time[i].item()
        tmp_est=score_network.score(theta=tmp_beta,x=tmp_x,t=t*torch.ones(1)).detach()
        tmp_true=true_diff_score(tmp_beta,torch.ones(1)*t,score_network).reshape(1,2)
        true_score_noncond = torch.cat((true_score_noncond,tmp_true), dim=0)
        est_score = torch.cat((est_score,tmp_est), dim=0)

    print("#########################################")
    plt.figure(figsize=(15,15))
    plt.subplot(221)
    # plt.scatter(time,true_score[:,0], color="red", label="true")
    # plt.scatter(time,est_score[:,0], color="blue", label="estimated")
    plt.plot(time[10:],true_score_noncond[10:,0], color="red", label="true")
    plt.plot(time[10:],est_score[10:,0], color="blue", label="estimated")
    
    plt.legend()
    plt.title("Dimension 0")
    plt.xlabel("t")
    plt.ylabel(r"$\nabla_{\theta}\log p_t(\theta)$ and $s_{\phi}(\theta,t)$",fontsize=12)
    plt.subplot(222)
    # plt.scatter(time,true_score[:,1], color="red", label="true")
    # plt.scatter(time,est_score[:,1], color="blue", label="estimated")
    plt.plot(time[10:],est_score[10:,1], color="blue", label="estimated")
    plt.plot(time[10:],true_score_noncond[10:,1], color="red", label="true")
    
    plt.legend()
    plt.title("Dimension 1")
    plt.ylabel(r"$\nabla_{\theta}\log p_t(\theta)$ and $s_{\phi}(\theta,t)$",fontsize=12)
    plt.xlabel("t")
    plt.subplot(223)
    # plt.scatter(time,true_score[:,0], color="red", label="true")
    # plt.scatter(time,est_score[:,0], color="blue", label="estimated")
    plt.plot(time[:10],est_score[:10,0], color="blue", marker='o',label="estimated")
    plt.plot(time[:10],true_score_noncond[:10,0], color="red", marker='o',label="true")
    plt.legend()
    plt.title("Dimension 0 - near true score")
    plt.xlabel("t")
    plt.ylabel(r"$\nabla_{\theta}\log p_t(\theta)$ and $s_{\phi}(\theta,t)$",fontsize=12)
    

    plt.subplot(224)
    # plt.scatter(time,true_score[:,1], color="red", label="true")
    # plt.scatter(time,est_score[:,1], color="blue", label="estimated")
    plt.plot(time[:10],est_score[:10,1], color="blue", marker='o',label="estimated")
    plt.plot(time[:10],true_score_noncond[:10,1], color="red", marker='o',label="true")
    plt.legend()
    plt.title("Dimension 1 - near true score")
    plt.xlabel("t")
    plt.ylabel(r"$\nabla_{\theta}\log p_t(\theta)$ and $s_{\phi}(\theta,t)$",fontsize=12)

    plt.suptitle("Evolution of the scores with time")
    plt.savefig(f"1_obs_{cov_mode}_{type_net}_net_score.pdf", format="pdf")
    plt.show()

 
    # try_cov = cov_post.repeat(tmp_tall_x.shape[0],1,1)
    prior_score_fn = get_vpdiff_gaussian_score(mu_prior,cov_prior,score_network)
    score_sampling = partial(score_network.score,x=tmp_x)
    print("############### EULER ###############")
    # sampling with euler
    samples_euler,_ = euler_sde_sampler( #sample with reverse SDE and the score at all times t
            score_fn=score_sampling,
            nsamples=n_samples,
            dim_theta=tmp_beta.shape[-1],
            beta=score_network.beta,
            debug=False,
            #theta_clipping_range=(-10,10),
        )
    samples_euler = samples_euler.detach()
    print(samples_euler[:5,:])

    print("############### HEUN ###############")

    samples_heun,_ = heun_ode_sampler( #sample with reverse SDE and the score at all times t
            score_sampling,
            n_samples,
            dim_theta=tmp_beta.shape[-1],
            beta=score_network.beta,
            #debug=False,
            #theta_clipping_range=(-10,10),
        )
    samples_heun = samples_heun.detach()
    print(samples_heun[:5,:])
    
    print("############### DDIM GAUSS ###############")
    print(tmp_x.size())
    steps=400
    samples_ddim_gauss = score_network.ddim(
                    shape=(n_samples,),
                    x=tmp_x,
                    eta=1
                    if steps == 1000
                    else 0.8
                    if steps == 400
                    else 0.5,  # corresponds to the equivalent time setting from section 4.1
                    steps=steps,
                    #theta_clipping_range=(-10,10),
                    prior=prior_beta,
                    prior_type="gaussian",
                    prior_score_fn=prior_score_fn,
                    clf_free_guidance=False,
                    dist_cov_est=None, #if GAUSS, use estimated cov at time 0 to compute Sigma_t,j^-1
                    dist_cov_est_prior=None,
                    cov_mode="GAUSS",
                    verbose=True,
                )
    samples_ddim_gauss = samples_ddim_gauss.detach()
    print(samples_ddim_gauss[:5,:])

    print("############### GEFFNER ###############")
    
    samples_langevin_geffner = score_network.annealed_langevin_geffner(
                        shape=(n_samples,),
                        x=tmp_x,
                        prior_score_fn=prior_score_fn,
                        clf_free_guidance=False,
                        steps=400,
                        lsteps=5,
                        tau=0.5,
                        #theta_clipping_range=(-10,10),
                        verbose=True,
                    )
    samples_langevin_geffner = samples_langevin_geffner.detach()
    
    true_samples=prior_beta.sample((n_samples,)) 
    
    plt.figure(figsize=(10,10))
    plt.subplot(4,2,1)
    sns.kdeplot(samples_euler[:,0], color="blue", label="euler")
    sns.kdeplot(true_samples[:,0], color="red", label="true")
    plt.legend()
    plt.title("Dimension 0")
    plt.ylabel(r"$p(\theta)$")

    plt.subplot(4,2,2)
    sns.kdeplot(samples_euler[:,1], color="blue", label="euler")
    sns.kdeplot(true_samples[:,1], color="red", label="true")
    plt.legend()
    plt.title("Dimension 1")
    plt.ylabel(r"$p(\theta)$")
    
    plt.subplot(4,2,3)
    sns.kdeplot(samples_ddim_gauss[:,0], color="green", label="ddim")
    sns.kdeplot(true_samples[:,0], color="red", label="true")
    plt.legend()
    plt.ylabel(r"$p(\theta)$")
    
    plt.subplot(4,2,4)
    sns.kdeplot(samples_ddim_gauss[:,1], color="green", label="ddim")
    sns.kdeplot(true_samples[:,1], color="red", label="true")
    plt.legend()
    plt.ylabel(r"$p(\theta)$")
    

    plt.subplot(4,2,5)
    sns.kdeplot(samples_langevin_geffner[:,0], color="grey", label="geffner")
    sns.kdeplot(true_samples[:,0], color="red", label="true")
    plt.legend()
    plt.ylabel(r"$p(\theta)$")

    plt.subplot(4,2,6)
    sns.kdeplot(samples_langevin_geffner[:,1], color="grey", label="geffner")
    sns.kdeplot(true_samples[:,1], color="red", label="true")
    plt.legend()
    plt.ylabel(r"$p(\theta)$")

    plt.subplot(4,2,7)
    sns.kdeplot(samples_heun[:,0], color="purple", label="heun")
    sns.kdeplot(true_samples[:,1], color="red", label="true")
    plt.legend()
    plt.ylabel(r"$p(\theta)$")

    plt.subplot(4,2,8)
    sns.kdeplot(samples_heun[:,0], color="purple", label="heun")
    sns.kdeplot(true_samples[:,1], color="red", label="true")
    plt.legend()
    plt.ylabel(r"$p(\theta)$")
    
    #plt.savefig(f"tall_posterior_{n_obs}_obs_{cov_mode}_{type_net}.pdf", format="pdf")
    plt.show()