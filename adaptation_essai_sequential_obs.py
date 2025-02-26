import torch
import numpy as np
from sbi.utils import BoxUniform
import seaborn as sns
import matplotlib.pyplot as plt
#from tasks.toy_examples import get_true_samples_nextra_obs as true
from torch.func import vmap
from functools import partial
from experiment_utils import _matrix_pow

from nse import NSE, NSELoss, ExplicitLoss,FNet
from sm_utils import train_with_validation
from tall_posterior_sampler import diffused_tall_posterior_score, euler_sde_sampler, tweedies_approximation, heun_ode_sampler
from vp_diffused_priors import get_vpdiff_gaussian_score
#from sbibm_posterior_estimation import run_train_sgm, run_sample_sgm

torch.manual_seed(42)
total_budget_train=5000
num_train = total_budget_train
n_epochs=3000
batch_size=500
n_samples=1000
type_net="default"
cov_mode="GAUSS"
training = False
sampling = not training
loss_type="denoising"
prior_beta_low=torch.tensor([0.0])
prior_beta_high=torch.tensor([1.0])
mu_prior = torch.ones(2)
cov_prior = torch.eye(2)*3
prior_beta = torch.distributions.MultivariateNormal(mu_prior, cov_prior)
cov_lik = torch.eye(2)*2

def simulator1(theta):
    return torch.distributions.MultivariateNormal(theta, cov_lik).sample() 

theta_true = torch.tensor([0.5,0.5]).unsqueeze(0)
x_obs = simulator1(theta_true)
print(x_obs)
print(theta_true.size())
print(x_obs.size())

cov_post = torch.linalg.inv(torch.linalg.inv(cov_prior)+torch.linalg.inv(cov_lik))

def mu_post(x):
    "mean of the individual posterior p(beta|x)"
    return cov_post@(torch.linalg.inv(cov_lik)@x.reshape(2,1) + torch.linalg.inv(cov_prior)@mu_prior.reshape(2,1))

def true_post_score(theta,x):
    "analytical score of the true individual posterior"
    return -torch.linalg.inv(cov_post)@(theta.reshape(2,1)-mu_post(x))

def true_diff_post_score(theta,x,t, score_net):
    "analytical score of the true diffused posterior p_t(beta|x)"
    alpha_t = score_net.alpha(t).item()
    return -torch.linalg.inv((1-alpha_t)*torch.eye(2)+alpha_t*cov_post)@(theta.reshape(2,1)-alpha_t**0.5*mu_post(x))

def cov_tall_post(x):
    "covariance matrix of the tall true posterior p(beta|x0,...xn)"
    return torch.linalg.inv(torch.linalg.inv(cov_prior)+x.shape[0]*torch.linalg.inv(cov_lik))

def mu_tall_post(x):
    "mean of the true tall posterior"
    tmp = torch.linalg.inv(cov_prior)@mu_prior.reshape(2,1)
    for i in range(x.shape[0]):
        tmp += torch.linalg.inv(cov_lik)@x[i,:].reshape(2,1)
    return cov_tall_post(x)@tmp

def true_tall_post_score(theta,x):
    "analytical score of the true tall posterior"
    return -torch.linalg.inv(cov_tall_post(x))@(theta.reshape(2,1)-mu_tall_post(x))

def true_diff_tall_post_score(theta, x,t,score_net):
    "analytical score of the true diffused tall posterior p_t(beta|x0,...xn)"
    alpha_t = score_net.alpha(t)
    return -torch.linalg.inv((1-alpha_t)*torch.eye(2)+alpha_t*cov_tall_post(x))@(theta.reshape(2,1)-alpha_t**0.5*mu_tall_post(x))

def wasserstein_dist(mu_1,mu_2,cov_1,cov_2):
    """Compute the Wasserstein distance between 2 Gaussians"""
    # G.Peyré & M. Cuturi (2020), Computational Optimal Transport, eq 2.41 
    sqrtcov1 = _matrix_pow(cov_1, 0.5)
    covterm = torch.trace(
        cov_1 + cov_2 - 2 * _matrix_pow(sqrtcov1 @ cov_2 @ sqrtcov1, 0.5)
    )
    return ((mu_1-mu_2)**2).sum() + covterm

#create training datset
beta_train=prior_beta.sample((num_train,)) # get different beta, get pairs (alpha,beta)
x_train=simulator1(theta=beta_train)
data_train = torch.utils.data.TensorDataset(beta_train, x_train)
print("########################################")
print("Training data:", beta_train.shape, x_train.shape)
print("########################################")

score_network = NSE(theta_dim=beta_train.size(1), 
                    x_dim=x_train.size(1),
                    net_type=type_net,
                    freqs=1,
                    hidden_features=[32,32])#, net_type="fnet")
print(score_network)

#define losses (explicit form or denoising approximation)
if loss_type=="denoising":
    loss=NSELoss(score_network)
else:
    loss=ExplicitLoss(inv_cov_prior=torch.linalg.inv(cov_prior), 
                             mu_prior=mu_prior, 
                             inv_cov_lik=torch.linalg.inv(cov_lik),
                             cov_post=cov_post,
                             estimator=score_network)

#define noise schedule
beta_min = 0.1
beta_max = 40
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

def true_matrices(t):
    alpha=score_network.alpha(t)
    A = (1-alpha)[...,None]**0.5*torch.linalg.inv((1-alpha)[...,None]*(torch.eye(2).repeat(t.shape[0],1,1))+alpha[...,None]*(cov_post.repeat(t.shape[0],1,1)))
    B = -alpha[...,None]**0.5 * (cov_post@torch.linalg.inv(cov_lik)).repeat(t.shape[0],1,1)
    C = -alpha**0.5*(cov_post@torch.linalg.inv(cov_prior)@mu_prior).repeat(t.shape[0],1)
    return A.reshape(A.size(0),-1),B.reshape(B.size(0),-1),C

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

beta_train_mean, beta_train_std = beta_train.mean(dim=0), beta_train.std(dim=0)
x_train_mean, x_train_std = x_train.mean(dim=0), x_train.std(dim=0)

# normalize theta
beta_train_norm = (beta_train - beta_train_mean) / beta_train_std
# normalize x
x_train_norm = (x_train - x_train_mean) / x_train_std
# dataset for dataloader

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
    file = "trained_network_gauss_default.pkl"
    score_network.load_state_dict(torch.load(file, weights_only=True))
    score_network.eval()

    # Sample from the true posterior
    tmp_beta= prior_beta.sample((1,))
    tmp_x = simulator1(tmp_beta)
    time = torch.linspace(1e-3,1.0,1000)
    true_score=true_diff_post_score(tmp_beta,tmp_x,torch.ones(1)*time[0].item(),score_network).reshape(1,2)
    est_score=score_network.score(theta=tmp_beta,x=tmp_x,t=time[0].item()*torch.ones(1)).detach()
    for i in range(1,time.size(0)):
        t= time[i].item()
        tmp_est=score_network.score(theta=tmp_beta,x=tmp_x,t=t*torch.ones(1)).detach()
        tmp_true=true_diff_post_score(tmp_beta,tmp_x,torch.ones(1)*t,score_network).reshape(1,2)
        true_score = torch.cat((true_score,tmp_true), dim=0)
        est_score = torch.cat((est_score,tmp_est), dim=0)

    # if type_net=="gaussian":
    #     true_A, true_B, true_C = true_matrices(time.unsqueeze(1))
    #     est_A, est_B, est_C=score_network.net.est_matrices(time.unsqueeze(1))
        
    #     plt.figure(figsize=(10,10))
    #     plt.plot(time,((true_A-est_A)**2).mean(dim=1), label="A")
    #     plt.plot(time,((true_B-est_B)**2).mean(dim=1), label="B")
    #     plt.plot(time,((true_C-est_C)**2).mean(dim=1), label="C")
    #     plt.legend()
    #     plt.xlabel(r"$t$")
    #     plt.title(r"Difference in $L_2-$norm between true and estimated matrices")
    #     plt.show()
    # if type_net=="gaussian_alpha":
    #     est_alpha = score_network.net.scaling_alpha(time.unsqueeze(1)).detach().squeeze(1)
    #     true_alpha= alpha(time)
    #     plt.figure(figsize=(10,10))
    #     plt.plot(time,true_alpha, label="true alpha")
    #     plt.plot(time,est_alpha, label="est alpha")
    #     plt.legend()
    #     plt.xlabel(r"$t$")
    #     plt.title(r"Difference in $L_2-$norm between true and estimated matrices")
    #     plt.show()
    # print("#########################################")
    # plt.figure(figsize=(15,15))
    # plt.subplot(221)
    # # plt.scatter(time,true_score[:,0], color="red", label="true")
    # # plt.scatter(time,est_score[:,0], color="blue", label="estimated")
    # plt.plot(time[10:],(est_score[10:,0]-true_score[10:,0])**2, color="blue", label="estimated")
    # plt.legend()
    # plt.title("Dimension 0")
    # plt.xlabel("t")
    # plt.ylabel(r"$||\nabla_{\theta}\log p_t(\theta|x_0)-s_{\phi}(\theta,x_0,t)||^2$",fontsize=12)
    # plt.subplot(222)
    # # plt.scatter(time,true_score[:,1], color="red", label="true")
    # # plt.scatter(time,est_score[:,1], color="blue", label="estimated")
    # plt.plot(time[10:],(est_score[10:,1]-true_score[10:,1])**2, color="blue", label="estimated")
    # plt.legend()
    # plt.title("Dimension 1")
    # plt.ylabel(r"$||\nabla_{\theta}\log p_t(\theta|x_0)-s_{\phi}(\theta,x_0,t)||^2$",fontsize=12)
    # plt.xlabel("t")
    # plt.subplot(223)
    # # plt.scatter(time,true_score[:,0], color="red", label="true")
    # # plt.scatter(time,est_score[:,0], color="blue", label="estimated")
    # plt.plot(time[:10],(est_score[:10,0]-true_score[:10,0])**2, color="blue", marker='o',label="estimated")
    # plt.legend()
    # plt.title("Dimension 0 - near true score")
    # plt.xlabel("t")
    # plt.ylabel(r"$||\nabla_{\theta}\log p_t(\theta|x_0)-s_{\phi}(\theta,x_0,t)||^2$",fontsize=12)

    # plt.subplot(224)
    # # plt.scatter(time,true_score[:,1], color="red", label="true")
    # # plt.scatter(time,est_score[:,1], color="blue", label="estimated")
    # plt.plot(time[:10],(est_score[:10,1]-true_score[:10,1])**2, color="blue", marker='o',label="estimated")
    # plt.legend()
    # plt.title("Dimension 1 - near true score")
    # plt.xlabel("t")
    # plt.ylabel(r"$||\nabla_{\theta}\log p_t(\theta|x_0)-s_{\phi}(\theta,x_0,t)||^2$",fontsize=12)

    # plt.suptitle("Evolution of the scores with time - n_obs = 1")
    # plt.savefig(f"1_obs_{cov_mode}_{type_net}_net_score.pdf", format="pdf")
    # plt.show()

    n_obs = 50
    tall_x = simulator1(tmp_beta.repeat(n_obs,1))
    prior_score_fn = get_vpdiff_gaussian_score(mu_prior,cov_prior,score_network)

    print(tall_x.size())
    wass_euler=[]
    wass_heun=[]
    wass_ddim_jac=[]
    wass_ddim_gauss=[]
    wass_geffner=[]
    x_abs = torch.linspace(0,n_obs,11)
    x_abs[0]=1
    # x_abs=torch.linspace(1,n_obs,n_obs)

    for i in x_abs:
        i=int(i.item())
        print(i)
        tmp_tall_x = (tall_x[:i,:])
        print(tmp_tall_x.size())
        print("################### COV ESTIMATION ############")
        cov_est = vmap(
                        lambda x: score_network.ddim(
                            shape=(n_samples,), x=x, steps=100, eta=0.5
                        ),randomness="different")(tmp_tall_x[:,None,:])
        # size (n_obs, n_samples, dim theta)
        # .mT transpose the last 2 dims
        cov_est = vmap(lambda x: torch.cov(x.mT))(cov_est)
        # size (n_extra+1,dim theta x dim theta)
        
        # try_cov = cov_post.repeat(tmp_tall_x.shape[0],1,1)
        print("########## START DIFF SCORE ###########")
        tall_score_fnc = partial(
                        diffused_tall_posterior_score,
                        prior=prior_beta, 
                        prior_type="gaussian",
                        prior_score_fn=prior_score_fn,  # analytical prior score function
                        x_obs=tmp_tall_x,  # observations
                        nse=score_network,  # trained score network
                        dist_cov_est=cov_est,
                        cov_mode=cov_mode,
                    )
        tall_score_fnc_jac = partial(
                        diffused_tall_posterior_score,
                        prior=prior_beta, 
                        prior_type="gaussian",
                        prior_score_fn=prior_score_fn,  # analytical prior score function
                        x_obs=tmp_tall_x,  # observations
                        nse=score_network,  # trained score network
                        #dist_cov_est=cov_est,
                        cov_mode="JAC",
                    )
        # true_tall_score=true_diff_tall_post_score(tmp_beta,tmp_tall_x,time[0].item()*torch.ones(1), score_network).reshape(1,2)
        # est_tall_score=tall_score_fnc(tmp_beta,time[0]).detach()
        # est_tall_score_jac=tall_score_fnc_jac(tmp_beta,time[0]).detach()
        
        # for i in range(1,time.size(0)):
        #     t= time[i]
        #     # print(t)
        #     tmp_est=tall_score_fnc(tmp_beta,t).detach()
        #     tmp_est_jac=tall_score_fnc_jac(tmp_beta,t).detach()
        #     tmp_true=true_diff_tall_post_score(tmp_beta,tmp_tall_x, t.item()*torch.ones(1), score_network).reshape(1,2)
        #     true_tall_score = torch.cat((true_tall_score,tmp_true), dim=0)
            
        #     est_tall_score = torch.cat((est_tall_score,tmp_est), dim=0)
        #     est_tall_score_jac = torch.cat((est_tall_score_jac,tmp_est_jac), dim=0)
        #     # if t>0.1680 and t<0.1730:
        #     #     print("in the loop",tmp_est)
        # # true_tall_score = torch.cat((true_tall_score,torch.zeros((100,2))),dim=0)
        # # est_tall_score = torch.cat((est_tall_score,torch.zeros((100,2))),dim=0)
        # # est_tall_score_jac = torch.cat((est_tall_score_jac,torch.zeros((100,2))),dim=0)
        # #print(time[168:173])
        # # print("est tall score",est_tall_score[168:173,:])
        # plt.figure(figsize=(15,15))
        # plt.subplot(221)
        # plt.plot(time,true_tall_score[:,0], color="red", label="true")# marker='o')
        # plt.plot(time,est_tall_score[:,0], color="blue", label=f"estimated {cov_mode}")#,marker='o')
        # plt.plot(time,est_tall_score_jac[:,0], color="green", label=f"estimated JAC")#,marker='o')
        
        # # plt.scatter(time[20:],(est_tall_score[20:,0]-true_tall_score[20:,0])**2, color="blue", label="GAUSS")
        # # plt.scatter(time[20:],(est_tall_score_jac[20:,0]-true_tall_score[20:,0])**2, color="green", label="JAC")
        # plt.legend()
        # plt.title("Dimension 0")
        # plt.xlabel("t")
        # plt.ylabel(r"$s_{\psi}(\theta,x_{0:n},t)$",fontsize=12)

        # plt.subplot(222)
        # plt.plot(time,true_tall_score[:,1], color="red", label="true")#,marker='o')
        # plt.plot(time,est_tall_score[:,1], color="blue", label=f"estimated {cov_mode}")#,marker='o')
        # plt.plot(time,est_tall_score_jac[:,1], color="green", label=f"estimated JAC")#,marker='o')
        
        # # plt.scatter(time[20:],(est_tall_score[20:,1]-true_tall_score[20:,1])**2, color="blue", label="GAUSS")
        # # plt.scatter(time[20:],(est_tall_score_jac[20:,1]-true_tall_score[20:,1])**2, color="green", label="JAC")
        # plt.legend()
        # plt.title("Dimension 1")
        # plt.ylabel(r"$s_{\psi}(\theta,x_{0:n},t)$",fontsize=12)

        # plt.xlabel("t")
        # plt.suptitle(f" Tall posterior scores with analytical individual scores - n_obs = {n_obs}")

        # plt.subplot(223)
        # # plt.scatter(time,true_tall_score[:,0], color="red", label="true")
        # # plt.scatter(time,est_tall_score[:,0], color="blue", label="estimated")
        # plt.plot(time,(est_tall_score[:,0]-true_tall_score[:,0])**2, color="blue", label="GAUSS")
        # plt.plot(time,(est_tall_score_jac[:,0]-true_tall_score[:,0])**2, color="green", label="JAC")
        # plt.legend()
        # plt.title("Dimension 0")
        # plt.xlabel("t")
        # plt.ylabel(r"$||\nabla_{\theta}\log p(\theta|x_0,...x_n)-s_{\psi}(\theta,x_{0:n},t)||^2$",fontsize=12)


        # plt.subplot(224)
        # # plt.scatter(time,true_tall_score[:,1], color="red", label="true")
        # # plt.scatter(time,est_tall_score[:,1], color="blue", label="estimated")
        # plt.plot(time,(est_tall_score[:,1]-true_tall_score[:,1])**2, color="blue", label="GAUSS")
        # plt.plot(time,(est_tall_score_jac[:,1]-true_tall_score[:,1])**2, color="green", label="JAC")
        # plt.legend()
        # plt.title("Dimension 1")
        # plt.ylabel(r"$||\nabla_{\theta}\log p(\theta|x_0,...x_n)-s_{\psi}(\theta,x_{0:n},t)||^2$",fontsize=12)
        
        # plt.savefig(f"{n_obs}_obs_{cov_mode}_{type_net}_net_score.pdf", format="pdf")
        # plt.show()

        true_samples_tall_post=torch.distributions.MultivariateNormal(mu_tall_post(tmp_tall_x).squeeze(1), cov_tall_post(tmp_tall_x)).sample((n_samples,)) 
        true_mean=mu_tall_post(tmp_tall_x)
        true_cov=cov_tall_post(tmp_tall_x)

        print("############### EULER ###############")
        # sampling with euler
        samples_euler,_ = euler_sde_sampler( #sample with reverse SDE and the score at all times t
                tall_score_fnc_jac,
                n_samples,
                dim_theta=tmp_beta.shape[-1],
                beta=score_network.beta,
                debug=False,
                #theta_clipping_range=(-10,10),
            )
        samples_euler = samples_euler.detach()
        mean_euler=samples_euler.mean(dim=0)
        cov_euler=samples_euler.mT.cov()
        wass_euler.append(wasserstein_dist(true_mean,mean_euler,true_cov,cov_euler).item())

        print("############### HEUN ###############")

        samples_heun,_ = heun_ode_sampler( #sample with reverse SDE and the score at all times t
                tall_score_fnc_jac,
                n_samples,
                dim_theta=tmp_beta.shape[-1],
                beta=score_network.beta,
                #debug=False,
                #theta_clipping_range=(-10,10),
            )
        samples_heun = samples_heun.detach()
        mean_heun=samples_heun.mean(dim=0)
        cov_heun=samples_heun.mT.cov()
        wass_heun.append(wasserstein_dist(true_mean,mean_heun,true_cov,cov_heun).item())
        
        print("############### DDIM GAUSS ###############")

        steps=400
        samples_ddim_gauss = score_network.ddim(
                        shape=(n_samples,),
                        x=tmp_tall_x,
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
                        dist_cov_est=cov_est, #if GAUSS, use estimated cov at time 0 to compute Sigma_t,j^-1
                        dist_cov_est_prior=None,
                        cov_mode="GAUSS",
                        verbose=True,
                    )
        samples_ddim_gauss = samples_ddim_gauss.detach()
        mean_ddim_gauss=samples_ddim_gauss.mean(dim=0)
        cov_ddim_gauss=samples_ddim_gauss.mT.cov()
        print(samples_ddim_gauss[:5,:])
        print(cov_ddim_gauss)
        wass_ddim_gauss.append(wasserstein_dist(true_mean,mean_ddim_gauss,true_cov,cov_ddim_gauss).item())

        print("############### GEFFNER ###############")
        
        samples_langevin_geffner = score_network.annealed_langevin_geffner(
                            shape=(n_samples,),
                            x=tmp_tall_x,
                            prior_score_fn=prior_score_fn,
                            clf_free_guidance=False,
                            steps=400,
                            lsteps=5,
                            tau=0.5,
                            theta_clipping_range=(-10,10),
                            verbose=True,
                        )
        samples_langevin_geffner = samples_langevin_geffner.detach()
        mean_langevin_geffner=samples_langevin_geffner.mean(dim=0)
        cov_langevin_geffner=samples_langevin_geffner.mT.cov()
        wass_geffner.append(wasserstein_dist(true_mean,mean_langevin_geffner,true_cov,cov_langevin_geffner).item())
        
        print("############### DDIM JAC ###############")

        steps = 1000
        samples_ddim_jac = score_network.ddim(
                        shape=(n_samples,),
                        x=tmp_tall_x,
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
                        cov_mode="JAC",
                        verbose=True,
                    )
        samples_ddim_jac = samples_ddim_jac.detach()
        mean_ddim_jac=samples_ddim_jac.mean(dim=0)
        cov_ddim_jac=samples_ddim_jac.mT.cov()
        wass_ddim_jac.append(wasserstein_dist(true_mean,mean_ddim_jac,true_cov,cov_ddim_jac).item())
        print(samples_ddim_jac[:5,:])

    print(x_abs)
    plt.figure(figsize=(10,10))
    plt.subplot(3,2,1)
    plt.plot(x_abs,wass_euler,label="euler", linestyle="dashed",color="blue",marker='o')
    plt.legend()
    plt.title("Euler")
    plt.xlabel("nb observations")
    plt.ylabel(r"$\mathcal{W}_2^2(true,est)$")
    plt.xticks(x_abs)

    
    plt.subplot(3,2,2)
    plt.plot(x_abs,wass_ddim_gauss,label="ddim GAUSS", linestyle="dashed",color="green", marker='o')
    plt.legend()
    plt.title("DDIM GAUSS")
    plt.xlabel("nb observations")
    plt.ylabel(r"$\mathcal{W}_2^2(true,est)$")
    plt.xticks(x_abs)
    
    plt.subplot(3,2,3)
    plt.plot(x_abs,wass_ddim_jac,label="ddim JAC",linestyle="dashed", color="orange",marker='o')
    plt.legend()
    plt.title("DDIM JAC")
    plt.xlabel("nb observations")
    plt.ylabel(r"$\mathcal{W}_2^2(true,est)$")
    plt.xticks(x_abs)


    plt.subplot(3,2,4)
    plt.plot(x_abs,wass_geffner,label="geffner", linestyle="dashed",color="grey",marker='o')
    plt.legend()
    plt.title("Geffner")
    plt.xlabel("nb observations")
    plt.ylabel(r"$\mathcal{W}_2^2(true,est)$")
    plt.xticks(x_abs)


    plt.subplot(3,2,5)
    plt.plot(x_abs,wass_heun,label="heun", linestyle="dashed",color="purple",marker='o')
    plt.legend()
    plt.title("Heun")
    plt.xlabel("nb observations")
    plt.ylabel(r"$\mathcal{W}_2^2(true,est)$")
    plt.xticks(x_abs)
    
    plt.tight_layout()
    plt.show()