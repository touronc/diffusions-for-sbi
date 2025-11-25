from sbi.inference import NPSE, SNPE_C
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from sbi import inference as sbi_inference
import mlxp
from sbi.utils.metrics import c2st, unbiased_mmd_squared
import pickle
import ot
from tqdm.auto import tqdm

DIM=2
#from get_true_samples_nextra_obs import get_posterior_samples
torch.manual_seed(42)
mu_prior = torch.ones(DIM)
cov_prior = torch.eye(DIM)*3
A = torch.randn(DIM,DIM)
D,Q = torch.linalg.eig(A+A.T)
eigenvalue = torch.exp(torch.randn(DIM)+1)
cov_prior=Q.real.T @torch.diag(eigenvalue)@Q.real

prior_beta = torch.distributions.MultivariateNormal(mu_prior, cov_prior)
cov_lik = torch.eye(DIM)*2
print(cov_prior, torch.linalg.eigvals(cov_prior))

#for the total conv bound with link with prec error
# cov_prior = torch.tensor([[ 2.8203, -0.6851],
#         [-0.6851,  2.5945]])

def simulator1(theta):
    return torch.distributions.MultivariateNormal(theta, cov_lik).sample()

cov_post = torch.linalg.inv(torch.linalg.inv(cov_prior)+torch.linalg.inv(cov_lik))
prec_post = torch.linalg.inv(cov_prior)+torch.linalg.inv(cov_lik)
l2_norm_cov_post = torch.linalg.norm(cov_post,2)
l2_norm_prec_post = torch.linalg.norm(prec_post,2)

def mu_post(x,cov_post,cov_prior):
    "mean of the individual posterior p(beta|x)"
    # print("check shape",x.shape, x.ndim)
    if x.ndim==1:
        x=x.unsqueeze(0)
    inv_lik = torch.linalg.inv(cov_lik)[None,:,:]@x.mT #(num_train,2,1)
    inv_prior = torch.linalg.inv(cov_prior)@mu_prior[:,None] #(2,1)
    # return cov_post@(torch.linalg.inv(cov_lik)@x.reshape(2,1) + torch.linalg.inv(cov_prior)@mu_prior.reshape(2,1))
    return cov_post[None,:,:]@(inv_lik+inv_prior)

def true_post_score(theta,x):
    "analytical score of the true individual posterior p(beta|x)"
    return -torch.linalg.inv(cov_post)@(theta.reshape(2,1)-mu_post(x))

def true_diff_post_score(theta,x,t, score_net):
    "analytical score of the true diffused posterior p_t(beta|x)"
    alpha_t = score_net.alpha(t).item()
    return -torch.linalg.inv((1-alpha_t)*torch.eye(2)+alpha_t*cov_post)@(theta.reshape(2,1)-alpha_t**0.5*mu_post(x))

def cov_tall_post(x,cov_prior):
    # print("ch", x.shape[0])
    "covariance matrix of the tall true posterior p(beta|x0,...xn)"
    return torch.linalg.inv(torch.linalg.inv(cov_prior)+x.shape[0]*torch.linalg.inv(cov_lik))

def mu_tall_post(x,cov_prior):
    "mean of the true tall posterior p(beta|x0,...xn)"
    tmp = torch.linalg.inv(cov_prior)@mu_prior.reshape(DIM,1)
    for i in range(x.shape[0]):
        tmp += torch.linalg.inv(cov_lik)@x[i,:].reshape(DIM,1)
    return cov_tall_post(x,cov_prior)@tmp

def true_tall_post_score(theta,x):
    "analytical score of the true tall posterior p(beta|x0,...xn)"
    return -torch.linalg.inv(cov_tall_post(x))@(theta.reshape(2,1)-mu_tall_post(x))

def true_diff_tall_post_score(theta, x,t,score_net):
    "analytical score of the true diffused tall posterior p_t(beta|x0,...xn)"
    alpha_t = score_net.alpha(t)
    return -torch.linalg.inv((1-alpha_t)*torch.eye(2)+alpha_t*cov_tall_post(x))@(theta.reshape(2,1)-alpha_t**0.5*mu_tall_post(x))

def _matrix_pow(matrix: torch.Tensor, p: float) -> torch.Tensor:
    r"""
    Power of a matrix using Eigen Decomposition.
    Args:
        matrix: matrix
        p: power
    Returns:
        Power of a matrix
    """
    L, V = torch.linalg.eig(matrix)
    L = L.real
    V = V.real
    return V @ torch.diag_embed(L.pow(p)) @ torch.linalg.inv(V)

def wasserstein_dist(mu_1,mu_2,cov_1,cov_2):
    """Compute the Wasserstein distance between 2 Gaussians"""
    # G.Peyré & M. Cuturi (2020), Computational Optimal Transport, eq 2.41
    sqrtcov1 = _matrix_pow(cov_1, 0.5)
    covterm = torch.trace(
        cov_1 + cov_2 - 2 * _matrix_pow(sqrtcov1 @ cov_2 @ sqrtcov1, 0.5)
    )
    return ((mu_1-mu_2)**2).sum() + covterm

class FakeNet(nn.Module):
    def __init__(self, noise_schedule, error=0.0):
        super(FakeNet, self).__init__()
        self.mean_t = noise_schedule.mean_t_fn #racine de alpha_t
        self.std_t = noise_schedule.std_fn # racine de 1-alpha_t
        self.fake_param = nn.Parameter(torch.randn(2,2))
        self.device = "cpu"#noise_schedule.device
        self.t_min = noise_schedule.t_min
        self.t_max = noise_schedule.t_max
        self.init_mean = noise_schedule.mean_t
        self.init_std = noise_schedule.std_t
        self.drift = noise_schedule.drift_fn
        self.diffusion = noise_schedule.diffusion_fn
        self.input_shape = noise_schedule.input_shape
        self.error = error

    def prior_score(self,input,time):
        dim = input.shape[-1]
        alpha_t = self.mean_t(time)**2
        std = self.std_t(time)
        cov_diff = (1-alpha_t)*torch.eye(dim)[None,:,:]+alpha_t*cov_prior[None,:,:] #1,dim,dim
        vector = input.mT-alpha_t**0.5*mu_prior[:,None]
        score = -torch.linalg.inv(cov_diff)@vector
        # print("help", input.shape, vector.shape, score.shape,(-torch.linalg.inv(cov_diff)).shape)
        return score

    def diff_prec(self,prec,time):
        """return diffused precision matrix for GAUSS method"""
        alpha_t = self.mean_t(time)**2
        return prec + alpha_t/(1-alpha_t)*torch.eye(DIM)


    def single_posterior_score(self,input,condition,time, list_prec=None,prec_prior=None):
        """return the (perturbed) indiv posterior score"""
        dim = input.shape[-1]
        alpha_t = self.mean_t(time)**2
        std = self.std_t(time)
        cov_diff = (1-alpha_t)*torch.eye(dim)+alpha_t*cov_post #dim,dim
        vector = input.mT-alpha_t**0.5*mu_post(condition)
        score = -torch.linalg.inv(cov_diff)@vector

        #perturb one dimension of the true score
        indices = torch.randint(0, DIM, (input.shape[0],1))
        one_hot = torch.nn.functional.one_hot(indices, num_classes=dim).float()

        # print("help",(score).size())
        return score.mT + self.error*one_hot

    def tall_posterior_score(self,input,condition,time, list_prec, prec_prior, error_lda=0.0):
        """return the compositional score estimate (eq 2 in paper)"""
        #ATTENTION on donne les covariances deja diffusees !
        dim = input.shape[-1]
        nobs = condition.shape[0]
        alpha_t = self.mean_t(time)**2
        std = self.std_t(time)
        indices = torch.randint(0, DIM, (input.shape[0],1))
        one_hot = torch.nn.functional.one_hot(indices, num_classes=dim).float()
        score_prior = self.prior_score(input, time)+error_lda*one_hot.mT #perturbed prior score
        # prec_prior = prec_prior + alpha_t/std**2 * torch.eye(dim) #(dim,dim)
        prec_post = torch.stack(list_prec)# + alpha_t/std**2 * torch.eye(dim)[None,:,:] #(nobs,dim,dim) compute diffused posterior precisions (GAUSS)
        Lda = (1-nobs)*prec_prior+torch.sum(prec_post, dim=0) #dim,dim
        tau = (1-nobs)*prec_prior[None,:,:]@score_prior #num_samples,dim,1
        for j in range(nobs):
            # print("chek",(prec_post[None,j,:]@self.posterior_score(input, condition[j],time).mT).shape)
            tau += prec_post[j,:]@self.single_posterior_score(input, condition[j],time).mT #num_samples,dim,1
        result = torch.linalg.solve(Lda,tau) #num_samples,dim,1
        # print(Lda.shape, tau.shape,"tall score",result.shape)
        return result.mT

    def true_tall_posterior_score(self,input,condition,time,list_prec=None, prec_prior=None):
        """returns the (perturbed) analytical tall posterior score"""
        dim = input.shape[-1]
        alpha_t = self.mean_t(time)**2
        cov_diff = (1-alpha_t)*torch.eye(dim)+alpha_t*cov_tall_post(condition) #dim,dim
        vector = input.mT-alpha_t**0.5*mu_tall_post(condition)
        score = -torch.linalg.inv(cov_diff)@vector
        indices = torch.randint(0, DIM, (input.shape[0],1))
        one_hot = torch.nn.functional.one_hot(indices, num_classes=dim).float()
        # print("help",(score).size())
        # print("cov", (mu_tall_post(condition)).shape,cov_diff.shape, vector.shape, score.shape)
        return score.mT + self.error*one_hot

    def sampling(self, num_samples, steps, condition, list_prec=None, prec_prior=None):
            ts = torch.linspace(self.t_max, self.t_min, steps)
            if condition.ndim==1:
                score_fn=self.single_posterior_score
                # print("single")
            else:
                # score_fn=self.tall_posterior_score
                score_fn=self.true_tall_posterior_score
                # print("tall")
            # condition_shape = score_estimator.condition_shape
            num_batches = 1
            init_shape = (num_samples, num_batches) + self.input_shape #(num_samples,1,2)

            eps = torch.randn(init_shape, device=self.device)
            mean, std, eps = torch.broadcast_tensors(self.init_mean, self.init_std, eps)
            samples = mean + std * eps
            pbar = tqdm(
                range(1, ts.numel()),
                disable=True,
                desc=f"Drawing {num_samples} posterior samples",
            )
            for i in pbar: #on va de t1 vers t0 (on denoise)
                t1 = ts[i - 1]
                t0 = ts[i]
                dt = t1 - t0 # positif (go from t1 to t0)
                dt_sqrt = torch.sqrt(dt)
                f = self.drift(samples, t1)
                g = self.diffusion(samples, t1)
                # score = self(samples, condition, t1)
                score = score_fn(samples, condition, t1, list_prec, prec_prior)
                f_backward = f - g**2 * score
                g_backward = g
                samples = samples - f_backward * dt + g_backward * torch.randn_like(samples) * dt_sqrt
            return samples


@mlxp.launch(config_path='./configs/')
def main(ctx: mlxp.Context):

    cfg = ctx.config
    logger = ctx.logger

    torch.manual_seed(cfg.seed)
    # mu_prior = torch.ones(2)
    # cov_prior = torch.eye(2)*3
    # A = torch.randn(2,2)
    # D,Q = torch.linalg.eig(A+A.T)
    # eigenvalue = torch.exp(torch.randn(2)+1)
    # cov_prior=Q.real.T @torch.diag(eigenvalue)@Q.real

    # prior_beta = torch.distributions.MultivariateNormal(mu_prior, cov_prior)
    # cov_lik = torch.eye(2)*20
    # print(cov_prior, torch.linalg.eigvals(cov_prior))

    # cov_post = torch.linalg.inv(torch.linalg.inv(cov_prior)+torch.linalg.inv(cov_lik))
    # prec_post = torch.linalg.inv(cov_prior)+torch.linalg.inv(cov_lik)
    # l2_norm_cov_post = torch.linalg.norm(cov_post,2)
    # l2_norm_prec_post = torch.linalg.norm(prec_post,2)
    num_train = cfg.num_train

    # def simulator(theta):
    #     return torch.distributions.MultivariateNormal(theta, cov_lik).sample()

    beta_train = prior_beta.sample((num_train,)) # dataset for the simulator
    x_train = simulator1(beta_train)

    print("Training dimensions : \n beta : ", beta_train.size(),
        "\n x : ",x_train.size())

    # diffusion model for beta parameters
    inference_beta = NPSE(prior=prior_beta, sde_type="vp")
    inference_beta.append_simulations(beta_train, x_train)
    score_estimator = inference_beta.train(max_num_epochs=10)

    error_list = torch.tensor([0.001,0.01,0.1,0.25])#,0.3,0.5])#,1,2])
    perturbed_score=FakeNet(score_estimator)

    num_samples = cfg.num_samples
    beta_o = prior_beta.sample()
    x_o = simulator1(beta_o) #size (2)

    if 0: #ordre de grandeur de epsilon_dsm
        n_obs = cfg.nextra
        extra_obs = prior_beta.sample((n_obs,))
        beta_o_long_full = torch.cat((beta_o.unsqueeze(0),extra_obs),dim=0)
        x_o_long_full = simulator1(beta_o_long_full)
        perturbed_score.error=0.0
        time = torch.linspace(0.1,1,100)
        epsilon_dsm = []
        epsilon_dsm_1 = []
        epsilon_dsm_2 = []
        for t in time :
            alpha_t = perturbed_score.mean_t(t)**2
            true_samples = torch.distributions.MultivariateNormal(alpha_t**0.5*mu_tall_post(x_o_long_full).squeeze(1), alpha_t*cov_tall_post(x_o_long_full)+(1-alpha_t)*torch.eye(2)).sample((num_samples,))
            est_score = score_estimator(true_samples, x_o, t)
            true_score = perturbed_score.single_posterior_score(true_samples[:,None,:], x_o,t)
            error = torch.mean(torch.linalg.norm(est_score[:,None,:]-true_score,2,dim=(1,2))**2)
            epsilon_dsm.append(error.item())
            est_score = score_estimator(true_samples, x_o_long_full[-1,:], t)
            true_score = perturbed_score.single_posterior_score(true_samples[:,None,:], x_o_long_full[-1,:],t)
            error = torch.mean(torch.linalg.norm(est_score[:,None,:]-true_score,2,dim=(1,2))**2)
            epsilon_dsm_2.append(error.item())
            est_score = score_estimator(true_samples, x_o_long_full[1,:], t)
            true_score = perturbed_score.single_posterior_score(true_samples[:,None,:], x_o_long_full[1,:],t)
            error = torch.mean(torch.linalg.norm(est_score[:,None,:]-true_score,2,dim=(1,2))**2)
            epsilon_dsm_1.append(error.item())

        plt.figure()
        plt.plot(time,epsilon_dsm, label=r"$x_0$")
        plt.plot(time,epsilon_dsm_1, label=r"$x_1$")
        plt.plot(time,epsilon_dsm_2, label=r"$x_2$")
        plt.title(f"{cfg.num_train} training data")
        plt.legend()
        plt.show()

    if 0: #ordre de grandeur de ||\Lambda^-1||
        n_obs = cfg.nextra
        nobs_list = torch.arange(1,n_obs+1)
        Lda_norm =[]
        list_prec_post =[]
        for _ in range(n_obs):
            A = torch.randn(2,2)
            D,Q = torch.linalg.eig(A+A.T)
            from scipy.stats import truncnorm
            import numpy as np
            mean, sd = 0, 1
            lower, upper = -np.inf, torch.log(torch.max(torch.linalg.eigvals(cov_prior).real)).item()  # upper bound = 2
            # Create truncated normal
            trunc = truncnorm((lower-mean)/sd, (upper-mean)/sd, loc=mean, scale=sd)
            eigenvalue = (torch.from_numpy(trunc.rvs(2))).float()
            eigenvalue = torch.exp(torch.randn(2)+1)
            cov_post_=Q.real.T @torch.diag(eigenvalue)@Q.real
            list_prec_post.append(torch.linalg.inv(cov_post_))
            check = torch.max(eigenvalue)<torch.max(torch.linalg.eigvals(cov_prior).real).item()
            print("true", check)
        list_prec_post = torch.stack(list_prec_post)
        for n in range(n_obs):
            # Lda_inv=torch.linalg.inv(-n*torch.linalg.inv(cov_prior)+(n+1)*prec_post)
            # print(torch.sum(list_prec_post[:n,:,:],dim=0).shape)

            Lda_inv=torch.linalg.inv(-n*torch.linalg.inv(cov_prior)+torch.sum(list_prec_post[:n+1,:,:],dim=0))
            d= torch.linalg.eigvals(Lda_inv)
            print("check",torch.min(d.real)>0)
            Lda_norm.append(torch.linalg.norm(Lda_inv,2).item())
        # Lda = torch.linalg.inv(-n_obs*diff_prec_prior+(n_obs+1)*diff_prec)
        plt.figure()
        plt.plot(nobs_list,Lda_norm, label=r"$\Vert\Lambda^{-1}\Vert_2$")
        plt.plot(nobs_list,1.0/(nobs_list), label=r"$\frac{1}{n}$")
        plt.xlabel(r"$n$")
        plt.legend()
        plt.show()
        print(h)

    if cfg.nextra==0:#covariance bounds
        # sampling with 0 extra observation
        true_samples = torch.distributions.MultivariateNormal(mu_post(x_o)[0,:,0], cov_post).sample((num_samples,))
        # inv_cov_true = torch.linalg.inv(cov_true)
        # l2_norm_prec_true = 1.0/torch.linalg.norm(cov_true,2)
        # test = torch.mean((cov_post-cov_true)**2)
        list_wass = []
        list_cov_emp = []
        list_cov_bound = []
        list_cov_bound_prev = []
        list_cov_bound_new = []
        list_cov_bound_new_1 = []
        list_prec_emp = []
        list_prec_bound = []
        error_list = torch.linspace(0.001,0.05,20)
        s = torch.linalg.norm(prec_post,2)
        m = l2_norm_cov_post**0.5
        delta = 2*(2*m+1.0/m/s)**2+4*(2**0.5+1)/s
        dico={}
        for error in error_list:
            perturbed_score.error = error.item()
            # perturbed_score.t_max = 2
            samples_beta = perturbed_score.sampling(num_samples,steps=200, condition=x_o)[:,0,:] #size (num_samples, 1, dim)
            cov_est = torch.cov(samples_beta.T)
            prec_est = torch.linalg.inv(cov_est)
            l2_norm_cov_est = torch.linalg.norm(cov_est,2) #l2 norm of \tilde \Sigma
            wass = wasserstein_dist(mu_post(x_o)[0,:,0],torch.mean(samples_beta,dim=0),cov_post,cov_est)**0.5
            if wass.item()<m/2**0.5 and wass.item()<(delta**0.5-2**0.5*(2*m+1.0/m/s))/2/(2**0.5+1):
            # cov_bound_prev = 2**1.5*torch.min(l2_norm_cov_post,l2_norm_cov_est)**0.5*wass+(1+3*2**0.5)*wass**2
            # cov_bound = 2**1.5*torch.max(l2_norm_cov_post,l2_norm_cov_est)**0.5*wass+(1+2**0.5)*wass**2
            # delta_a = (9+2**0.5)*wass**2+8*2**0.5*wass*l2_norm_cov_post**0.5
            # cov_bound_new_1 = torch.max((2**0.5*wass-delta_a**0.5/2)**2,(2**0.5*wass+delta_a**0.5/2)**2)
                cov_bound_new = 2**1.5*wass*m+wass**2*(1+2**0.5)/(1-2**0.5*wass/m)
            # diff_cov_est = torch.mean((cov_est-cov_true)**2)**0.5 #distance between \tilde \Sigma and \Sigma_emp
            # diff_cov = torch.sum((cov_est-cov_post)**2)**0.5 #distance between \tilde \Sigma and \Sigma
                diff_cov = torch.linalg.norm(cov_est-cov_post,2)
            # diff_prec_est = torch.mean((inv_cov_est-prec_priorinv_cov_true)**2)**0.5 #distance between \tilde \Sigma^(-1) and \Sigma_emp^(-1)
            # diff_prec = torch.mean((prec_est-torch.linalg.inv(cov_post))**2)**0.5 # distance between \tilde \Sigma^(-1) and \Sigma^(-1)
                diff_prec = torch.linalg.norm(prec_est-torch.linalg.inv(cov_post),2)
            # prec_bound = diff_cov*l2_norm_prec_post**2/(1-diff_cov*l2_norm_prec_post)
                prec_bound = cov_bound_new*l2_norm_prec_post**2/(1-cov_bound_new*l2_norm_prec_post)
                list_wass.append(wass.item())
                list_cov_emp.append(diff_cov.item())
            # list_cov_pop.append(diff_cov.item())
                list_prec_emp.append(diff_prec.item())
            # list_prec_pop.append(diff_prec.item())
            # list_cov_bound.append(cov_bound.item())
            # list_cov_bound_prev.append(cov_bound_prev.item())
                list_cov_bound_new.append(cov_bound_new.item())
            # list_cov_bound_new_1.append(cov_bound_new_1.item())
                list_prec_bound.append(prec_bound.item())
            # check_wass = (wass.item()<l2_norm_cov_post**0.5/2**0.5)
            # check = (diff_cov<1.0/l2_norm_prec_post)
            # print("wass",check_wass)
            # print(check)
            # plt.subplot(121)
            # sns.kdeplot(samples_beta[:,0].detach(), label=f"{error.item()}")
            # plt.subplot(122)
            # sns.kdeplot(samples_beta[:,1].detach(), label=f"{error.item()}")
            # plt.show()
        dico["cov_bound"]=list_cov_bound_new
        dico["cov_emp"]=list_cov_emp
        dico["prec_bound"]=list_prec_bound
        dico["cov_emp"]=list_prec_emp
        dico["wass"]=list_wass
        dico["epsilon_dsm"]=error_list
        logger.log_artifacts(dico, artifact_name=f"cov_prec_bound_seed_{cfg.seed}",
                            artifact_type='pickle')
        fig = plt.figure(figsize=(11,8))
        plt.subplot(221)
        plt.plot(error_list,list_wass, color="blue", marker="o", label=r"$\mathcal{W}_2(p,\tilde p)$")
        plt.xlabel(r"$\sup_t \sup_{\theta}||\nabla \log p_t(\theta\mid x)-s_{\phi}(\theta,x,t)||_2$")
        plt.legend()
        plt.title("Wasserstein error w.r.t. score error")

        plt.subplot(222)
        plt.scatter(list_wass,list_cov_emp, color="red", marker="o",label="empirical")
        # plt.plot(list_wass,list_cov_pop, color="orange",marker="o", label=r"$||\Sigma_{\text{true}}-\tilde \Sigma||_2^2$")
        # plt.scatter(list_wass,list_cov_bound, color="green",marker="o", label="theoretical bound")
        # plt.scatter(list_wass,list_cov_bound_prev, color="blue",marker="o", label="wrong bound")
        plt.scatter(list_wass,list_cov_bound_new, color="orange",marker="o", label="theoretical")
        # plt.scatter(list_wass,list_cov_bound_new_1, color="purple",marker="o", label="polynom bound")
        ymin, ymax = plt.ylim()
        plt.vlines(list_wass, ymin, ymax, color="black", ls="dashed",lw=0.7)
        plt.xlabel(r"$\mathcal{W}_2(p,\tilde p)$")
        plt.ylabel(r"$||\Sigma-\tilde \Sigma||_2$")

        plt.title("Covariance error w.r.t. Wasserstein distance")
        plt.legend()

        plt.subplot(223)
        plt.scatter(list_wass, list_prec_emp,color="red", marker="o",label="empirical")
        # plt.scatter(list_cov_pop, list_prec_pop,color="orange",marker="o", label=r"$||\Sigma^{-1}_{\text{true}}-\tilde \Sigma^{-1}||_2^2$")
        plt.scatter(list_wass,list_prec_bound, color="green",marker="o", label="theoretical")
        ymin, ymax = plt.ylim()
        plt.vlines(list_wass, ymin, ymax, color="black", ls="dashed",lw=0.7)
        plt.xlabel(r"$\mathcal{W}_2(p,\tilde p)$")
        plt.ylabel(r"$||\Sigma^{-1}-\tilde \Sigma^{-1}||_2$")
        # plt.xlabel(r"$||\Sigma_{\text{emp}}-\tilde \Sigma||_2$")
        # plt.title("Precision error w.r.t covariance error")
        plt.title("Precision error w.r.t Wasserstein error")
        plt.legend()
        plt.tight_layout()

        plt.show()
        print(h)

    if 0:#link between epsilon_dsm and epsilon and cov error
        print("start")
        error_list=torch.linspace(0.01,0.25,500)
        true_samples = torch.distributions.MultivariateNormal(mu_post(x_o)[0,:,0], cov_post).sample((num_samples,))

        list_wass = []
        list_cov_emp = []
        list_cov_bound = []
        list_cov_bound_prev = []
        list_cov_bound_new = []
        list_cov_bound_new_1 = []
        list_prec_emp = []
        list_prec_bound = []
        plot_error = []
        m = (l2_norm_cov_post**0.5).item()
        delta = (2*(2*m+1.0/m/l2_norm_prec_post)**2+4*(2**0.5+1)/l2_norm_prec_post).item()
        for error in error_list:
            perturbed_score.error = error.item()
            # perturbed_score.t_max = 2
            samples_beta = perturbed_score.sampling(num_samples,steps=200, condition=x_o)[:,0,:] #size (num_samples, 1, dim)
            cov_est = torch.cov(samples_beta.T)
            prec_est = torch.linalg.inv(cov_est)
            l2_norm_cov_est = torch.linalg.norm(cov_est,2) #l2 norm of \tilde \Sigma
            wass = wasserstein_dist(mu_post(x_o)[0,:,0],torch.mean(samples_beta,dim=0),cov_post,cov_est)**0.5
            # cov_bound_prev = 2**1.5*torch.min(l2_norm_cov_post,l2_norm_cov_est)**0.5*wass+(1+3*2**0.5)*wass**2
            # cov_bound = 2**1.5*torch.max(l2_norm_cov_post,l2_norm_cov_est)**0.5*wass+(1+2**0.5)*wass**2
            # delta_a = (9+2**0.5)*wass**2+8*2**0.5*wass*l2_norm_cov_post**0.5
            # cov_bound_new_1 = torch.max((2**0.5*wass-delta_a**0.5/2)**2,(2**0.5*wass+delta_a**0.5/2)**2)
            cov_bound_new = 2**1.5*wass*m+wass**2*(1+2**0.5)/(1-2**0.5*wass/m)
            # diff_cov_est = torch.mean((cov_est-cov_true)**2)**0.5 #distance between \tilde \Sigma and \Sigma_emp
            # diff_cov = torch.sum((cov_est-cov_post)**2)**0.5 #distance between \tilde \Sigma and \Sigma
            diff_cov = torch.linalg.norm(cov_est-cov_post,2)
            # diff_prec_est = torch.mean((inv_cov_est-prec_priorinv_cov_true)**2)**0.5 #distance between \tilde \Sigma^(-1) and \Sigma_emp^(-1)
            # diff_prec = torch.mean((prec_est-torch.linalg.inv(cov_post))**2)**0.5 # distance between \tilde \Sigma^(-1) and \Sigma^(-1)
            diff_prec = torch.linalg.norm(prec_est-torch.linalg.inv(cov_post),2)
            # prec_bound = diff_cov*l2_norm_prec_post**2/(1-diff_cov*l2_norm_prec_post)
            prec_bound = cov_bound_new*l2_norm_prec_post**2/(1-cov_bound_new*l2_norm_prec_post)
            # print(wass.item(), (delta**0.5-2**0.5*(2*m+1.0/m/l2_norm_prec_post.item()))/2/(2**0.5+1))
            if wass.item()<m/2**0.5 and wass.item()<(delta**0.5-2**0.5*(2*m+1.0/m/l2_norm_prec_post.item()))/2/(2**0.5+1):
                check = diff_cov<1.0/l2_norm_prec_post
                print("valid assumption", check)
                list_wass.append(wass.item())
                list_cov_emp.append(diff_cov.item())
                list_prec_emp.append(diff_prec.item())
                list_cov_bound.append(cov_bound_new.item())
                list_prec_bound.append(prec_bound.item())
                plot_error.append(error.item())


        fig, ax = plt.subplots(1,4,figsize=(11,8))

        ax[0].loglog(torch.tensor(plot_error)**2,list_wass, color="blue", label=r"$\mathcal{W}_2(p,\tilde p)$")
        ax[0].loglog(torch.tensor(plot_error)**2,torch.tensor(plot_error), color="blue", ls="dotted", label=r"$\epsilon_\text{DSM}^2 \to \epsilon_\text{DSM}$")
        ax[0].set_xlabel(r"$\sup_t \sup_{\theta}||\nabla \log p_t(\theta\mid x)-s_{\phi}(\theta,x,t)||^2$")
        ax[0].legend()
        ax[0].set_title("Wasserstein error w.r.t. score error")
        ax[1].loglog(torch.tensor(plot_error)**2,list_cov_emp, color="red",label=r"$||\Sigma -\tilde \Sigma||_2$")
        # ax[1].plot(error_list**2,list_cov_bound, color="red", ls="dotted",label="bound")
        ax[1].set_xlabel(r"$\sup_t \sup_{\theta}||\nabla \log p_t(\theta\mid x)-s_{\phi}(\theta,x,t)||^2$")

        ax[1].legend()
        ax[1].set_title("Covariance error w.r.t. score error")
        # plt.plot(list_wass,list_cov_pop, color="orange",marker="o", label=r"$||\Sigma_{\text{true}}-\tilde \Sigma||_2^2$")
        ax[2].loglog(torch.tensor(plot_error)**2,list_prec_emp, color="green", label=r"$||\Sigma^{-1}-\tilde \Sigma^{-1}||_2$")
        # ax[2].plot(error_list**2,list_prec_bound, color="green", ls="dotted", label="bound")
        ax[2].set_xlabel(r"$\sup_t \sup_{\theta}||\nabla \log p_t(\theta\mid x)-s_{\phi}(\theta,x,t)||^2$")
        ax[2].legend()
        ax[2].set_title("Precision error w.r.t. score error")
        ax[3].scatter(list_wass,list_cov_emp, color="green", label=r"$||\Sigma^{-1}-\tilde \Sigma^{-1}||_2$")
        # ax[3].plot(list_wass,list_wass, color="green", ls="dotted", label=r"$||\Sigma^{-1}-\tilde \Sigma^{-1}||_2$")
        ax[3].plot(list_wass,(torch.tensor(list_wass)+torch.tensor(list_wass)**2)/(1-torch.tensor(list_wass)), color="green", ls="dotted", label=r"$||\Sigma^{-1}-\tilde \Sigma^{-1}||_2$")
        ax[3].set_xscale("log")
        ax[3].set_yscale("log")
        # plt.scatter(list_wass,list_cov_bound_prev, color="blue",marker="o", label="wrong bound")
        # plt.scatter(list_wass,list_cov_bound_new, color="orange",marker="o", label="new bound")
        # plt.scatter(list_wass,list_cov_bound_new_1, color="purple",marker="o", label="polynom bound")
        # ymin, ymax = plt.ylim()
        # plt.vlines(list_wass, ymin, ymax, color="black", ls="dashed")

        # plt.xlabel(r"$\mathcal{W}_2(p,\tilde p)$")
        # plt.title("Covariance error w.r.t. Wasserstein distance")
        # plt.legend()
        # plt.subplot(223)
        # plt.scatter(list_cov_emp, list_prec_emp,color="red", marker="o",label=r"$||\Sigma^{-1}_{\text{emp}}-\tilde \Sigma^{-1}||_2$")
        # # plt.scatter(list_cov_pop, list_prec_pop,color="orange",marker="o", label=r"$||\Sigma^{-1}_{\text{true}}-\tilde \Sigma^{-1}||_2^2$")
        # plt.scatter(list_cov_emp,list_prec_bound, color="green",marker="o", label="theoretical bound")
        # ymin, ymax = plt.ylim()
        # plt.vlines(list_cov_emp, ymin, ymax, color="black", ls="dashed")
        # plt.xlabel(r"$||\Sigma_{\text{emp}}-\tilde \Sigma||_2$")
        # plt.title("Precision error w.r.t covariance error")
        # plt.legend()
        fig.tight_layout()

        plt.show()
        print(h)

    if 0:#covariance bounds for several x
        # sampling with 0 extra observation
        n_obs = cfg.nextra
        # extra_obs = prior_beta.sample((n_obs,))
        # beta_o_long_full = torch.cat((beta_o.unsqueeze(0),extra_obs),dim=0)
        extra_obs = [x_o]
        for _ in range(n_obs):
            extra_obs.append(simulator1(beta_o))
        x_o_long_full = torch.stack(extra_obs)
        # inv_cov_true = torch.linalg.inv(cov_true)
        # l2_norm_prec_true = 1.0/torch.linalg.norm(cov_true,2)
        # test = torch.mean((cov_post-cov_true)**2)
        list_wass = []
        list_cov_emp = []
        list_cov_bound = []
        list_cov_bound_prev = []
        list_prec_emp = []
        list_prec_bound = []
        fig = plt.figure(figsize=(11,8))
        for j in range(n_obs+1):
            diff_xj=[]
            true_samples = torch.distributions.MultivariateNormal(mu_post(x_o_long_full[j])[0,:,0], cov_post).sample((num_samples,))
            for i,error in enumerate(error_list):
                perturbed_score.error = error.item()
                samples_beta = perturbed_score.sampling(num_samples,steps=2000, condition=x_o_long_full[j][None,:])[:,0,:] #size (num_samples, 1, dim)
                cov_est = torch.cov(samples_beta.T)
                prec_est = torch.linalg.inv(cov_est)
                l2_norm_cov_est = torch.linalg.norm(cov_est,2) #l2 norm of \tilde \Sigma
                # wass = wasserstein_dist(mu_post(x_o_long_full[j]),torch.mean(samples_beta),cov_post,cov_est)**0.5
                diff_cov = torch.mean((cov_est-cov_post)**2)**0.5 #distance between \tilde \Sigma and \Sigma
                diff_prec = torch.mean((prec_est-torch.linalg.inv(cov_post))**2)**0.5 # distance between \tilde \Sigma^(-1) and \Sigma^(-1)
                diff_xj.append(diff_prec.item())
            plt.plot(error_list,diff_xj, marker="o",label=rf"$x_{{{j}}}$")
            plt.legend()
        plt.show()
            # cov_bound_prev = 2**1.5*torch.min(l2_norm_cov_post,l2_norm_cov_est)**0.5*wass+(1+3*2**0.5)*wass**2
            # cov_bound = 2**1.5*torch.max(l2_norm_cov_post,l2_norm_cov_est)**0.5*wass+(1+2**0.5)*wass**2

            # diff_cov_est = torch.mean((cov_est-cov_true)**2)**0.5 #distance between \tilde \Sigma and \Sigma_emp

            # diff_prec_est = torch.mean((inv_cov_est-prec_priorinv_cov_true)**2)**0.5 #distance between \tilde \Sigma^(-1) and \Sigma_emp^(-1)
            # prec_bound = diff_cov*l2_norm_prec_post**2/(1-diff_cov*l2_norm_prec_post)
            # list_wass.append(wass.item())
            # list_cov_emp.append(diff_cov.item())
            # # list_cov_pop.append(diff_cov.item())
            # list_prec_emp.append(diff_prec.item())
            # # list_prec_pop.append(diff_prec.item())
            # list_cov_bound.append(cov_bound.item())
            # list_cov_bound_prev.append(cov_bound_prev.item())
            # list_prec_bound.append(prec_bound.item())
            # check = (diff_cov<1.0/l2_norm_prec_post)
            # print(check)

        # plt.subplot(221)
        # plt.plot(error_list,list_wass, color="blue", marker="o", label=r"$\mathcal{W}_2(p,\tilde p)$")
        # plt.xlabel(r"$\sup_t \sup_{\theta}||\nabla \log p_t(\theta\mid x)-s_{\phi}(\theta,x,t)||_2$")
        # plt.legend()
        # plt.title("Wasserstein error w.r.t. score error")
        # plt.subplot(222)
        # plt.scatter(list_wass,list_cov_emp, color="red", marker="o",label=r"$||\Sigma_{\text{emp}}-\tilde \Sigma||_2$")
        # # plt.plot(list_wass,list_cov_pop, color="orange",marker="o", label=r"$||\Sigma_{\text{true}}-\tilde \Sigma||_2^2$")
        # plt.scatter(list_wass,list_cov_bound, color="green",marker="o", label="theoretical bound")
        # plt.scatter(list_wass,list_cov_bound_prev, color="blue",marker="o", label="prev bound")
        # ymin, ymax = plt.ylim()
        # plt.vlines(list_wass, ymin, ymax, color="black", ls="dashed")

        # plt.xlabel(r"$\mathcal{W}_2(p,\tilde p)$")
        # plt.title("Covariance error w.r.t. Wasserstein distance")
        # plt.legend()
        # plt.subplot(223)
        # plt.scatter(list_cov_emp, list_prec_emp,color="red", marker="o",label=r"$||\Sigma^{-1}_{\text{emp}}-\tilde \Sigma^{-1}||_2$")
        # # plt.scatter(list_cov_pop, list_prec_pop,color="orange",marker="o", label=r"$||\Sigma^{-1}_{\text{true}}-\tilde \Sigma^{-1}||_2^2$")
        # plt.scatter(list_cov_emp,list_prec_bound, color="green",marker="o", label="theoretical bound")
        # ymin, ymax = plt.ylim()
        # plt.vlines(list_cov_emp, ymin, ymax, color="black", ls="dashed")
        # plt.xlabel(r"$||\Sigma_{\text{emp}}-\tilde \Sigma||_2$")
        # plt.title("Precision error w.r.t covariance error")
        # plt.legend()
        # plt.tight_layout()

        # plt.show()

    if 0:#for the L2 norm WITHOUT square
        # inference with several additional observations
        n_obs = cfg.nextra
        extra_obs = prior_beta.sample((n_obs,))
        beta_o_long_full = torch.cat((beta_o.unsqueeze(0),extra_obs),dim=0)
        x_o_long_full = simulator1(beta_o_long_full)
        print("\nDimensions for ", n_obs+1, " observations: \n beta : ", beta_o_long_full.size(),
        "\n x : ",x_o_long_full.size())
        eps=0.1 #error of precision matrices
        eps_lda=0.01 #error of prior precision
        error_lda = 0.05 #error for prior score

        fig = plt.figure(figsize=(13.5,10))
        plt.suptitle(fr"$\epsilon_{{\lambda}}={eps_lda}, \ \ \epsilon=\sup_t \sup_i ||\Sigma^{{-1}}_{{t,i}}-\tilde \Sigma^{{-1}}_{{t,i}}||\leq {eps}, \ \ \epsilon_{{\text{{DSM,}},\lambda}}=\sup_t E_{{\nu_t}}||\nabla \log p_t(\theta)-s(\theta,t)||\leq {error_lda}$")
        colors = ["red","blue","green","orange"]
        for k,n_obs in enumerate([2,5,7,cfg.nextra]):
            x_o_long = x_o_long_full[:n_obs+1,:]
            beta_o_long = beta_o_long_full[:n_obs+1,:]
            for i, error in enumerate(error_list) :
                tall_error_list=[]
                error_bound_list=[]
                Lda_error = []
                Lda_bound = []
                time_Lda = []
                time_score = []
                for time in torch.linspace(0.05,1.0,100):
                    perturbed_score.error=0.0
                    alpha_t = perturbed_score.mean_t(time)**2
                    true_samples = torch.distributions.MultivariateNormal(alpha_t**0.5*mu_tall_post(x_o_long).squeeze(1), alpha_t*cov_tall_post(x_o_long)+(1-alpha_t)*torch.eye(2)).sample((num_samples,))
                    diff_prec = perturbed_score.diff_prec(prec_post,time) #true diffused post precisions
                    M = torch.linalg.norm(diff_prec,2) #all the cov are equal in our case
                    diff_prec_prior = perturbed_score.diff_prec(torch.linalg.inv(cov_prior),time) #true diffused prior precision
                    M_lda = torch.linalg.norm(diff_prec_prior,2)
                    # Lda = torch.linalg.norm(torch.linalg.inv(-n_obs*diff_prec_prior+(n_obs+1)*diff_prec),2)
                    Lda = torch.linalg.inv(-n_obs*diff_prec_prior+(n_obs+1)*diff_prec)
                    norm = torch.linalg.norm(Lda,2)
                    L=0
                    list_prec=[] #store covariance of **perturbed** indiv posterior
                    for j in range(n_obs+1):
                        norm_indiv_score = torch.linalg.norm(perturbed_score.single_posterior_score(true_samples[:,None,:],x_o_long[j,:],time),2, dim=(1,2))
                        list_prec.append(diff_prec+eps*torch.eye(2)) #Sigma tilde
                        if L<=torch.mean(norm_indiv_score).item():
                            L=torch.mean(norm_indiv_score).item()
                    tilde_Lda = torch.linalg.inv(-n_obs*(diff_prec_prior+eps_lda*torch.eye(2))+torch.stack(list_prec).sum(dim=0))
                    # test = torch.linalg.norm(Lda-tilde_Lda,2)
                    # bound=(n_obs*eps_lda+(n_obs+1)*error)*norm**2/(1-(n_obs*eps_lda+(n_obs+1)*error)*norm)
                    # if n_obs*eps_lda+(n_obs+1)*error<1/norm :
                    #     time_Lda.append(time.item())
                    #     Lda_error.append(test)
                    #     Lda_bound.append(bound)

                    L_lda = torch.mean(torch.linalg.norm(perturbed_score.prior_score(true_samples[:,None,:],time),2,dim=(1,2))) #nsamples,dim,1
                    true_diff_tall_score = perturbed_score.tall_posterior_score(true_samples[:,None,:], x_o_long,time,[diff_prec for _ in range(n_obs+1)],diff_prec_prior)
                    perturbed_score.error = error
                    est_diff_tall_score = perturbed_score.tall_posterior_score(true_samples[:,None,:], x_o_long,time,list_prec,diff_prec_prior+eps_lda*torch.eye(2),error_lda)
                    tall_error = torch.sum((true_diff_tall_score-est_diff_tall_score)**2,dim=2)**0.5 #(numsample,1)
                    # print("hey",torch.linalg.norm(true_diff_tall_score-est_diff_tall_score,2,dim=(1,2))-tall_error[:,0])
                    # print(est_diff_tall_score.shape, tall_error.shape,torch.linalg.norm(true_diff_tall_score-est_diff_tall_score,2,dim=(1,2)).shape)
                    tall_error = torch.linalg.norm(true_diff_tall_score-est_diff_tall_score,2,dim=(1,2))
                    error_bound = n_obs*(L_lda+error_lda)*norm
                    frac = (n_obs*eps_lda+(n_obs+1)*eps)*norm/(1-norm*(n_obs*eps_lda+(n_obs+1)*eps))
                    error_bound*=eps_lda+frac*(M_lda+eps_lda)#M_lda/(1-M_lda*eps_lda)
                    error_bound+=(n_obs+1)*(L+error)*norm*(eps+frac*(M+eps))#*M/(1-M*eps))
                    error_bound += norm*(n_obs*M_lda*error_lda+(n_obs+1)*M*error)
                    if n_obs*eps_lda+(n_obs+1)*eps<1/norm :#and eps_lda<1/M_lda and eps<1/M:
                        # print("true")
                        # tall_error_list.append(torch.log(torch.mean(tall_error)).item())
                        # error_bound_list.append(torch.log(error_bound).item())
                        tall_error_list.append(torch.mean(tall_error).item())
                        error_bound_list.append(error_bound.item())
                        time_score.append(time.item())
                # plt.subplot(2,2,i+1)
                # plt.plot(time_Lda,Lda_error, color="blue", label=r"$||\Lambda_t^{-1}-\tilde \Lambda_t^{-1}||$")
                # plt.plot(time_Lda,Lda_bound, color="red", label="analytical bound")
                # # plt.plot(torch.linspace(1e-3,1.0,500),tall_error_list, color="blue", label=r"$\mathcal{W}_2(p,\tilde p)$")
                # # plt.plot(torch.linspace(1e-3,1.0,500),error_bound_list, color="red", label="analytical bound")
                # plt.xlabel(r"$t$")
                # plt.legend()
                # plt.title(fr"$\epsilon=||\Sigma^{{-1}}_t-\tilde \Sigma_t^{{-1}}||_2={round(error.item(),3)}$")

                plt.subplot(2,2,k+1)
                plt.plot(time_score,tall_error_list, color=colors[i], label=fr"$\epsilon_{{\text{{DSM}}}}={round(error.item(),2)}$")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
                plt.plot(time_score,error_bound_list, ls=(0,(5,5)), color=colors[i])#, label="analytical bound")
                plt.xlabel(r"$t$", fontsize=11)
                if k%2==0:
                    plt.ylabel(r"$\log \sup_t E_{{\nu_t}}||\nabla \log \nu_t(\theta\mid x_{{1:n}})-s(\theta,x_{{1:n}},t)||$", fontsize=11)
                plt.legend()
                plt.title(rf"$n={n_obs+1}$ observations")
            # plt.title(fr"$\sup_t \sup_i E_{{\nu_t}}||\nabla \log p_t(\theta\mid x_i)-s(\theta,x_i,t)||\leq {round(error.item(),3)}$")
            # plt.subplot(222)
            # plt.scatter(list_wass,list_cov_emp, color="red", marker="o",label=r"$||\Sigma_{\text{emp}}-\tilde \Sigma||_2$")
            # # plt.plot(list_wass,list_cov_pop, color="orange",marker="o", label=r"$||\Sigma_{\text{true}}-\tilde \Sigma||_2^2$")
            # plt.scatter(list_wass,list_cov_bound, color="green",marker="o", label="theoretical bound")
            # plt.scatter(list_wass,list_cov_bound_prev, color="blue",marker="o", label="prev bound")
            # ymin, ymax = plt.ylim()
            # plt.vlines(list_wass, ymin, ymax, color="black", ls="dashed")
        plt.tight_layout()
        plt.show()
        print(h)

    if 0:#for the L2 norm WITH square
        # inference with several additional observations
        n_obs = cfg.nextra
        # extra_obs = prior_beta.sample((n_obs,))
        # beta_o_long_full = torch.cat((beta_o.unsqueeze(0),extra_obs),dim=0)
        # x_o_long_full = simulator1(beta_o_long_full)
        extra_obs = [x_o]
        for _ in range(n_obs):
            extra_obs.append(simulator1(beta_o))
        x_o_long_full = torch.stack(extra_obs)
        print("\nDimensions for ", n_obs+1, " observations:\n x : ",x_o_long_full.size())
        eps=0.1 #error of precision matrices
        eps_lda=0.01 #error of prior precision
        error_lda = 0.01 #error for prior score

        fig = plt.figure(figsize=(13.5,10))
        plt.suptitle(fr"$\epsilon_{{\lambda}}={eps_lda}, \ \ \epsilon=\sup_t \sup_i ||\Sigma^{{-1}}_{{t,i}}-\tilde \Sigma^{{-1}}_{{t,i}}||\leq {eps}, \ \ \epsilon_{{\text{{DSM,}},\lambda}}^2=\sup_t E_{{\nu_t}}[||\nabla \log p_t(\theta)-s(\theta,t)||^2]\leq {round(error_lda**2,3)}$")
        colors = ["red","blue","green","orange"]
        for k, n_obs in enumerate([2,5,7,cfg.nextra]):
            x_o_long = x_o_long_full[:n_obs+1,:]
            # beta_o_long = beta_o_long_full[:n_obs+1,:]
            for i, error in enumerate(error_list) :
                tall_error_list=[]
                error_bound_list=[]
                Lda_error = []
                Lda_bound = []
                time_Lda = []
                time_score = []
                for time in torch.linspace(0.2,1.0,100):
                    perturbed_score.error=0.0
                    alpha_t = perturbed_score.mean_t(time)**2
                    true_samples = torch.distributions.MultivariateNormal(alpha_t**0.5*mu_tall_post(x_o_long).squeeze(1), alpha_t*cov_tall_post(x_o_long)+(1-alpha_t)*torch.eye(2)).sample((num_samples,))
                    diff_prec = perturbed_score.diff_prec(prec_post,time) #true diffused post precisions
                    M = torch.linalg.norm(diff_prec,2) #all the cov are equal in our case
                    diff_prec_prior = perturbed_score.diff_prec(torch.linalg.inv(cov_prior),time) #true diffused prior precision
                    M_lda = torch.linalg.norm(diff_prec_prior,2)
                    # Lda = torch.linalg.norm(torch.linalg.inv(-n_obs*diff_prec_prior+(n_obs+1)*diff_prec),2)
                    Lda = torch.linalg.inv(-n_obs*diff_prec_prior+(n_obs+1)*diff_prec)
                    norm = torch.linalg.norm(Lda,2)
                    L=0
                    list_prec=[] #store covariance of **perturbed** indiv posterior
                    for j in range(n_obs+1):
                        norm_indiv_score = torch.linalg.norm(perturbed_score.single_posterior_score(true_samples[:,None,:],x_o_long[j,:],time),2, dim=(1,2))
                        list_prec.append(diff_prec+eps*torch.eye(2)) #Sigma tilde
                        if L<=torch.mean(norm_indiv_score**2).item():
                            L=torch.mean(norm_indiv_score**2).item()
                    tilde_Lda = torch.linalg.inv(-n_obs*(diff_prec_prior+eps_lda*torch.eye(2))+torch.stack(list_prec).sum(dim=0))
                    # test = torch.linalg.norm(Lda-tilde_Lda,2)
                    # bound=(n_obs*eps_lda+(n_obs+1)*error)*norm**2/(1-(n_obs*eps_lda+(n_obs+1)*error)*norm)
                    # if n_obs*eps_lda+(n_obs+1)*error<1/norm :
                    #     time_Lda.append(time.item())
                    #     Lda_error.append(test)
                    #     Lda_bound.append(bound)

                    L_lda = torch.mean(torch.linalg.norm(perturbed_score.prior_score(true_samples[:,None,:],time),2,dim=(1,2))**2) #nsamples,dim,1
                    true_diff_tall_score = perturbed_score.tall_posterior_score(true_samples[:,None,:], x_o_long,time,[diff_prec for _ in range(n_obs+1)],diff_prec_prior)
                    perturbed_score.error = error
                    est_diff_tall_score = perturbed_score.tall_posterior_score(true_samples[:,None,:], x_o_long,time,list_prec,diff_prec_prior+eps_lda*torch.eye(2),error_lda)
                    # tall_error = torch.sum((true_diff_tall_score-est_diff_tall_score)**2,dim=2) #(numsample,1)
                    tall_error=torch.linalg.norm(true_diff_tall_score-est_diff_tall_score, 2, dim=(1,2))**2
                    error_bound = n_obs*(L_lda**0.5+error_lda)*norm
                    frac = (n_obs*eps_lda+(n_obs+1)*eps)*norm/(1-norm*(n_obs*eps_lda+(n_obs+1)*eps))
                    error_bound*=eps_lda+frac*(M_lda+eps_lda)#*M_lda/(1-M_lda*eps_lda)
                    error_bound+=(n_obs+1)*(L**0.5+error)*norm*(eps+frac*(M+eps))#*M/(1-M*eps))
                    error_bound += norm*(n_obs*M_lda*error_lda+(n_obs+1)*M*error)
                    # print("1",n_obs*eps_lda+(n_obs+1)*eps<1/norm)
                    # print("2",eps_lda<1/M_lda)
                    if n_obs*eps_lda+(n_obs+1)*eps<1/norm :#and eps_lda<1/M_lda and eps<1/M:
                        # print("true")
                        # tall_error_list.append(torch.log(torch.mean(tall_error)).item())
                        # error_bound_list.append(torch.log(error_bound**2).item())
                        tall_error_list.append(torch.mean(tall_error).item())
                        error_bound_list.append((error_bound**2).item())
                        time_score.append(time.item())
                # plt.subplot(2,2,i+1)
                # plt.plot(time_Lda,Lda_error, color="blue", label=r"$||\Lambda_t^{-1}-\tilde \Lambda_t^{-1}||$")
                # plt.plot(time_Lda,Lda_bound, color="red", label="analytical bound")
                # # plt.plot(torch.linspace(1e-3,1.0,500),tall_error_list, color="blue", label=r"$\mathcal{W}_2(p,\tilde p)$")
                # # plt.plot(torch.linspace(1e-3,1.0,500),error_bound_list, color="red", label="analytical bound")
                # plt.xlabel(r"$t$")
                # plt.legend()
                # plt.title(fr"$\epsilon=||\Sigma^{{-1}}_t-\tilde \Sigma_t^{{-1}}||_2={round(error.item(),3)}$")

                plt.subplot(2,2,k+1)

                plt.plot(time_score,tall_error_list, color=colors[i], label=fr"$\epsilon_{{\text{{DSM}}}}={round(error.item(),2)}$")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
                plt.plot(time_score,error_bound_list, ls=(0,(5,5)), color=colors[i])#, label="analytical bound")
                plt.xlabel(r"$t$", fontsize=11)
                if k%2==0:
                    plt.ylabel(r"$\log \sup_t E_{{\nu_t}}[||\nabla \log \nu_t(\theta\mid x_{{1:n}})-s(\theta,x_{{1:n}},t)||^2]$", fontsize=11)
                plt.legend()
                plt.title(rf"$n={n_obs+1}$ observations")

            # plt.title(fr"$\sup_t \sup_i E_{{\nu_t}}||\nabla \log p_t(\theta\mid x_i)-s(\theta,x_i,t)||\leq {round(error.item(),3)}$")
            # plt.subplot(222)
            # plt.scatter(list_wass,list_cov_emp, color="red", marker="o",label=r"$||\Sigma_{\text{emp}}-\tilde \Sigma||_2$")
            # # plt.plot(list_wass,list_cov_pop, color="orange",marker="o", label=r"$||\Sigma_{\text{true}}-\tilde \Sigma||_2^2$")
            # plt.scatter(list_wass,list_cov_bound, color="green",marker="o", label="theoretical bound")
            # plt.scatter(list_wass,list_cov_bound_prev, color="blue",marker="o", label="prev bound")
            # ymin, ymax = plt.ylim()
            # plt.vlines(list_wass, ymin, ymax, color="black", ls="dashed")
        plt.tight_layout()
        plt.show()
        print(h)

    if 0:#for the L2 norm WITH square and estimated cov
            # inference with several additional observations
            n_obs = cfg.nextra
            extra_obs = prior_beta.sample((n_obs,))
            beta_o_long_full = torch.cat((beta_o.unsqueeze(0),extra_obs),dim=0)
            x_o_long_full = simulator1(beta_o_long_full)
            print("\nDimensions for ", n_obs+1, " observations: \n beta : ", beta_o_long_full.size(),
            "\n x : ",x_o_long_full.size())
            # eps=0.1 #error of precision matrices
            eps_lda=0.001 #error of prior precision
            error_lda = 0.01 #error for prior score

            fig = plt.figure(figsize=(13.5,10))
            colors = ["red","blue","green","orange"]
            for k,n_obs in enumerate([2,5,7,cfg.nextra]):
                print(k, n_obs)
                x_o_long = x_o_long_full[:n_obs+1,:]
                # beta_o_long = beta_o_long_full[:n_obs+1,:]
                for i, error in enumerate(error_list) :
                    tall_error_list=[]
                    error_bound_list=[]
                    time_score = []
                    perturbed_score.error=error
                    list_est_prec=[] #\tilde \Sigma at time 0
                    eps_max=0 #error in precision matrix at time 0 (same for all time t)
                    for xi in range(n_obs+1):
                        samples_beta = perturbed_score.sampling(num_samples,steps=1000, condition=x_o_long[xi,:][None,:])[:,0,:] #size (num_samples, 1, dim)
                        cov_est = torch.cov(samples_beta.T)
                        prec_est = torch.linalg.inv(cov_est)
                        list_est_prec.append(prec_est)
                        eps = torch.linalg.norm(prec_est-prec_post,2)
                        if eps_max<=eps:
                            eps_max=eps
                    for time in torch.tensor([0.1,0.5,0.9]):#torch.linspace(0.2,1.0,100):
                        perturbed_score.error=0.0
                        alpha_t = perturbed_score.mean_t(time)**2
                        true_samples = torch.distributions.MultivariateNormal(alpha_t**0.5*mu_tall_post(x_o_long).squeeze(1), alpha_t*cov_tall_post(x_o_long)+(1-alpha_t)*torch.eye(2)).sample((num_samples,))
                        diff_prec = perturbed_score.diff_prec(prec_post,time) #true diffused post precisions \Sigma_t
                        M = torch.linalg.norm(diff_prec,2) #all the cov are equal in our case
                        diff_prec_prior = perturbed_score.diff_prec(torch.linalg.inv(cov_prior),time) #true diffused prior precision
                        M_lda = torch.linalg.norm(diff_prec_prior,2)
                        # Lda = torch.linalg.norm(torch.linalg.inv(-n_obs*diff_prec_prior+(n_obs+1)*diff_prec),2)
                        Lda = torch.linalg.inv(-n_obs*diff_prec_prior+(n_obs+1)*diff_prec)
                        norm = torch.linalg.norm(Lda,2)
                        L=0
                        list_prec=[] #store covariance of **perturbed** indiv posterior
                        for j in range(n_obs+1):
                            norm_indiv_score = torch.linalg.norm(perturbed_score.single_posterior_score(true_samples[:,None,:],x_o_long[j,:],time),2, dim=(1,2))
                            list_prec.append(perturbed_score.diff_prec(list_est_prec[j],time)) #Sigma tilde
                            if L<=torch.mean(norm_indiv_score**2).item():
                                L=torch.mean(norm_indiv_score**2).item()

                        L_lda = torch.mean(torch.linalg.norm(perturbed_score.prior_score(true_samples[:,None,:],time),2,dim=(1,2))**2) #nsamples,dim,1
                        true_diff_tall_score = perturbed_score.tall_posterior_score(true_samples[:,None,:], x_o_long,time,[diff_prec for _ in range(n_obs+1)],diff_prec_prior)
                        perturbed_score.error = error
                        est_diff_tall_score = perturbed_score.tall_posterior_score(true_samples[:,None,:], x_o_long,time,list_prec,diff_prec_prior+eps_lda*torch.eye(2),error_lda)
                        # tall_error = torch.sum((true_diff_tall_score-est_diff_tall_score)**2,dim=2) #(numsample,1)
                        tall_error=torch.linalg.norm(true_diff_tall_score-est_diff_tall_score, 2, dim=(1,2))**2
                        error_bound = n_obs*(L_lda**0.5+error_lda)*norm
                        frac = (n_obs*eps_lda+(n_obs+1)*eps_max)*norm/(1-norm*(n_obs*eps_lda+(n_obs+1)*eps_max))
                        error_bound*=eps_lda+frac*M_lda/(1-M_lda*eps_lda)
                        error_bound+=(n_obs+1)*(L**0.5+error)*norm*(eps_max+frac*M/(1-M*eps_max))
                        error_bound += norm*(n_obs*M_lda*error_lda+(n_obs+1)*M*error)
                        if n_obs*eps_lda+(n_obs+1)*eps_max<1/norm :#and eps_lda<1/M_lda and eps_max<1/M:
                            # print("true")
                            tall_error_list.append(torch.log(torch.mean(tall_error)).item())
                            error_bound_list.append(torch.log(error_bound**2).item())
                            time_score.append(time.item())
                    # plt.subplot(2,2,i+1)
                    # plt.plot(time_Lda,Lda_error, color="blue", label=r"$||\Lambda_t^{-1}-\tilde \Lambda_t^{-1}||$")
                    # plt.plot(time_Lda,Lda_bound, color="red", label="analytical bound")
                    # # plt.plot(torch.linspace(1e-3,1.0,500),tall_error_list, color="blue", label=r"$\mathcal{W}_2(p,\tilde p)$")
                    # # plt.plot(torch.linspace(1e-3,1.0,500),error_bound_list, color="red", label="analytical bound")
                    # plt.xlabel(r"$t$")
                    # plt.legend()
                    # plt.title(fr"$\epsilon=||\Sigma^{{-1}}_t-\tilde \Sigma_t^{{-1}}||_2={round(error.item(),3)}$")

                    plt.subplot(2,2,k+1)
                    plt.plot(time_score,tall_error_list, color=colors[i], label=fr"$\epsilon_{{\text{{DSM}}}}={round(error.item(),3)}$")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
                    plt.plot(time_score,error_bound_list, ls=(0,(5,5)), color=colors[i])#, label="analytical bound")
                    plt.xlabel(r"$t$", fontsize=11)
                    if k%2==0:
                        plt.ylabel(r"$\log \sup_t E_{{\nu_t}}[||\nabla \log \nu_t(\theta\mid x_{{1:n}})-s(\theta,x_{{1:n}},t)||^2]$", fontsize=11)
                    plt.legend()
                    plt.title(rf"$n={n_obs+1}$ observations, $\epsilon={round(eps_max.item(),3)}$")
                # plt.title(fr"$\sup_t \sup_i E_{{\nu_t}}||\nabla \log p_t(\theta\mid x_i)-s(\theta,x_i,t)||\leq {round(error.item(),3)}$")
                # plt.subplot(222)
                # plt.scatter(list_wass,list_cov_emp, color="red", marker="o",label=r"$||\Sigma_{\text{emp}}-\tilde \Sigma||_2$")
                # # plt.plot(list_wass,list_cov_pop, color="orange",marker="o", label=r"$||\Sigma_{\text{true}}-\tilde \Sigma||_2^2$")
                # plt.scatter(list_wass,list_cov_bound, color="green",marker="o", label="theoretical bound")
                # plt.scatter(list_wass,list_cov_bound_prev, color="blue",marker="o", label="prev bound")
                # ymin, ymax = plt.ylim()
                # plt.vlines(list_wass, ymin, ymax, color="black", ls="dashed")
            plt.suptitle(fr"$\epsilon_{{\lambda}}={eps_lda}, \ \ \epsilon_{{\text{{DSM,}},\lambda}}^2=\sup_t E_{{\nu_t}}[||\nabla \log p_t(\theta)-s(\theta,t)||^2]\leq {round(error_lda**2,4)}$")
            plt.tight_layout()
            plt.show()

    if 0:#for the L2 norm WITH square and estimated cov
        # inference with several additional observations
        n_obs = cfg.nextra
        extra_obs = prior_beta.sample((n_obs,))
        beta_o_long_full = torch.cat((beta_o.unsqueeze(0),extra_obs),dim=0)
        x_o_long_full = simulator1(beta_o_long_full)
        print("\nDimensions for ", n_obs+1, " observations: \n beta : ", beta_o_long_full.size(),
        "\n x : ",x_o_long_full.size())
        # eps=0.1 #error of precision matrices
        eps_lda=0.001 #error of prior precision
        error_lda = 0.01 #error for prior score

        fig1,ax1 = plt.subplots(2,2,figsize=(13.5,10))
        ax1 = ax1.flatten()
        colors = ["red","blue","green","orange"]
        for k,n_obs in enumerate([2,5,7,cfg.nextra]):
            print(k, n_obs)
            x_o_long = x_o_long_full[:n_obs+1,:]
            cov_bound_tensor = torch.zeros((n_obs+1,len(error_list),5)) #nextra+1,4,5
            fig2, ax2 = plt.subplots(k+1,3,figsize=(13.5,10))
            ax2 = ax2.flatten()
            fig3, ax3 = plt.subplots(k+1,3,figsize=(13.5,10))
            ax3 = ax3.flatten()

            for i, error in enumerate(error_list) :
                anal_tall_error_list=[]
                # check_list=[]
                tall_error_list=[]
                error_bound_list=[]
                # Lda_error = []
                # Lda_bound = []
                # time_Lda = []
                time_score = []
                perturbed_score.error=error
                list_est_prec=[] #\tilde \Sigma at time 0
                eps_max=0 #error in precision matrix at time 0 (same for all time t)
                for xi in range(n_obs+1):
                    samples_beta = perturbed_score.sampling(num_samples,steps=1000, condition=x_o_long[xi,:][None,:])[:,0,:] #size (num_samples, 1, dim)
                    cov_est = torch.cov(samples_beta.T)
                    prec_est = torch.linalg.inv(cov_est)
                    l2_norm_cov_est = torch.linalg.norm(cov_est,2) #l2 norm of \tilde \Sigma
                    list_est_prec.append(prec_est)
                    wass = wasserstein_dist(mu_post(x_o_long_full[xi]),torch.mean(samples_beta),cov_post,cov_est)**0.5
                    cov_bound_prev = 2**1.5*torch.min(l2_norm_cov_post,l2_norm_cov_est)**0.5*wass+(1+3*2**0.5)*wass**2
                    cov_bound = 2**1.5*torch.max(l2_norm_cov_post,l2_norm_cov_est)**0.5*wass+(1+2**0.5)*wass**2
                    diff_cov = torch.mean((cov_est-cov_post)**2)**0.5 #distance between \tilde \Sigma and \Sigma
                    diff_prec = torch.mean((prec_est-torch.linalg.inv(cov_post))**2)**0.5 # distance between \tilde \Sigma^(-1) and \Sigma^(-1)
                    prec_bound = diff_cov*l2_norm_prec_post**2/(1-diff_cov*l2_norm_prec_post)
                    check = (diff_cov<1.0/l2_norm_prec_post)
                    cov_bound_tensor[xi,i,0] = diff_cov.item() #diff between cov
                    cov_bound_tensor[xi,i,1] = cov_bound.item() #anal diff between cov
                    cov_bound_tensor[xi,i,2] = cov_bound_prev.item() #false diff between cov
                    cov_bound_tensor[xi,i,3] = diff_prec.item() #diff between prec
                    cov_bound_tensor[xi,i,4] = prec_bound.item() #anal diff between prec
                    eps = torch.linalg.norm(prec_est-prec_post,2)
                    if eps_max<=eps:
                        eps_max=eps

                for time in torch.linspace(0.2,1.0,100):
                    perturbed_score.error=0.0
                    alpha_t = perturbed_score.mean_t(time)**2
                    true_samples = torch.distributions.MultivariateNormal(alpha_t**0.5*mu_tall_post(x_o_long).squeeze(1), alpha_t*cov_tall_post(x_o_long)+(1-alpha_t)*torch.eye(2)).sample((num_samples,))
                    diff_prec = perturbed_score.diff_prec(prec_post,time) #true diffused post precisions \Sigma_t
                    M = torch.linalg.norm(diff_prec,2) #all the cov are equal in our case
                    diff_prec_prior = perturbed_score.diff_prec(torch.linalg.inv(cov_prior),time) #true diffused prior precision
                    M_lda = torch.linalg.norm(diff_prec_prior,2)
                    # Lda = torch.linalg.norm(torch.linalg.inv(-n_obs*diff_prec_prior+(n_obs+1)*diff_prec),2)
                    Lda = torch.linalg.inv(-n_obs*diff_prec_prior+(n_obs+1)*diff_prec)
                    norm = torch.linalg.norm(Lda,2)
                    L=0
                    list_prec=[] #store covariance of diffused **perturbed** indiv posterior
                    for j in range(n_obs+1):
                        norm_indiv_score = torch.linalg.norm(perturbed_score.single_posterior_score(true_samples[:,None,:],x_o_long[j,:],time),2, dim=(1,2))
                        list_prec.append(perturbed_score.diff_prec(list_est_prec[j],time)) #estimated diffused post precision Sigma tilde
                        if L<=torch.mean(norm_indiv_score**2).item():
                            L=torch.mean(norm_indiv_score**2).item()
                    # tilde_Lda = torch.linalg.inv(-n_obs*(diff_prec_prior+eps_lda*torch.eye(2))+torch.stack(list_prec).sum(dim=0))
                    # test = torch.linalg.norm(Lda-tilde_Lda,2)
                    # bound=(n_obs*eps_lda+(n_obs+1)*error)*norm**2/(1-(n_obs*eps_lda+(n_obs+1)*error)*norm)
                    # if n_obs*eps_lda+(n_obs+1)*error<1/norm :
                    #     time_Lda.append(time.item())
                    #     Lda_error.append(test)
                    #     Lda_bound.append(bound)

                    L_lda = torch.mean(torch.linalg.norm(perturbed_score.prior_score(true_samples[:,None,:],time),2,dim=(1,2))**2) #nsamples,dim,1
                    true_diff_tall_score = perturbed_score.tall_posterior_score(true_samples[:,None,:], x_o_long,time,[diff_prec for _ in range(n_obs+1)],diff_prec_prior)
                    perturbed_score.error = error
                    est_diff_tall_score = perturbed_score.tall_posterior_score(true_samples[:,None,:], x_o_long,time,list_prec,diff_prec_prior+eps_lda*torch.eye(2),error_lda)
                    # tall_error = torch.sum((true_diff_tall_score-est_diff_tall_score)**2,dim=2) #(numsample,1)
                    anal_diff_tall_score = perturbed_score.true_tall_posterior_score(true_samples[:,None,:],x_o_long,time)
                    tall_error=torch.linalg.norm(true_diff_tall_score-est_diff_tall_score, 2, dim=(1,2))**2
                    anal_tall_error = torch.linalg.norm(anal_diff_tall_score-est_diff_tall_score,2, dim=(1,2))**2
                    # check = torch.linalg.norm(anal_diff_tall_score-true_diff_tall_score,2, dim=(1,2))**2
                    error_bound = n_obs*(L_lda**0.5+error_lda)*norm
                    frac = (n_obs*eps_lda+(n_obs+1)*eps_max)*norm/(1-norm*(n_obs*eps_lda+(n_obs+1)*eps_max))
                    error_bound*=eps_lda+frac*M_lda/(1-M_lda*eps_lda)
                    error_bound+=(n_obs+1)*(L**0.5+error)*norm*(eps_max+frac*M/(1-M*eps_max))
                    error_bound += norm*(n_obs*M_lda*error_lda+(n_obs+1)*M*error)
                    if n_obs*eps_lda+(n_obs+1)*eps_max<1/norm and eps_lda<1/M_lda and eps_max<1/M:
                        # print("true")
                        # check_list.append(torch.log(torch.mean(check)).item())
                        tall_error_list.append(torch.log(torch.mean(tall_error)).item())
                        anal_tall_error_list.append(torch.log(torch.mean(anal_tall_error)).item())
                        error_bound_list.append(torch.log(error_bound**2).item())
                        time_score.append(time.item())
                # plt.subplot(2,2,i+1)
                # plt.plot(time_Lda,Lda_error, color="blue", label=r"$||\Lambda_t^{-1}-\tilde \Lambda_t^{-1}||$")
                # plt.plot(time_Lda,Lda_bound, color="red", label="analytical bound")
                # # plt.plot(torch.linspace(1e-3,1.0,500),tall_error_list, color="blue", label=r"$\mathcal{W}_2(p,\tilde p)$")
                # # plt.plot(torch.linspace(1e-3,1.0,500),error_bound_list, color="red", label="analytical bound")
                # plt.xlabel(r"$t$")
                # plt.legend()
                # plt.title(fr"$\epsilon=||\Sigma^{{-1}}_t-\tilde \Sigma_t^{{-1}}||_2={round(error.item(),3)}$")

                ax1[k].plot(time_score,tall_error_list, color=colors[i], label=fr"$\epsilon_{{\text{{DSM}}}}={round(error.item(),3)}$")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
                ax1[k].plot(time_score,error_bound_list, ls=(0,(5,5)), color=colors[i])#, label="analytical bound")
                # ax1[k].plot(time_score,anal_tall_error_list, ls="dotted", color=colors[i])#, label="analytical bound")
                # ax1[k].plot(time_score,check_list, ls="dashdot", color=colors[i])#, label="analytical bound")
                ax1[k].set_xlabel(r"$t$", fontsize=11)
                if k%2==0:
                    ax1[k].set_ylabel(r"$\log \sup_t E_{{\nu_t}}[||\nabla \log \nu_t(\theta\mid x_{{1:n}})-s(\theta,x_{{1:n}},t)||^2]$", fontsize=11)
                ax1[k].legend()
                ax1[k].set_title(rf"$n={n_obs+1}$ observations, $\epsilon={round(eps_max.item(),3)}$")

            # for j in range(n_obs+1):
            #     ax2[j].scatter(error_list,cov_bound_tensor[j,:,0],color="blue",label="empirical bound")
            #     ax2[j].scatter(error_list,cov_bound_tensor[j,:,1], color="red", label="analytical bound")
            #     ax2[j].scatter(error_list,cov_bound_tensor[j,:,2], color="green", label="previous bound" )
            #     # ymin, ymax = ax2[j].get_ylim()
            #     ax2[j].set_xlabel(r"$\epsilon_{\text{DSM}}$")
            #     ax2[j].set_ylabel(r"$||\Sigma_{\text{emp}}-\tilde \Sigma||_2$")

            #     ax3[j].scatter(cov_bound_tensor[j,:,0], cov_bound_tensor[j,:,3],color="blue",label="empirical bound")
            #     ax3[j].scatter(cov_bound_tensor[j,:,0],cov_bound_tensor[j,:,4], color="red", label="analytical bound")
            #     ax3[j].set_xlabel(r"$||\Sigma_{\text{emp}}-\tilde \Sigma||_2$")
            #     ax3[j].set_ylabel(r"$||\Sigma^{-1}_{\text{emp}}-\tilde \Sigma^{-1}||_2$")
            #     # ymin3, ymax3 = ax3[j].get_ylim()
            #     # print(ymin3, ymax3)
            #     for err in error_list:
            #         ax2[j].axvline(err, color="black", ls="dashed")
            #     for err in cov_bound_tensor[j,:,0]:
            #         ax3[j].axvline(err, color="black", ls="dashed")
            #     ax2[j].legend()
            #     ax3[j].legend()
            #     ax2[j].set_title(rf"$p(\theta\mid x_{{{j}}})$")
            #     ax3[j].set_title(rf"$p(\theta\mid x_{{{j}}})$")
            # fig2.tight_layout()
            # fig3.tight_layout()
            # fig2.savefig(f"eurips/cov_bound_{n_obs+1}_observations.png")
            # fig3.savefig(f"eurips/prec_bound_{n_obs+1}_observations.png")

            # plt.title(fr"$\sup_t \sup_i E_{{\nu_t}}||\nabla \log p_t(\theta\mid x_i)-s(\theta,x_i,t)||\leq {round(error.item(),3)}$")
            # plt.subplot(222)
            # plt.scatter(list_wass,list_cov_emp, color="red", marker="o",label=r"$||\Sigma_{\text{emp}}-\tilde \Sigma||_2$")
            # # plt.plot(list_wass,list_cov_pop, color="orange",marker="o", label=r"$||\Sigma_{\text{true}}-\tilde \Sigma||_2^2$")
            # plt.scatter(list_wass,list_cov_bound, color="green",marker="o", label="theoretical bound")
            # plt.scatter(list_wass,list_cov_bound_prev, color="blue",marker="o", label="prev bound")
            # ymin, ymax = plt.ylim()
            # plt.vlines(list_wass, ymin, ymax, color="black", ls="dashed")
        fig1.suptitle(fr"$\epsilon_{{\lambda}}={eps_lda}, \ \ \epsilon_{{\text{{DSM,}},\lambda}}^2=\sup_t E_{{\nu_t}}[||\nabla \log p_t(\theta)-s(\theta,t)||^2]\leq {round(error_lda**2,4)}$")
        fig1.tight_layout()
        plt.show()

    if 0:#first motivation figure
        import matplotlib.colors as mcolors
        import numpy as np

        np.random.seed(cfg.seed)
        colors = ["#caf0f8","#90e0ef","#00b4d8", "#0077b6","#03045e"]  # e.g., orange → green → blue
        colors = ["#00b4d8", "#0077b6","#03045e"]  # e.g., orange → green → blue

        # Create a custom colormap
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)
        error_list = [0.01, 1]

        n_obs = cfg.nextra
        extra_obs = [x_o]
        for _ in range(n_obs):
            extra_obs.append(simulator1(beta_o))
        x_o_long = torch.stack(extra_obs)
        print("\nDimensions for ", n_obs+1, " observations: \n x : ",x_o_long.size())

        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(8,6))

        colors = ["red","blue","green","orange"]
        true_samples = torch.distributions.MultivariateNormal(mu_post(x_o).squeeze(2).squeeze(0), cov_post).sample((num_samples,))
        true_samples_tall = torch.distributions.MultivariateNormal(mu_tall_post(x_o_long).squeeze(1), cov_tall_post(x_o_long)).sample((num_samples,))
        rv = torch.distributions.MultivariateNormal(mu_post(x_o).squeeze(2).squeeze(0), cov_post)
        rv_tall = torch.distributions.MultivariateNormal(mu_tall_post(x_o_long).squeeze(1), cov_tall_post(x_o_long))
        x = np.linspace(-5,5,200)
        y = np.linspace(-5,5,200)
        xx, yy = np.mgrid[-2:5:.01, -1:5:.01]
        xx, yy = np.meshgrid(x,y)
        pos = np.dstack((xx, yy))
        x_tall = np.linspace(-3,3,200)
        y_tall = np.linspace(-3,3,200)
        xx_tall, yy_tall = np.meshgrid(x_tall,y_tall)
        pos_tall = np.dstack((xx_tall, yy_tall))
        dico={}
        dico["observations"]=x_o_long
        dico["nb_obs"]=1+n_obs
        dico["cov_post"]=cov_post
        dico["cov_prior"]=cov_prior
        for i, error in enumerate(error_list) :
            perturbed_score.error=error
            samples_beta = perturbed_score.sampling(num_samples,steps=500, condition=x_o)[:,0,:] #size (num_samples, 1, dim)
            if i==1 :
                perturbed_score.error=error+0.9
            samples_beta_tall = perturbed_score.sampling(num_samples,steps=500, condition=x_o_long)[:,0,:] #size (num_samples, 1, dim)
            cov_est_indiv = torch.cov(samples_beta.T)
            cov_est_tall = torch.cov(samples_beta_tall.T)

            wass_indiv = wasserstein_dist(torch.mean(samples_beta, dim=0), mu_post(x_o)[0,:,0], cov_est_indiv, cov_post)**0.5
            wass_tall = wasserstein_dist(torch.mean(samples_beta_tall, dim=0), mu_tall_post(x_o_long).squeeze(1), cov_est_tall, cov_tall_post(x_o_long))**0.5
            # sns.kdeplot(x=samples_beta[:,0], y=samples_beta[:,1],fill=True,ax=axs[i,0], label="est")
            # sns.kdeplot(x=samples_beta_tall[:,0], y=samples_beta_tall[:,1],fill=True, ax=axs[i,1], label="true")
            # sns.kdeplot(x=true_samples[:,0], y=true_samples[:,1],fill=True,ax=axs[0,i], cmap=cmap)
            # sns.kdeplot(x=true_samples_tall[:,0], y=true_samples_tall[:,1],fill=True, ax=axs[1,i],cmap=cmap)
            if i==0:
                dico["small_score_error"]={"wass_indiv":wass_indiv.item(),
                                       "wass_tall":wass_tall.item(),
                                       "indiv_samples":samples_beta,
                                       "tall_samples":samples_beta_tall}
            else:
                dico["tall_score_error"]={"wass_indiv":wass_indiv.item(),
                                       "wass_tall":wass_tall.item(),
                                       "indiv_samples":samples_beta,
                                       "tall_samples":samples_beta_tall}

            axs[0,i].scatter(x=samples_beta[:100,0], y=samples_beta[:100,1],s=10, c="#c1121f", alpha=0.8,label=rf"$\mathcal{{W}}_2(p,\tilde p)= {round(wass_indiv.item(),3)}$")
            axs[1,i].scatter(x=samples_beta_tall[-40:,0], y=samples_beta_tall[-40:,1],s=10, c="#c1121f", alpha=0.8,label=rf"$\mathcal{{W}}_2(p,\tilde p)= {round(wass_tall.item(),3)}$")
            axs[0,i].contour(xx, yy, rv.log_prob(torch.tensor(pos, dtype=torch.float32)).exp().numpy(), levels=4, cmap=cmap,linewidths=1)
            axs[1,i].contour(xx_tall, yy_tall, rv_tall.log_prob(torch.tensor(pos_tall, dtype=torch.float32)).exp().numpy(), levels=4, cmap=cmap, linewidths=1)

            # axs[0,i].legend()
            # axs[1,i].legend()
            axs[0, i].text(0.50, 0.12, fr"$\mathcal{{W}}_2(p, \tilde{{p}}$)={round(wass_indiv.item(),3)}", transform=axs[0, i].transAxes, fontsize=10, verticalalignment='top', horizontalalignment='center')
            axs[1, i].text(0.50, 0.12, fr"$\mathcal{{W}}_2(p, \tilde{{p}}$)={round(wass_tall.item(),3)}", transform=axs[1, i].transAxes, fontsize=10, verticalalignment='top', horizontalalignment='center')

        # axs[0,0].set_title(r"$p(\theta\mid x_0)$")
        # axs[0,1].set_title(fr"$p(\theta\mid x_0,\ldots,x_{{{n_obs}}})$")
        axs[1,0].set_xlabel("")
        axs[1,1].set_xlabel("")
        axs[1,1].set_ylim([-6,3])
        axs[0,0].set_ylabel(r"$p(\theta\mid x_0)$")
        axs[1,0].set_ylabel(fr"$p(\theta\mid x_0,\ldots,x_{{{n_obs}}})$")
        axs[0,0].set_title(r"$\epsilon_\text{DSM}<\epsilon_c$")
        axs[0,1].set_title(r"$\epsilon_\text{DSM}\geq\epsilon_c$")
        plt.show() #onutilise seed 42 en haut et seed 54 qd on lance
        logger.log_artifacts(dico, artifact_name=f"{cfg.nextra+1}_obs_seed_{cfg.seed}_data",
                            artifact_type='pickle')
        print(h)

    if 0:#first motivation figure with known data
        import matplotlib.colors as mcolors
        import numpy as np
        import pickle
        filename = 'logs/eurips/11_obs_seed_42_data_1'
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        np.random.seed(cfg.seed)
        colors = ["#caf0f8","#90e0ef","#00b4d8", "#0077b6","#03045e"]  # e.g., orange → green → blue
        colors = ["#00b4d8", "#0077b6","#03045e"]  # e.g., orange → green → blue

        # Create a custom colormap
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)
        error_list = [0.01, 1]

        #change
        print(data.keys())
        n_obs = data["nb_obs"]-1
        x_o_long = data["observations"]
        x_o = x_o_long[0]
        cov_post_tmp = data["cov_post"]
        cov_prior_tmp = data["cov_prior"]

        print(x_o)
        print("\nDimensions for ", n_obs+1, " observations: \n x : ",x_o_long.size())

        #change
        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(8,6))

        colors = ["red","blue","green","orange"]
        # true_samples = torch.distributions.MultivariateNormal(mu_post(x_o).squeeze(2).squeeze(0), cov_post_tmp).sample((num_samples,))
        # true_samples_tall = torch.distributions.MultivariateNormal(mu_tall_post(x_o_long).squeeze(1), cov_tall_post(x_o_long)).sample((num_samples,))
        rv = torch.distributions.MultivariateNormal(mu_post(x_o,cov_post_tmp,cov_prior_tmp).squeeze(2).squeeze(0), cov_post_tmp)
        rv_tall = torch.distributions.MultivariateNormal(mu_tall_post(x_o_long,cov_prior_tmp).squeeze(1), cov_tall_post(x_o_long,cov_prior_tmp))
        x = np.linspace(-5,5,200)
        y = np.linspace(-5,5,200)
        xx, yy = np.mgrid[-2:5:.01, -1:5:.01]
        xx, yy = np.meshgrid(x,y)
        pos = np.dstack((xx, yy))
        x_tall = np.linspace(-3,3,200)
        y_tall = np.linspace(-3,3,200)
        xx_tall, yy_tall = np.meshgrid(x_tall,y_tall)
        pos_tall = np.dstack((xx_tall, yy_tall))

        p_indiv_small = data["small_score_error"]["indiv_samples"]
        p_tall_small = data["small_score_error"]["tall_samples"]
        p_indiv_tall = data["tall_score_error"]["indiv_samples"]
        p_tall_tall = data["tall_score_error"]["tall_samples"]
        wass_indiv_small = data["small_score_error"]["wass_indiv"]
        wass_tall_small = data["small_score_error"]["wass_tall"]
        wass_indiv_big = data["tall_score_error"]["wass_indiv"]
        wass_tall_big = data["tall_score_error"]["wass_tall"]

        axs[0,0].scatter(x=p_indiv_small[:100,0], y=p_indiv_small[:100,1],s=10, c="#c1121f", alpha=0.8)
        axs[1,0].scatter(x=p_tall_small[-40:,0], y=p_tall_small[-40:,1],s=10, c="#c1121f", alpha=0.8)
        axs[0,0].contour(xx, yy, rv.log_prob(torch.tensor(pos, dtype=torch.float32)).exp().numpy(), levels=4, cmap=cmap,linewidths=1)
        axs[1,0].contour(xx_tall, yy_tall, rv_tall.log_prob(torch.tensor(pos_tall, dtype=torch.float32)).exp().numpy(), levels=4, cmap=cmap, linewidths=1)
        # axs[0, 0].text(0.50, 0.12, fr"$\mathcal{{W}}_2(p, \tilde{{p}}$)={round(wass_indiv_small,3)}", transform=axs[0, 0].transAxes, fontsize=10, verticalalignment='top', horizontalalignment='center')
        # axs[1, 0].text(0.50, 0.12, fr"$\mathcal{{W}}_2(p, \tilde{{p}}$)={round(wass_tall_small,3)}", transform=axs[1, 0].transAxes, fontsize=10, verticalalignment='top', horizontalalignment='center')

        axs[0,1].scatter(x=p_indiv_tall[:100,0], y=p_indiv_tall[:100,1],s=10, c="#c1121f", alpha=0.8)
        axs[1,1].scatter(x=p_tall_tall[-40:,0], y=p_tall_tall[-40:,1],s=10, c="#c1121f", alpha=0.8)
        axs[0,1].contour(xx, yy, rv.log_prob(torch.tensor(pos, dtype=torch.float32)).exp().numpy(), levels=4, cmap=cmap,linewidths=1)
        axs[1,1].contour(xx_tall, yy_tall, rv_tall.log_prob(torch.tensor(pos_tall, dtype=torch.float32)).exp().numpy(), levels=4, cmap=cmap, linewidths=1)
        # axs[0, 1].text(0.50, 0.12, fr"$\mathcal{{W}}_2(p, \tilde{{p}}$)={round(wass_indiv_big,3)}", transform=axs[0, 1].transAxes, fontsize=10, verticalalignment='top', horizontalalignment='center')
        # axs[1, 1].text(0.50, 0.12, fr"$\mathcal{{W}}_2(p, \tilde{{p}}$)={round(wass_tall_big,3)}", transform=axs[1, 1].transAxes, fontsize=10, verticalalignment='top', horizontalalignment='center')

        # axs[0,0].set_title(r"$p(\theta\mid x_0)$")
        # axs[0,1].set_title(fr"$p(\theta\mid x_0,\ldots,x_{{{n_obs}}})$")
        axs[1,0].set_xlabel("")
        axs[1,1].set_xlabel("")
        axs[1,1].set_ylim([-4,3])
        # axs[1,1].set_xlim([-4,4])
        axs[0,0].set_ylabel(r"$p(\theta\mid x_0)$")
        axs[1,0].set_ylabel(fr"$p(\theta\mid x_0,\ldots,x_{{{n_obs}}})$")
        axs[0,0].set_title(r"$\text{small} \ \epsilon_\text{DSM}^2$")
        axs[0,1].set_title(r"$\text{large} \ \epsilon_\text{DSM}^2$")

        axs[0,0].grid(which='major', alpha=0.5)
        axs[0,1].grid(which='major', alpha=0.5)
        axs[1,0].grid(which='major', alpha=0.5)
        axs[1,1].grid(which='major', alpha=0.5)
        plt.tight_layout()
        plt.show() #onutilise seed 42 en haut et seed 54 qd on lance

        print(h)

    if 0:#dependence in nb of obs with bound on epsilon
        dico={}
        n_obs = cfg.nextra
        list_nobs = torch.arange(0,n_obs,2)
        dico["list_nobs"]=list_nobs+1
        extra_obs = [x_o]
        for _ in range(n_obs):
            extra_obs.append(simulator1(beta_o))
        x_o_long_full = torch.stack(extra_obs)
        print("\nDimensions for ", n_obs+1, " observations: \n x : ",x_o_long_full.size())
        eps_lda=0.0#01 #error of prior precision
        error_lda = 0.0#1 #score error for prior
        error = 0.001**0.5 #score error on posterior
        fig4,ax4 = plt.subplots(figsize=(6.5,4.5))
        dico["epsilon_dsm_2"]=error**2
        colors = ["red","blue","green","orange"]
        list_prec_tmp=[] #store covariance of indiv posterior at time 0
        eps_max_list = []
        list_wass_max = []
        m = (l2_norm_cov_post**0.5).item()
        delta = (2*(2*m+1.0/m/l2_norm_prec_post)**2+4*(2**0.5+1)/l2_norm_prec_post).item()
        perturbed_score.error=error
        i=0
        for xi in range(n_obs+1):
            samples_beta = perturbed_score.sampling(num_samples,steps=200, condition=x_o_long_full[xi,:][None,:])[:,0,:] #size (num_samples, 1, dim)
            cov_est = torch.cov(samples_beta.T)
            diff_cov = torch.linalg.norm(cov_est-cov_post,2)
            prec_est = torch.linalg.inv(cov_est)
            l2_norm_cov_est = torch.linalg.norm(cov_est,2) #l2 norm of \tilde \Sigma
            wass = wasserstein_dist(mu_post(x_o)[0,:,0],torch.mean(samples_beta,dim=0),cov_post,cov_est)**0.5
            eps = torch.linalg.norm(prec_est-prec_post,2)
            list_prec_tmp.append(prec_est)
            eps_max_list.append(eps.item())
            check = diff_cov<1.0/l2_norm_prec_post
            print("valid assumption", check)
            if wass.item()<m/2**0.5 and wass.item()<(delta**0.5-2**0.5*(2*m+1.0/m/l2_norm_prec_post.item()))/2/(2**0.5+1):

                list_wass_max.append(wass.item())
            i+=1

        eps = torch.max(torch.tensor(eps_max_list)) #take the max prec error on all
        eta = torch.max(torch.tensor(list_wass_max)) #take the max wass error that verifies the assumption
        index_eps= eps_max_list.index(eps)
        cov_bound_new = 2**1.5*eta*m+eta**2*(1+2**0.5)/(1-2**0.5*eta/m)
        prec_bound = cov_bound_new*l2_norm_prec_post**2/(1-cov_bound_new*l2_norm_prec_post)
        print(eps,prec_bound)
        print([eps<prec_bound for eps in eps_max_list])
        dico["epsilon"]=eps
        ax4.set_title(fr"$\epsilon_{{\lambda}}={eps_lda}, \ \ \epsilon\leq {round(eps.item(),3)}, \ \ \epsilon_{{\text{{DSM}},\lambda}}^2\leq {round(error_lda**2,3)} \ \ \epsilon_\text{{DSM}}^2\leq{round(error**2,3)}$")
        for j, time in enumerate(torch.tensor([0.1,0.5,0.9])):
            dico_tmp={}
            tall_error_list_sq = []
            tall_error_list_sq_1 = []
            error_bound_list_sq=[]
            error_bound_list=[]
            time_score = []
            for k, n_obs in enumerate(list_nobs):
                x_o_long = x_o_long_full[:n_obs+1,:]
                perturbed_score.error=0.0
                alpha_t = perturbed_score.mean_t(time)**2
                true_samples = torch.distributions.MultivariateNormal(alpha_t**0.5*mu_tall_post(x_o_long).squeeze(1), alpha_t*cov_tall_post(x_o_long)+(1-alpha_t)*torch.eye(2)).sample((num_samples,))
                diff_prec = perturbed_score.diff_prec(prec_post,time) #true diffused post precisions
                M = torch.linalg.norm(diff_prec,2) #all the cov are equal in our case
                diff_prec_prior = perturbed_score.diff_prec(torch.linalg.inv(cov_prior),time) #true diffused prior precision
                M_lda = torch.linalg.norm(diff_prec_prior,2)
                Lda = torch.linalg.inv(-n_obs*diff_prec_prior+(n_obs+1)*diff_prec)
                norm = torch.linalg.norm(Lda,2)
                if n_obs*eps_lda+(n_obs+1)*eps<1/norm :#and eps_lda<1/M_lda and eps<1/M:
                    L_list = []
                    list_prec=[] #store covariance of with eps **fixed**
                    list_prec_1=[] #store **estimated** covariance of indiv posterior
                    # print("hey", perturbed_score.error)
                    for i in range(n_obs+1):
                        norm_indiv_score = torch.linalg.norm(perturbed_score.single_posterior_score(true_samples[:,None,:],x_o_long[i,:],time),2, dim=(1,2))
                        list_prec.append(diff_prec+eps*torch.eye(2)) #Sigma tilde
                        list_prec_1.append(perturbed_score.diff_prec(list_prec_tmp[i],time)) #with estimated cov

                        L_list.append(torch.mean(norm_indiv_score**2).item())

                    L_sq = torch.max(torch.tensor(L_list))
                    L_lda_sq = torch.mean(torch.linalg.norm(perturbed_score.prior_score(true_samples[:,None,:],time),2,dim=(1,2))**2) #nsamples,dim,1

                    true_diff_tall_score = perturbed_score.tall_posterior_score(true_samples[:,None,:], x_o_long,time,[diff_prec for _ in range(n_obs+1)],diff_prec_prior)
                    perturbed_score.error = error
                    # print("hahah", perturbed_score.error)

                    est_diff_tall_score = perturbed_score.tall_posterior_score(true_samples[:,None,:], x_o_long,time,list_prec,diff_prec_prior+eps_lda*torch.eye(2),error_lda)
                    est_diff_tall_score_1 = perturbed_score.tall_posterior_score(true_samples[:,None,:], x_o_long,time,list_prec_1,diff_prec_prior+eps_lda*torch.eye(2),error_lda)

                    # tall_error=torch.linalg.norm(true_diff_tall_score-est_diff_tall_score, 2, dim=(1,2))**2
                    tall_error_sq=torch.linalg.norm(true_diff_tall_score-est_diff_tall_score, 2, dim=(1,2))**2
                    tall_error_sq_1=torch.linalg.norm(true_diff_tall_score-est_diff_tall_score_1, 2, dim=(1,2))**2
                    # tall_error=torch.linalg.norm(true_diff_tall_score-est_diff_tall_score, 2, dim=(1,2))
                    error_bound_sq = n_obs*(L_lda_sq**0.5+error_lda)*norm
                    frac = (n_obs*eps_lda+(n_obs+1)*prec_bound)*norm/(1-norm*(n_obs*eps_lda+(n_obs+1)*prec_bound))
                    error_bound_sq*=eps_lda+frac*(M_lda+eps_lda)#*M_lda/(1-M_lda*eps_lda)
                    error_bound_sq+=(n_obs+1)*(L_sq**0.5+error)*norm*(prec_bound+frac*(M+prec_bound))#*M/(1-M*eps))
                    error_bound_sq += norm*(n_obs*M_lda*error_lda+(n_obs+1)*M*error)

                    tall_error_list_sq.append(torch.mean(tall_error_sq).item())
                    tall_error_list_sq_1.append(torch.mean(tall_error_sq_1).item())
                    # tall_error_list.append(torch.mean(tall_error).item())
                    error_bound_list_sq.append((error_bound_sq**2).item())
                    # error_bound_list.append((error_bound).item())
                    time_score.append(time.item())
            dico_tmp["bound"]=error_bound_list_sq
            dico_tmp["empirical"]=tall_error_list_sq
            dico[time]=dico_tmp

            ax4.semilogy(list_nobs+1,tall_error_list_sq, color=colors[j], label=fr"$t={round(time.item(),1)}$")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
            # ax4.semilogy(list_nobs+1,tall_error_list_sq_1, color=colors[j], ls="solid")#,label=fr"$t={round(time.item(),1)}$")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
            ax4.semilogy(list_nobs+1,error_bound_list_sq, ls="dashed",color=colors[j])#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
            ax4.set_xlabel(r"$n$")
        logger.log_artifacts(dico, artifact_name=f"nobs_evol_seed_{cfg.seed}",
                            artifact_type='pickle')
        ax4.set_xlim(0,cfg.nextra+2)

        # ax4.set_xticks(torch.arange(1, cfg.nextra+2,10))
        ax4.set_xticks(torch.tensor([1,10,20,30,40,50]))
        ax4.set_ylabel(r"$\mathrm{E}_{p_t(\theta\mid x_{1:n})}\left[\Vert \nabla_{\theta}\log p_t(\theta\mid x_{1:n})-s_{\phi}(\theta,t,x_{1:n})\Vert_2^2\right]$")
        ax4.plot([], [], c='k', ls="dashed", label='theoretical')
        ax4.plot([], [], c='k', ls="solid", label='empirical')
        handles, labels = ax4.get_legend_handles_labels()
        fig4.legend(handles, labels, loc="center right", bbox_to_anchor=(1,0.5))

        fig4.tight_layout(rect=[0, 0, 0.8, 1])
        plt.show() #use nbos=50 seed =2 and seed en haut=42

    if 0:#dependence in nb of obs
        dico={}
        n_obs = cfg.nextra
        list_nobs = torch.arange(0,n_obs,2)
        dico["list_nobs"]=list_nobs+1
        extra_obs = [x_o]
        for _ in range(n_obs):
            extra_obs.append(simulator1(beta_o))
        x_o_long_full = torch.stack(extra_obs)
        print("\nDimensions for ", n_obs+1, " observations: \n x : ",x_o_long_full.size())
        eps_lda=0.0#01 #error of prior precision
        error_lda = 0.0#1 #score error for prior
        error = 0.05**0.5 #score error on posterior
        fig4,ax4 = plt.subplots(figsize=(6.5,4.5))
        dico["epsilon_dsm_2"]=error**2
        colors = ["red","blue","green","orange"]
        list_prec_tmp=[] #store covariance of indiv posterior at time 0
        eps_max_list = []
        perturbed_score.error=error
        for xi in range(n_obs+1):
            samples_beta = perturbed_score.sampling(num_samples,steps=200, condition=x_o_long_full[xi,:][None,:])[:,0,:] #size (num_samples, 1, dim)
            cov_est = torch.cov(samples_beta.T)
            prec_est = torch.linalg.inv(cov_est)
            eps = torch.linalg.norm(prec_est-prec_post,2)
            list_prec_tmp.append(prec_est)
            eps_max_list.append(eps.item())
        eps = torch.max(torch.tensor(eps_max_list)) #take the max prec error on all
        dico["epsilon"]=eps
        ax4.set_title(fr"$\epsilon_{{\lambda}}={eps_lda}, \ \ \epsilon\leq {round(eps.item(),3)}, \ \ \epsilon_{{\text{{DSM}},\lambda}}^2\leq {round(error_lda**2,3)} \ \ \epsilon_\text{{DSM}}^2\leq{round(error**2,3)}$")
        for j, time in enumerate(torch.tensor([0.1,0.5,0.9])):
            dico_tmp={}
            tall_error_list_sq = []
            tall_error_list_sq_1 = []
            error_bound_list_sq=[]
            error_bound_list=[]
            time_score = []
            for k, n_obs in enumerate(list_nobs):
                x_o_long = x_o_long_full[:n_obs+1,:]
                perturbed_score.error=0.0
                alpha_t = perturbed_score.mean_t(time)**2
                true_samples = torch.distributions.MultivariateNormal(alpha_t**0.5*mu_tall_post(x_o_long).squeeze(1), alpha_t*cov_tall_post(x_o_long)+(1-alpha_t)*torch.eye(2)).sample((num_samples,))
                diff_prec = perturbed_score.diff_prec(prec_post,time) #true diffused post precisions
                M = torch.linalg.norm(diff_prec,2) #all the cov are equal in our case
                diff_prec_prior = perturbed_score.diff_prec(torch.linalg.inv(cov_prior),time) #true diffused prior precision
                M_lda = torch.linalg.norm(diff_prec_prior,2)
                Lda = torch.linalg.inv(-n_obs*diff_prec_prior+(n_obs+1)*diff_prec)
                norm = torch.linalg.norm(Lda,2)
                if n_obs*eps_lda+(n_obs+1)*eps<1/norm :#and eps_lda<1/M_lda and eps<1/M:
                    L_list = []
                    list_prec=[] #store covariance of with eps **fixed**
                    list_prec_1=[] #store **estimated** covariance of indiv posterior
                    # print("hey", perturbed_score.error)
                    for i in range(n_obs+1):
                        norm_indiv_score = torch.linalg.norm(perturbed_score.single_posterior_score(true_samples[:,None,:],x_o_long[i,:],time),2, dim=(1,2))
                        list_prec.append(diff_prec+eps*torch.eye(2)) #Sigma tilde
                        list_prec_1.append(perturbed_score.diff_prec(list_prec_tmp[i],time)) #with estimated cov

                        L_list.append(torch.mean(norm_indiv_score**2).item())

                    L_sq = torch.max(torch.tensor(L_list))
                    L_lda_sq = torch.mean(torch.linalg.norm(perturbed_score.prior_score(true_samples[:,None,:],time),2,dim=(1,2))**2) #nsamples,dim,1

                    true_diff_tall_score = perturbed_score.tall_posterior_score(true_samples[:,None,:], x_o_long,time,[diff_prec for _ in range(n_obs+1)],diff_prec_prior)
                    perturbed_score.error = error
                    # print("hahah", perturbed_score.error)

                    est_diff_tall_score = perturbed_score.tall_posterior_score(true_samples[:,None,:], x_o_long,time,list_prec,diff_prec_prior+eps_lda*torch.eye(2),error_lda)
                    est_diff_tall_score_1 = perturbed_score.tall_posterior_score(true_samples[:,None,:], x_o_long,time,list_prec_1,diff_prec_prior+eps_lda*torch.eye(2),error_lda)

                    # tall_error=torch.linalg.norm(true_diff_tall_score-est_diff_tall_score, 2, dim=(1,2))**2
                    tall_error_sq=torch.linalg.norm(true_diff_tall_score-est_diff_tall_score, 2, dim=(1,2))**2
                    tall_error_sq_1=torch.linalg.norm(true_diff_tall_score-est_diff_tall_score_1, 2, dim=(1,2))**2
                    # tall_error=torch.linalg.norm(true_diff_tall_score-est_diff_tall_score, 2, dim=(1,2))
                    error_bound_sq = n_obs*(L_lda_sq**0.5+error_lda)*norm
                    frac = (n_obs*eps_lda+(n_obs+1)*eps)*norm/(1-norm*(n_obs*eps_lda+(n_obs+1)*eps))
                    error_bound_sq*=eps_lda+frac*(M_lda+eps_lda)#*M_lda/(1-M_lda*eps_lda)
                    error_bound_sq+=(n_obs+1)*(L_sq**0.5+error)*norm*(eps+frac*(M+eps))#*M/(1-M*eps))
                    error_bound_sq += norm*(n_obs*M_lda*error_lda+(n_obs+1)*M*error)

                    tall_error_list_sq.append(torch.mean(tall_error_sq).item())
                    tall_error_list_sq_1.append(torch.mean(tall_error_sq_1).item())
                    # tall_error_list.append(torch.mean(tall_error).item())
                    error_bound_list_sq.append((error_bound_sq**2).item())
                    # error_bound_list.append((error_bound).item())
                    time_score.append(time.item())
            dico_tmp["bound"]=error_bound_list_sq
            dico_tmp["empirical"]=tall_error_list_sq
            dico[time]=dico_tmp

            ax4.semilogy(list_nobs+1,tall_error_list_sq, color=colors[j], label=fr"$t={round(time.item(),1)}$")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
            # ax4.semilogy(list_nobs+1,tall_error_list_sq_1, color=colors[j], ls="solid")#,label=fr"$t={round(time.item(),1)}$")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
            ax4.semilogy(list_nobs+1,error_bound_list_sq, ls="dashed",color=colors[j])#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
            ax4.set_xlabel(r"$n$")
        logger.log_artifacts(dico, artifact_name=f"nobs_evol_seed_{cfg.seed}",
                            artifact_type='pickle')
        ax4.set_xlim(0,cfg.nextra+2)

        # ax4.set_xticks(torch.arange(1, cfg.nextra+2,10))
        ax4.set_xticks(torch.tensor([1,10,20,30,40,50]))
        ax4.set_ylabel(r"$\mathrm{E}_{p_t(\theta\mid x_{1:n})}\left[\Vert \nabla_{\theta}\log p_t(\theta\mid x_{1:n})-s_{\phi}(\theta,t,x_{1:n})\Vert_2^2\right]$")
        ax4.plot([], [], c='k', ls="dashed", label='theoretical')
        ax4.plot([], [], c='k', ls="solid", label='empirical')
        handles, labels = ax4.get_legend_handles_labels()
        fig4.legend(handles, labels, loc="center right", bbox_to_anchor=(1,0.5))

        fig4.tight_layout(rect=[0, 0, 0.8, 1])
        plt.show() #use nbos=50 seed =2 and seed en haut=42

    if 0:#dependence in epsilon (error of covariance)
        dico={}
        n_obs = cfg.nextra
        list_cov_error = torch.linspace(0.01,0.25,100)#0.5,0.7]
        dico["cov_error"]=list_cov_error
        dico["nobs"]=n_obs
        extra_obs = [x_o]
        for _ in range(n_obs):
            extra_obs.append(simulator1(beta_o))
        x_o_long_full = torch.stack(extra_obs)
        print("\nDimensions for ", n_obs+1, " observations: \n x : ",x_o_long_full.size())
        eps_lda=0.0#001 #error of prior precision
        error_lda = 0.0#01 #error for prior score
        error = 0.0#01 #small score error on indiv posteriors
        dico["epsilon_dsm"]=error
        fig3,ax3 = plt.subplots(figsize=(6.5,4.5))
        fig3.suptitle(fr"$\epsilon_{{\lambda}}={eps_lda}, \ \ n= {n_obs+1}, \ \ \epsilon_{{\text{{DSM}},\lambda}}^2\leq {round(error_lda**2,3)} \ \ \epsilon_\text{{DSM}}^2\leq{round(error**2,3)}$")
        colors = ["red","blue","green","orange"]
        for j, time in enumerate(torch.tensor([0.1,0.5,0.9])):#linspace(0.1,1.0,9)):
            dico_tmp={}
            tall_error_list=[]
            tall_error_list_sq = []
            error_bound_list_sq=[]
            error_bound_list=[]
            time_score = []
            for k, eps in enumerate(list_cov_error):
                x_o_long = x_o_long_full[:n_obs+1,:]
                perturbed_score.error=0.0
                alpha_t = perturbed_score.mean_t(time)**2
                true_samples = torch.distributions.MultivariateNormal(alpha_t**0.5*mu_tall_post(x_o_long).squeeze(1), alpha_t*cov_tall_post(x_o_long)+(1-alpha_t)*torch.eye(DIM)).sample((num_samples,))
                diff_prec = perturbed_score.diff_prec(prec_post,time) #true diffused post precisions
                M = torch.linalg.norm(diff_prec,2) #all the cov are equal in our case
                diff_prec_prior = perturbed_score.diff_prec(torch.linalg.inv(cov_prior),time) #true diffused prior precision
                M_lda = torch.linalg.norm(diff_prec_prior,2)
                # Lda = torch.linalg.norm(torch.linalg.inv(-n_obs*diff_prec_prior+(n_obs+1)*diff_prec),2)
                Lda = torch.linalg.inv(-n_obs*diff_prec_prior+(n_obs+1)*diff_prec)
                norm = torch.linalg.norm(Lda,2)
                L_sq=0
                L=0
                list_prec=[] #store covariance of **perturbed** indiv posterior
                for i in range(n_obs+1):
                    norm_indiv_score = torch.linalg.norm(perturbed_score.single_posterior_score(true_samples[:,None,:],x_o_long[i,:],time),2, dim=(1,2))
                    list_prec.append(diff_prec+eps*torch.eye(DIM)) #Sigma tilde
                    if L_sq<=torch.mean(norm_indiv_score**2).item():
                        L_sq=torch.mean(norm_indiv_score**2).item()
                    if L<=torch.mean(norm_indiv_score).item():
                        L=torch.mean(norm_indiv_score).item()

                L_lda_sq = torch.mean(torch.linalg.norm(perturbed_score.prior_score(true_samples[:,None,:],time),2,dim=(1,2))**2) #nsamples,dim,1
                L_lda = torch.mean(torch.linalg.norm(perturbed_score.prior_score(true_samples[:,None,:],time),2,dim=(1,2))**2) #nsamples,dim,1
                true_diff_tall_score = perturbed_score.tall_posterior_score(true_samples[:,None,:], x_o_long,time,[diff_prec for _ in range(n_obs+1)],diff_prec_prior)
                perturbed_score.error = error
                est_diff_tall_score = perturbed_score.tall_posterior_score(true_samples[:,None,:], x_o_long,time,list_prec,diff_prec_prior+eps_lda*torch.eye(DIM),error_lda)

                tall_error=torch.linalg.norm(true_diff_tall_score-est_diff_tall_score, 2, dim=(1,2))**2
                tall_error_sq=torch.linalg.norm(true_diff_tall_score-est_diff_tall_score, 2, dim=(1,2))**2
                tall_error=torch.linalg.norm(true_diff_tall_score-est_diff_tall_score, 2, dim=(1,2))
                error_bound_sq = n_obs*(L_lda_sq**0.5+error_lda)*norm
                frac = (n_obs*eps_lda+(n_obs+1)*eps)*norm/(1-norm*(n_obs*eps_lda+(n_obs+1)*eps))
                error_bound_sq*=eps_lda+frac*(M_lda+eps_lda)#*M_lda/(1-M_lda*eps_lda)
                error_bound_sq+=(n_obs+1)*(L_sq**0.5+error)*norm*(eps+frac*(M+eps))#*M/(1-M*eps))
                error_bound_sq += norm*(n_obs*M_lda*error_lda+(n_obs+1)*M*error)

                error_bound = n_obs*(L_lda+error_lda)*norm
                error_bound*=eps_lda+frac*(M_lda+eps_lda)#*M_lda/(1-M_lda*eps_lda)
                error_bound+=(n_obs+1)*(L+error)*norm*(eps+frac*(M+eps))#*M/(1-M*eps))
                error_bound += norm*(n_obs*M_lda*error_lda+(n_obs+1)*M*error)

                if n_obs*eps_lda+(n_obs+1)*eps<1/norm :#and eps_lda<1/M_lda and eps<1/M:
                    tall_error_list_sq.append(torch.mean(tall_error_sq).item())
                    tall_error_list.append(torch.mean(tall_error).item())
                    error_bound_list_sq.append((error_bound**2).item())
                    error_bound_list.append((error_bound).item())
                    time_score.append(time.item())
            dico_tmp["bound"]=error_bound_list_sq
            dico_tmp["empirical"]=tall_error_list_sq
            dico[round(time.item(),2)]=dico_tmp
            ax3.loglog(list_cov_error,tall_error_list_sq, color=colors[j], label=fr"$t={round(time.item(),1)}$")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
            ax3.loglog(list_cov_error,error_bound_list_sq, ls="dashed",color=colors[j])#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
            ax3.set_xlabel(r"$\epsilon$")

            # ax2[j].loglog(list_cov_error,tall_error_list_sq, color="blue", label="empirical squared error")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
            # ax2[j].loglog(list_cov_error,error_bound_list_sq, ls="dotted",color="blue", label="theoretical squared bound")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
            # ax2[j].loglog(list_cov_error,tall_error_list, color="red", label="empirical error")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
            # ax2[j].loglog(list_cov_error,error_bound_list, ls="dotted",color="red", label="theoretical bound")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")

            # ax2[j].set_title(fr"$t={round(time.item(),3)}$")

            # ax1[j].plot(list_cov_error,tall_error_list_sq, color="blue", label="empirical squared error")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
            # ax1[j].plot(list_cov_error,error_bound_list_sq, ls="dotted",color="blue", label="theoretical squared bound")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")

            # ax1[j].plot(list_cov_error,tall_error_list, color="red", label="empirical error")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
            # ax1[j].plot(list_cov_error,error_bound_list, ls="dotted",color="red", label="theoretical bound")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
            # ax1[j].set_title(fr"$t={round(time.item(),3)}$")
            # if j>=6:
            #     ax1[j].set_xlabel(r"$\epsilon$")
            #     ax2[j].set_xlabel(r"$\epsilon$")

        logger.log_artifacts(dico, artifact_name=f"epsilon_evol_seed_{cfg.seed}",
                            artifact_type='pickle')
        ax3.plot([], [], c='k', ls="dashed", label='theoretical')
        ax3.plot([], [], c='k', ls="solid", label='empirical')
        handles, labels = ax3.get_legend_handles_labels()
        fig3.legend(handles, labels, loc="center right", bbox_to_anchor=(1,0.5))
        fig3.tight_layout(rect=[0, 0, 0.8, 1])
        plt.show()

    if 0:#dependence in epsilon_dsm (error of score) with estimated cov
        dico={}
        n_obs = cfg.nextra
        dico["nobs"]=n_obs
        list_error = torch.linspace(0.01,1.75,50)#0.5,0.7]
        dico["list_epsilon_dsm_2"]=list_error**2
        extra_obs = [x_o]
        for _ in range(n_obs):
            extra_obs.append(simulator1(beta_o))
        x_o_long_full = torch.stack(extra_obs)
        print("\nDimensions for ", n_obs+1, " observations: \n x : ",x_o_long_full.size())
        eps_lda=0.0#01 #error of prior precision
        error_lda = 0.0#1 #error for prior score
        #eps = 0.05 #error on covariance matrices
        # eps_max=0 #error in precision matrix at time 0 (same for all time t)
        list_est_prec = []

        fig3,ax3 = plt.subplots(figsize=(6.5,4.5))
        fig3.suptitle(fr"$\epsilon_{{\lambda}}={eps_lda}, \ \ n= {n_obs+1}, \ \ \epsilon_{{\text{{DSM}},\lambda}}^2\leq {round(error_lda**2,3)}$")# \ \ \epsilon\leq{round(eps_max.item(),3)}$")
        colors = ["red","blue","green","orange","pink","brown"]
        # eps_max=0.001
        # fig3.suptitle(fr"$\epsilon_{{\lambda}}={eps_lda}, \ \ n= {n_obs+1}, \ \ \epsilon_{{\text{{DSM}},\lambda}}^2\leq {round(error_lda**2,3)} \ \ \epsilon\leq{round(eps_max,3)}$")
        for j, time in enumerate(torch.tensor([0.1,0.5,0.9])):#linspace(0.1,1.0,9)):
            dico_tmp={}
            tall_error_list=[]
            tall_error_list_sq = []
            error_bound_list_sq=[]
            error_bound_list=[]
            time_score = []
            perturbed_score.error=0.0
            alpha_t = perturbed_score.mean_t(time)**2
            true_samples = torch.distributions.MultivariateNormal(alpha_t**0.5*mu_tall_post(x_o_long_full).squeeze(1), alpha_t*cov_tall_post(x_o_long_full)+(1-alpha_t)*torch.eye(DIM)).sample((num_samples,))
            diff_prec = perturbed_score.diff_prec(prec_post,time) #true diffused post precisions
            M = torch.linalg.norm(diff_prec,2) #all the cov are equal in our case
            diff_prec_prior = perturbed_score.diff_prec(torch.linalg.inv(cov_prior),time) #true diffused prior precision
            M_lda = torch.linalg.norm(diff_prec_prior,2)
            # Lda = torch.linalg.norm(torch.linalg.inv(-n_obs*diff_prec_prior+(n_obs+1)*diff_prec),2)
            Lda = torch.linalg.inv(-n_obs*diff_prec_prior+(n_obs+1)*diff_prec)
            norm = torch.linalg.norm(Lda,2)
            true_diff_tall_score = perturbed_score.tall_posterior_score(true_samples[:,None,:], x_o_long_full,time,[diff_prec for _ in range(n_obs+1)],diff_prec_prior)

            for k, error in enumerate(reversed(list_error)):
                # x_o_long = x_o_long_full[:n_obs+1,:]
                list_prec=[] #store covariance of **perturbed** indiv posterior
                eps_max_list = []
                perturbed_score.error=error
                for xi in range(n_obs+1):
                    samples_beta = perturbed_score.sampling(num_samples,steps=200, condition=x_o_long_full[xi,:][None,:])[:,0,:] #size (num_samples, 1, dim)
                    cov_est = torch.cov(samples_beta.T)
                    prec_est = torch.linalg.inv(cov_est)
                    # list_est_prec.append(prec_est)
                    eps_tmp = torch.linalg.norm(prec_est-prec_post,2)
                    list_prec.append(perturbed_score.diff_prec(prec_est,time)) #Sigma tilde
                    eps_max_list.append(eps_tmp.item())
                eps_max = torch.max(torch.tensor(eps_max_list))
                print(eps_max)
                if k==0:
                    eps = eps_max
                if n_obs*eps_lda+(n_obs+1)*eps_max<1/norm :#and eps_lda<1/M_lda and eps<1/M:
                    L_list = []
                    perturbed_score.error=0.0
                    for i in range(n_obs+1):
                        norm_indiv_score = torch.linalg.norm(perturbed_score.single_posterior_score(true_samples[:,None,:],x_o_long_full[i,:],time),2, dim=(1,2))
                        L_list.append(torch.mean(norm_indiv_score**2))
                    L_sq = torch.max(torch.tensor(L_list))
                    L_lda_sq = torch.mean(torch.linalg.norm(perturbed_score.prior_score(true_samples[:,None,:],time),2,dim=(1,2))**2) #nsamples,dim,1
                    # L_lda = torch.mean(torch.linalg.norm(perturbed_score.prior_score(true_samples[:,None,:],time),2,dim=(1,2))) #nsamples,dim,1

                    perturbed_score.error = error
                    est_diff_tall_score = perturbed_score.tall_posterior_score(true_samples[:,None,:], x_o_long_full,time,list_prec,diff_prec_prior+eps_lda*torch.eye(DIM),error_lda)
                    tall_error_sq=torch.linalg.norm(true_diff_tall_score-est_diff_tall_score, 2, dim=(1,2))**2

                    # tall_error=torch.linalg.norm(true_diff_tall_score-est_diff_tall_score, 2, dim=(1,2))
                    error_bound_sq = n_obs*(L_lda_sq**0.5+error_lda)*norm
                    frac = (n_obs*eps_lda+(n_obs+1)*eps)*norm/(1-norm*(n_obs*eps_lda+(n_obs+1)*eps))
                    error_bound_sq*=eps_lda+frac*(M_lda+eps_lda)#*M_lda/(1-M_lda*eps_lda)
                    error_bound_sq+=(n_obs+1)*(L_sq**0.5+error)*norm*(eps+frac*(M+eps))#*M/(1-M*eps))
                    error_bound_sq += norm*(n_obs*M_lda*error_lda+(n_obs+1)*M*error)

                    tall_error_list_sq.append(torch.mean(tall_error_sq).item())
                    # tall_error_list.append(torch.mean(tall_error).item())
                    error_bound_list_sq.append((error_bound_sq**2).item())
                    # error_bound_list.append((new_error_bound).item())
                    # time_score.append(time.item())
                # tau_error = 4*(n_obs)**2*(2*eps_lda**2*(error_lda**2+L_lda_sq)+M_lda**2*error_lda**2)+4*(1+n_obs)**2*(2*eps_max**2*(error**2+L_sq)+M**2*error**2)
                # # tau_tilde = 2**0.5*n_obs*(M_lda+eps_lda)*(error_lda**2+L_lda_sq)**0.5+2**0.5*(n_obs+1)*(M+eps_max)*(L_sq+error**2)**0.5
                # tau_tilde = 2*n_obs**2*(M_lda**2+eps_lda**2)*2*(error_lda**2+L_lda_sq)*(2+n_obs)+4*(n_obs+1)*(M**2+eps_max**2)*2*(L_sq+error**2)

                # new_error_bound = norm*tau_error**0.5+norm*frac*tau_tilde
                # new_error_bound = 2*norm**2*tau_error+2*norm**2*frac**2*tau_tilde**2
                # new_error_bound = 2*norm**2*tau_error+2*norm**2*frac**2*tau_tilde

                # error_bound = n_obs*(L_lda+error_lda)*norm
                # error_bound*=eps_lda+frac*(M_lda+eps_lda)#*M_lda/(1-M_lda*eps_lda)
                # error_bound+=(n_obs+1)*(L+error)*norm*(eps+frac*(M+eps))#*M/(1-M*eps))
                # error_bound += norm*(n_obs*M_lda*error_lda**0.5+(n_obs+1)*M*error**0.5)

                # if n_obs*eps_lda+(n_obs+1)*eps_max<1/norm :#and eps_lda<1/M_lda and eps<1/M:
                #     tall_error_list_sq.append(torch.mean(tall_error_sq).item())
                #     # tall_error_list.append(torch.mean(tall_error).item())
                #     error_bound_list_sq.append((error_bound_sq**2).item())
                #     # error_bound_list.append((new_error_bound).item())
                #     time_score.append(time.item())
            # plt.subplot(3,3,j+1)
            # ax2[j].plot(torch.log(list_error),torch.log(torch.tensor(tall_error_list)), color=colors[1], label="error")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
            dico_tmp["bound"]=list(reversed(error_bound_list_sq))
            dico_tmp["empirical"]=list(reversed(tall_error_list_sq))
            dico[time]=dico_tmp
            ax3.loglog(list_error**2,list(reversed(tall_error_list_sq)), color=colors[j], label=fr"$t={round(time.item(),1)}, \epsilon={round(eps.item(),3)}$")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
            ax3.loglog(list_error**2,list(reversed(error_bound_list_sq)), ls="dashed",color=colors[j])#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
            # ax3.loglog(list_error**2,error_bound_list, ls="dashed",color=colors[j])#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
            # ax3.loglog(list_error**2,list_error**2, ls="dotted",color="orange")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
            # ax3.loglog(list_error**2,list_error+list_error**2, ls="dotted",color="brown")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
            ax3.set_xlabel(r"$\epsilon_{\text{DSM}}^2$")

            # ax2[j].loglog(list_error,tall_error_list_sq, color="blue", label="empirical squared error")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
            # ax2[j].loglog(list_error,error_bound_list_sq, ls="dotted",color="blue", label="theoretical squared bound")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")

            # ax2[j].loglog(list_error,tall_error_list, color="red", label="empirical error")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
            # ax2[j].loglog(list_error,error_bound_list, ls="dotted",color="red", label="theoretical bound")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
            # ax2[j].loglog(list_error,list_error**2, ls="dotted",color="green", label=r"$\epsilon_\text{DSM}^4$")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
            # ax2[j].loglog(list_error,list_error, ls="dotted",color="orange", label=r"$\epsilon_\text{DSM}^2$")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")

            # ax2[j].set_xlabel(r"$\epsilon_\text{DSM}^2$")
            # ax2[j].legend()
            # ax2[j].set_title(fr"$t={round(time.item(),3)}$")
            # ax1[j].plot(list_error**2,tall_error_list, color="red", label="error")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
        logger.log_artifacts(dico, artifact_name=f"evol_epsilon_dsm_seed_{cfg.seed}",
                            artifact_type='pickle')
        ax3.plot([], [], c='k', ls="dashed", label='theoretical')
        ax3.plot([], [], c='k', ls="solid", label='empirical')
        handles, labels = ax3.get_legend_handles_labels()
        fig3.legend(handles, labels, loc="center right", bbox_to_anchor=(1,0.5))
        plt.tight_layout(rect=[0, 0, 0.8, 1])
        plt.show()

    if 0:#dependence in epsilon_dsm (error of score)
        n_obs = cfg.nextra
        list_error = torch.linspace(0.01,1.75,50)#0.5,0.7]
        extra_obs = [x_o]
        for _ in range(n_obs):
            extra_obs.append(simulator1(beta_o))
        x_o_long_full = torch.stack(extra_obs)
        print("\nDimensions for ", n_obs+1, " observations: \n x : ",x_o_long_full.size())
        eps_lda=0.0#01 #error of prior precision
        error_lda = 0.0#1 #error for prior score
        list_est_prec = []

        fig3,ax3 = plt.subplots(figsize=(6.5,4.5))

        # fig3.suptitle(fr"$\epsilon_{{\lambda}}={eps_lda}, \ \ n= {n_obs+1}, \ \ \epsilon_{{\text{{DSM}},\lambda}}^2\leq {round(error_lda**2,3)}$")# \ \ \epsilon\leq{round(eps_max.item(),3)}$")
        colors = ["red","blue","green","orange","pink","brown"]

        for j, time in enumerate(torch.tensor([0.1,0.5,0.9])):#linspace(0.1,1.0,9)):
            tall_error_list=[]
            tall_error_list_sq = []
            error_bound_list_sq=[]
            error_bound_list=[]
            time_score = []
            error_max = list_error[-1]
            list_prec=[] #store covariance of **perturbed** indiv posterior
            eps_max_list = []
            perturbed_score.error=error_max
            for xi in range(n_obs+1):
                samples_beta = perturbed_score.sampling(num_samples,steps=500, condition=x_o_long_full[xi,:][None,:])[:,0,:] #size (num_samples, 1, dim)
                cov_est = torch.cov(samples_beta.T)
                prec_est = torch.linalg.inv(cov_est)
                eps = torch.linalg.norm(prec_est-prec_post,2)
                list_prec.append(perturbed_score.diff_prec(prec_est,time)) #Sigma tilde
                eps_max_list.append(eps.item())

            eps_max = torch.max(torch.tensor(eps_max_list))
            perturbed_score.error=0.0
            alpha_t = perturbed_score.mean_t(time)**2
            true_samples = torch.distributions.MultivariateNormal(alpha_t**0.5*mu_tall_post(x_o_long_full).squeeze(1), alpha_t*cov_tall_post(x_o_long_full)+(1-alpha_t)*torch.eye(DIM)).sample((num_samples,))
            diff_prec = perturbed_score.diff_prec(prec_post,time) #true diffused post precisions
            M = torch.linalg.norm(diff_prec,2) #all the cov are equal in our case
            diff_prec_prior = perturbed_score.diff_prec(torch.linalg.inv(cov_prior),time) #true diffused prior precision
            M_lda = torch.linalg.norm(diff_prec_prior,2)
            # Lda = torch.linalg.norm(torch.linalg.inv(-n_obs*diff_prec_prior+(n_obs+1)*diff_prec),2)
            Lda = torch.linalg.inv(-n_obs*diff_prec_prior+(n_obs+1)*diff_prec)
            norm = torch.linalg.norm(Lda,2)
            if n_obs*eps_lda+(n_obs+1)*eps_max<1/norm :#and eps_lda<1/M_lda and eps<1/M:
                L_list = []
                for i in range(n_obs+1):
                    norm_indiv_score = torch.linalg.norm(perturbed_score.single_posterior_score(true_samples[:,None,:],x_o_long_full[i,:],time),2, dim=(1,2))
                    L_list.append(torch.mean(norm_indiv_score**2))
                L_sq = torch.max(torch.tensor(L_list))
                L_lda_sq = torch.mean(torch.linalg.norm(perturbed_score.prior_score(true_samples[:,None,:],time),2,dim=(1,2))**2) #nsamples,dim,1
                # L_lda = torch.mean(torch.linalg.norm(perturbed_score.prior_score(true_samples[:,None,:],time),2,dim=(1,2))) #nsamples,dim,1
                true_diff_tall_score = perturbed_score.tall_posterior_score(true_samples[:,None,:], x_o_long_full,time,[diff_prec for _ in range(n_obs+1)],diff_prec_prior)
                for k, error in enumerate(list_error):
                    perturbed_score.error = error
                    est_diff_tall_score = perturbed_score.tall_posterior_score(true_samples[:,None,:], x_o_long_full,time,[diff_prec+eps_max*torch.eye(DIM) for _ in range(n_obs+1)],diff_prec_prior+eps_lda*torch.eye(DIM),error_lda)
                    tall_error_sq=torch.linalg.norm(true_diff_tall_score-est_diff_tall_score, 2, dim=(1,2))**2

                    error_bound_sq = n_obs*(L_lda_sq**0.5+error_lda)*norm
                    frac = (n_obs*eps_lda+(n_obs+1)*eps_max)*norm/(1-norm*(n_obs*eps_lda+(n_obs+1)*eps_max))
                    error_bound_sq*=eps_lda+frac*(M_lda+eps_lda)#*M_lda/(1-M_lda*eps_lda)
                    error_bound_sq+=(n_obs+1)*(L_sq**0.5+error)*norm*(eps_max+frac*(M+eps))#*M/(1-M*eps))
                    error_bound_sq += norm*(n_obs*M_lda*error_lda+(n_obs+1)*M*error)

                    tall_error_list_sq.append(torch.mean(tall_error_sq).item())
                    # tall_error_list.append(torch.mean(tall_error).item())
                    error_bound_list_sq.append((error_bound_sq**2).item())
                    # error_bound_list.append((new_error_bound).item())
                    time_score.append(time.item())

            # ax2[j].plot(torch.log(list_error),torch.log(torch.tensor(tall_error_list)), color=colors[1], label="error")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
            ax3.loglog(list_error**2,tall_error_list_sq, color=colors[j], label=fr"$t={round(time.item(),1)}$")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
            ax3.loglog(list_error**2,error_bound_list_sq, ls="dashed",color=colors[j])#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
            # ax3.loglog(list_error**2,error_bound_list, ls="dashed",color=colors[j])#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
            # ax3.loglog(list_error**2,list_error**2, ls="dotted",color="orange")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
            # ax3.loglog(list_error**2,list_error+list_error**2, ls="dotted",color="brown")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
            ax3.set_xlabel(r"$\epsilon_{\text{DSM}}^2$")

        fig3.suptitle(fr"$\epsilon_{{\lambda}}={eps_lda}, \ \ n= {n_obs+1}, \ \ \epsilon_{{\text{{DSM}},\lambda}}^2\leq {round(error_lda**2,3)} \ \ \epsilon\leq{round(eps_max.item(),3)}$")
        ax3.plot([], [], c='k', ls="dashed", label='theoretical')
        ax3.plot([], [], c='k', ls="solid", label='empirical')
        handles, labels = ax3.get_legend_handles_labels()
        fig3.legend(handles, labels, loc="center right", bbox_to_anchor=(1,0.5))
        plt.tight_layout(rect=[0, 0, 0.8, 1])
        plt.show()

    if 0:#total error with link with prec error
        n_obs = cfg.nextra
        list_error = torch.linspace(0.01,1.75,50)#0.5,0.7]
        extra_obs = [x_o]
        for _ in range(n_obs):
            extra_obs.append(simulator1(beta_o))
        x_o_long_full = torch.stack(extra_obs)
        print("\nDimensions for ", n_obs+1, " observations: \n x : ",x_o_long_full.size())
        eps_lda=0.0#01 #error of prior precision
        error_lda = 0.0#1 #error for prior score
        list_est_prec = []
        error = 5e-4
        fig3,ax3 = plt.subplots(figsize=(6.5,4.5))
        fig4,ax4 = plt.subplots(figsize=(6.5,4.5))

        # fig3.suptitle(fr"$\epsilon_{{\lambda}}={eps_lda}, \ \ n= {n_obs+1}, \ \ \epsilon_{{\text{{DSM}},\lambda}}^2\leq {round(error_lda**2,3)}$")# \ \ \epsilon\leq{round(eps_max.item(),3)}$")
        colors = ["red","blue","green","orange","pink","brown"]
        tall_error_list_sq = []
        tall_error_list_sq_1 = []
        error_bound_list_sq=[]
        time_score = []
        for j, time in enumerate(torch.linspace(0.1,1.0,40)):
            perturbed_score.error=0.0
            alpha_t = perturbed_score.mean_t(time)**2
            true_samples = torch.distributions.MultivariateNormal(alpha_t**0.5*mu_tall_post(x_o_long_full).squeeze(1), alpha_t*cov_tall_post(x_o_long_full)+(1-alpha_t)*torch.eye(DIM)).sample((num_samples,))
            diff_prec = perturbed_score.diff_prec(prec_post,time) #true diffused post precisions
            s = torch.linalg.norm(diff_prec,2)
            m = torch.linalg.norm(torch.linalg.inv(diff_prec),2)**0.5 #norm of inv diff prec
            delta = 2*(2*m+1.0/m/s)**2+4*(2**0.5+1)/s
            M = torch.linalg.norm(diff_prec,2) #all the cov are equal in our case
            diff_prec_prior = perturbed_score.diff_prec(torch.linalg.inv(cov_prior),time) #true diffused prior precision
            M_lda = torch.linalg.norm(diff_prec_prior,2)
            # Lda = torch.linalg.norm(torch.linalg.inv(-n_obs*diff_prec_prior+(n_obs+1)*diff_prec),2)
            Lda = torch.linalg.inv(-n_obs*diff_prec_prior+(n_obs+1)*diff_prec)
            norm = torch.linalg.norm(Lda,2)
            list_prec=[] #store covariance of **perturbed** indiv posterior
            eps_max_list = []
            eps_list = []
            perturbed_score.error=error
            for xi in range(n_obs+1):
                samples_beta = perturbed_score.sampling(num_samples,steps=200, condition=x_o_long_full[xi,:][None,:])[:,0,:] #size (num_samples, 1, dim)
                cov_est = torch.cov(samples_beta.T)
                prec_est = torch.linalg.inv(cov_est)
                eps = torch.linalg.norm(prec_est-prec_post,2)
                list_prec.append(perturbed_score.diff_prec(prec_est,time)) #Sigma tilde
                wass = wasserstein_dist(torch.mean(samples_beta,dim=0),mu_post(x_o_long_full[xi,:])[0,:,0],cov_est,cov_post)**0.5
                eps_list.append(eps.item())# print(wass.item(),m,s)
                if wass.item()<m/2**0.5 and wass.item()<(delta**0.5-2**0.5*(2*m+1.0/m/s))/2/(2**0.5+1):
                    print("yes")
                    gamma = (2**1.5*m*wass+wass**2*(1+2**0.5))*m/(m-wass*2**0.5)
                    epsilon_j = gamma*s**2/(1-gamma*s)
                    eps_max_list.append(epsilon_j.item())
                    # print("gamma",gamma.item())
                    # print("eps",epsilon_j.item(),1-gamma*s,gamma*s**2, gamma*s)
            if len(eps_max_list)==0:
                eps_max = torch.max(torch.tensor(eps_list))
            else:
                eps_max = torch.max(torch.tensor(eps_max_list)) #return max gamma_j
            print(eps_max, 1/norm)
            if n_obs*eps_lda+(n_obs+1)*eps_max<1/norm :#and eps_lda<1/M_lda and eps<1/M:
                print("check")
                perturbed_score.error = 0.0
                L_list = []
                for i in range(n_obs+1):
                    norm_indiv_score = torch.linalg.norm(perturbed_score.single_posterior_score(true_samples[:,None,:],x_o_long_full[i,:],time),2, dim=(1,2))
                    L_list.append(torch.mean(norm_indiv_score**2))
                L_sq = torch.max(torch.tensor(L_list))
                L_lda_sq = torch.mean(torch.linalg.norm(perturbed_score.prior_score(true_samples[:,None,:],time),2,dim=(1,2))**2) #nsamples,dim,1
                true_diff_tall_score = perturbed_score.tall_posterior_score(true_samples[:,None,:], x_o_long_full,time,[diff_prec for _ in range(n_obs+1)],diff_prec_prior)
                perturbed_score.error = error
                # est_diff_tall_score = perturbed_score.tall_posterior_score(true_samples[:,None,:], x_o_long_full,time,[diff_prec+eps_max*torch.eye(2) for _ in range(n_obs+1)],diff_prec_prior+eps_lda*torch.eye(2),error_lda)
                est_diff_tall_score_1 = perturbed_score.tall_posterior_score(true_samples[:,None,:], x_o_long_full,time,list_prec,diff_prec_prior+eps_lda*torch.eye(DIM),error_lda)
                # tall_error_sq=torch.linalg.norm(true_diff_tall_score-est_diff_tall_score, 2, dim=(1,2))**2
                tall_error_sq_1=torch.linalg.norm(true_diff_tall_score-est_diff_tall_score_1, 2, dim=(1,2))**2

                error_bound_sq = n_obs*(L_lda_sq**0.5+error_lda)*norm
                frac = (n_obs*eps_lda+(n_obs+1)*eps_max)*norm/(1-norm*(n_obs*eps_lda+(n_obs+1)*eps_max))
                error_bound_sq*=eps_lda+frac*(M_lda+eps_lda)#*M_lda/(1-M_lda*eps_lda)
                error_bound_sq+=(n_obs+1)*(L_sq**0.5+error)*norm*(eps_max+frac*(M+eps))#*M/(1-M*eps))
                error_bound_sq += norm*(n_obs*M_lda*error_lda+(n_obs+1)*M*error)

                # tall_error_list_sq.append(torch.mean(tall_error_sq).item())
                tall_error_list_sq_1.append(torch.mean(tall_error_sq_1).item())
                # tall_error_list.append(torch.mean(tall_error).item())
                error_bound_list_sq.append((error_bound_sq**2).item())
                # error_bound_list.append((new_error_bound).item())
                time_score.append(time.item())

            # ax2[j].plot(torch.log(list_error),torch.log(torch.tensor(tall_error_list)), color=colors[1], label="error")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
        # ax3.loglog(time_score,tall_error_list_sq, color=colors[j], label=fr"$t={round(time.item(),1)}$")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
        print("final",len(tall_error_list_sq_1))
        # ax3.loglog(time_score,tall_error_list_sq_1, marker="o", color=colors[0], ls="solid",label="empirical")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
        # ax3.loglog(time_score,error_bound_list_sq, marker="o", ls="dashed",color=colors[0], label="theoretical")
        ax4.semilogy(time_score,tall_error_list_sq_1, marker="o", color=colors[0], ls="solid",label="empirical")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
        ax4.semilogy(time_score,error_bound_list_sq, marker="o", ls="dashed",color=colors[0], label="theoretical")
        # ax3.loglog(list_error**2,error_bound_list, ls="dashed",color=colors[j])#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
        # ax3.loglog(list_error**2,list_error**2, ls="dotted",color="orange")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
        # ax3.loglog(list_error**2,list_error+list_error**2, ls="dotted",color="brown")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
        ax3.set_xlabel(r"$t$")
        ax4.set_xlabel(r"$t$")

        fig3.suptitle(fr"$\epsilon_{{\lambda}}={eps_lda}, \ \ n= {n_obs+1}, \ \ \epsilon_{{\text{{DSM}},\lambda}}^2\leq {round(error_lda**2,3)} \ \ \epsilon\leq{round(eps_max.item(),3)}$")
        fig4.suptitle(fr"$\epsilon_{{\lambda}}={eps_lda}, \ \ n= {n_obs+1}, \ \ \epsilon_{{\text{{DSM}},\lambda}}^2\leq {round(error_lda**2,3)}, \ \ \epsilon\leq{round(eps_max.item(),3)} \ \ \epsilon_\text{{DSM}}^2\leq{round(error**2,4)}$")
        # ax3.plot([], [], c='k', ls="dashed", label='theoretical')
        # ax3.plot([], [], c='k', ls="solid", label='empirical')
        # ax4.plot([], [], c='k', ls="dashed", label='theoretical')
        # ax4.plot([], [], c='k', ls="solid", label='empirical')
        handles, labels = ax3.get_legend_handles_labels()
        fig3.legend(handles, labels, loc="center right", bbox_to_anchor=(1,0.5))
        handles, labels = ax4.get_legend_handles_labels()
        fig4.legend(handles, labels, loc="center right", bbox_to_anchor=(1,0.5))
        fig3.tight_layout(rect=[0, 0, 0.8, 1])
        fig4.tight_layout(rect=[0, 0, 0.8, 1])
        plt.show()

    if 0:#plot from pickle files (figure 2)
        import pickle
        filename="logs/eurips/epsilon_evol_seed_1"
        with open(filename, 'rb') as f:
            data_epsilon = pickle.load(f)
        filename="logs/eurips/m_nobs_evol_seed_42"
        with open(filename, 'rb') as f:
            data_nobs = pickle.load(f)
        filename="logs/eurips/evol_epsilon_dsm_seed_10"
        with open(filename, 'rb') as f:
            data_epsilon_dsm = pickle.load(f)
        print(data_epsilon.keys())
        print(data_epsilon_dsm.keys())
        print(data_nobs.keys())
        width = 2#1.7
        police = 15
        police_xlabel = 14
        policeticks = 11
        time = [0.1,0.5,0.9]
        colors = ["red","C0","C2"]
        colors=["#c1121f","#00b4d8","#0A0DC8", "#0077b6","green"]
        fig, ax = plt.subplots(1,3, figsize=(14.82,5))
        ax[0].loglog(data_epsilon["cov_error"],data_epsilon[time[0]]["empirical"], color=colors[0], lw=width,label=fr"$t={time[0]}$")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
        ax[0].loglog(data_epsilon["cov_error"],data_epsilon[time[0]]["bound"], ls="dashed",lw=width,color=colors[0])#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
        ax[0].loglog(data_epsilon["cov_error"],data_epsilon[time[1]]["empirical"], color=colors[1], lw= width, label=fr"$t={time[1]}$")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
        ax[0].loglog(data_epsilon["cov_error"],data_epsilon[time[1]]["bound"], ls="dashed",lw=width,color=colors[1])#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
        ax[0].loglog(data_epsilon["cov_error"],data_epsilon[time[2]]["empirical"], color=colors[2], lw=width,label=fr"$t={time[2]}$")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
        ax[0].loglog(data_epsilon["cov_error"],data_epsilon[time[2]]["bound"], ls="dashed",lw=width,color=colors[2])#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
        ax[0].set_xlabel(r"$\epsilon$", fontsize=police_xlabel)
        ax[0].set_ylabel("MSE of compositional score", fontsize=police)
        nobs = data_epsilon["nobs"]
        # epsilon_dsm = data_epsilon["epsilon_dsm"]
        epsilon_dsm = 0
        ax[0].set_title(fr"$n= {nobs+1}, \ \ \epsilon_\text{{DSM}}^2={epsilon_dsm}$",fontsize=police)
        ax[0].tick_params(axis='x', labelsize=policeticks)
        ax[0].tick_params(axis='y', labelsize=policeticks)

        ax[1].semilogy(data_nobs["list_nobs"],data_nobs[time[0]]["empirical"], color=colors[0], lw=width,label=fr"$t={time[0]}$")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
        ax[1].semilogy(data_nobs["list_nobs"],data_nobs[time[0]]["bound"], ls="dashed",lw=width,color=colors[0])#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
        ax[1].semilogy(data_nobs["list_nobs"],data_nobs[time[1]]["empirical"], color=colors[1], lw=width, label=fr"$t={time[1]}$")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
        ax[1].semilogy(data_nobs["list_nobs"],data_nobs[time[1]]["bound"], ls="dashed",lw=width,color=colors[1])#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
        ax[1].semilogy(data_nobs["list_nobs"],data_nobs[time[2]]["empirical"], color=colors[2], lw=width,label=fr"$t={time[2]}$")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
        ax[1].semilogy(data_nobs["list_nobs"],data_nobs[time[2]]["bound"], ls="dashed",lw=width,color=colors[2])#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
        ax[1].set_xlabel(r"$n$",fontsize=police_xlabel)
        ax[1].set_xlim(-1,51)
        ax[1].set_xticks(torch.tensor([1,10,20,30,40,50]))
        epsilon_dsm = data_nobs["epsilon_dsm_2"]
        epsilon = data_nobs["epsilon"].item()
        ax[1].set_title(fr"$\epsilon_\text{{DSM}}^2={round(epsilon_dsm,2)}, \ \ \epsilon={round(epsilon,2)}$",fontsize=police)
        ax[1].tick_params(axis='x', labelsize=policeticks)
        ax[1].tick_params(axis='y', labelsize=policeticks)

        ax[2].loglog(data_epsilon_dsm["list_epsilon_dsm_2"],data_epsilon_dsm[time[0]]["empirical"], color=colors[0], lw=width,label=fr"$t={time[0]}$")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
        ax[2].loglog(data_epsilon_dsm["list_epsilon_dsm_2"],data_epsilon_dsm[time[0]]["bound"], ls="dashed",lw=width,color=colors[0])#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
        ax[2].loglog(data_epsilon_dsm["list_epsilon_dsm_2"],data_epsilon_dsm[time[1]]["empirical"], color=colors[1], lw=width, label=fr"$t={time[1]}$")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
        ax[2].loglog(data_epsilon_dsm["list_epsilon_dsm_2"],data_epsilon_dsm[time[1]]["bound"], ls="dashed",lw=width,color=colors[1])#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
        ax[2].loglog(data_epsilon_dsm["list_epsilon_dsm_2"],data_epsilon_dsm[time[2]]["empirical"], color=colors[2], lw=width,label=fr"$t={time[2]}$")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
        ax[2].loglog(data_epsilon_dsm["list_epsilon_dsm_2"],data_epsilon_dsm[time[2]]["bound"], ls="dashed",lw=width,color=colors[2])#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
        nobs = data_epsilon_dsm["nobs"]
        eps = data_epsilon_dsm["eps"]
        ax[2].set_xlabel(r"$\epsilon_{\text{DSM}}^2$", fontsize=police_xlabel)
        ax[2].set_title(fr"$n={nobs+1}, \ \ \epsilon={round(eps.item(),2)}$",fontsize=police)
        ax[2].tick_params(axis='x', labelsize=policeticks)
        ax[2].tick_params(axis='y', labelsize=policeticks)

        ax[2].plot([], [], c='k', ls="dashed", label='theoretical', lw=width)
        ax[2].plot([], [], c='k', ls="solid", label='empirical',lw=width)

        ax[0].grid(which='major', alpha=0.5)
        ax[1].grid(which='major', alpha=0.5)
        ax[2].grid(which='major', alpha=0.5)

        handles, labels = ax[2].get_legend_handles_labels()
        # fig.legend(handles, labels, loc="center bottom", bbox_to_anchor=(1,0.5))
        fig.legend(handles, labels,loc='lower center', ncol=5,bbox_to_anchor=(0.5, -0.01),fontsize=police)

        plt.tight_layout(rect=[0.01, 0.05,1, 1])
        plt.show()
        # dico={}
        # for k, v in data.items():
        #     if isinstance(k, torch.Tensor):
        #         dico[round(k.item(),2)]=v
        #         # print(k.item(), v)   # convert tensor to a Python number
        #     else:
        #         dico[k]=v
        #         # print(k, v)
        # logger.log_artifacts(dico, artifact_name="epsilon_evol_seed_0",
                            # artifact_type='pickle')

    if 0: #evol in epsilon_dsm estimated cov corrigée
        dico={}
        n_obs = cfg.nextra
        dico["nobs"]=n_obs
        list_error = torch.linspace(0.01,1.75,50)#0.5,0.7]
        dico["list_epsilon_dsm_2"]=list_error**2
        extra_obs = [x_o]
        for _ in range(n_obs):
            extra_obs.append(simulator1(beta_o))
        x_o_long_full = torch.stack(extra_obs)
        print("\nDimensions for ", n_obs+1, " observations: \n x : ",x_o_long_full.size())
        eps_lda=0.0#01 #error of prior precision
        error_lda = 0.0#1 #error for prior score
        fig3,ax3 = plt.subplots(figsize=(6.5,4.5))
        fig3.suptitle(fr"$\epsilon_{{\lambda}}={eps_lda}, \ \ n= {n_obs+1}, \ \ \epsilon_{{\text{{DSM}},\lambda}}^2\leq {round(error_lda**2,3)}$")# \ \ \epsilon\leq{round(eps_max.item(),3)}$")
        colors = ["red","blue","green"]
        list_eps = []
        tensor_prec = {} #store the estimated prec at time 0
        for k, error in enumerate(list_error):
            list_prec=[] #store covariance of **perturbed** indiv posterior
            perturbed_score.error=error
            eps_max_list=[]
            for xi in range(n_obs+1):
                samples_beta = perturbed_score.sampling(num_samples,steps=200, condition=x_o_long_full[xi,:][None,:])[:,0,:] #size (num_samples, 1, dim)
                cov_est = torch.cov(samples_beta.T)
                prec_est = torch.linalg.inv(cov_est)
                list_prec.append(prec_est)
                eps_tmp = torch.linalg.norm(prec_est-prec_post,2)
                list_prec.append(prec_est) #Sigma tilde
                eps_max_list.append(eps_tmp.item())
            eps = torch.max(torch.tensor(eps_max_list))
            list_eps.append(eps.item())
            tensor_prec[k]=list_prec
        eps_max = torch.max(torch.tensor(list_eps))
        dico["eps"]=eps_max
        print("hey", eps_max)
        for j, time in enumerate(torch.tensor([0.1,0.5,0.9])):#linspace(0.1,1.0,9)):
            dico_tmp={}
            tall_error_list=[]
            tall_error_list_sq = []
            error_bound_list_sq=[]
            error_bound_list=[]
            time_score = []
            perturbed_score.error=0.0
            alpha_t = perturbed_score.mean_t(time)**2
            true_samples = torch.distributions.MultivariateNormal(alpha_t**0.5*mu_tall_post(x_o_long_full).squeeze(1), alpha_t*cov_tall_post(x_o_long_full)+(1-alpha_t)*torch.eye(DIM)).sample((num_samples,))
            diff_prec = perturbed_score.diff_prec(prec_post,time) #true diffused post precisions
            M = torch.linalg.norm(diff_prec,2) #all the cov are equal in our case
            diff_prec_prior = perturbed_score.diff_prec(torch.linalg.inv(cov_prior),time) #true diffused prior precision
            M_lda = torch.linalg.norm(diff_prec_prior,2)
            Lda = torch.linalg.inv(-n_obs*diff_prec_prior+(n_obs+1)*diff_prec)
            norm = torch.linalg.norm(Lda,2)
            true_diff_tall_score = perturbed_score.tall_posterior_score(true_samples[:,None,:], x_o_long_full,time,[diff_prec for _ in range(n_obs+1)],diff_prec_prior)
            for k, error in enumerate(list_error):
                list_prec_t = []
                if n_obs*eps_lda+(n_obs+1)*eps_max<1/norm :#and eps_lda<1/M_lda and eps<1/M:
                    L_list = []
                    perturbed_score.error=0.0
                    for i in range(n_obs+1):
                        norm_indiv_score = torch.linalg.norm(perturbed_score.single_posterior_score(true_samples[:,None,:],x_o_long_full[i,:],time),2, dim=(1,2))
                        L_list.append(torch.mean(norm_indiv_score**2))
                        list_prec_t.append(perturbed_score.diff_prec(tensor_prec[k][i],time))
                    L_sq = torch.max(torch.tensor(L_list))
                    L_lda_sq = torch.mean(torch.linalg.norm(perturbed_score.prior_score(true_samples[:,None,:],time),2,dim=(1,2))**2) #nsamples,dim,1
                    perturbed_score.error = error
                    est_diff_tall_score = perturbed_score.tall_posterior_score(true_samples[:,None,:], x_o_long_full,time,list_prec_t,diff_prec_prior+eps_lda*torch.eye(DIM),error_lda)
                    tall_error_sq=torch.linalg.norm(true_diff_tall_score-est_diff_tall_score, 2, dim=(1,2))**2

                    error_bound_sq = n_obs*(L_lda_sq**0.5+error_lda)*norm
                    frac = (n_obs*eps_lda+(n_obs+1)*eps_max)*norm/(1-norm*(n_obs*eps_lda+(n_obs+1)*eps_max))
                    error_bound_sq*=eps_lda+frac*(M_lda+eps_lda)#*M_lda/(1-M_lda*eps_lda)
                    error_bound_sq+=(n_obs+1)*(L_sq**0.5+error)*norm*(eps_max+frac*(M+eps_max))#*M/(1-M*eps))
                    error_bound_sq += norm*(n_obs*M_lda*error_lda+(n_obs+1)*M*error)

                    tall_error_list_sq.append(torch.mean(tall_error_sq).item())
                    error_bound_list_sq.append((error_bound_sq**2).item())

            dico_tmp["bound"]=error_bound_list_sq
            dico_tmp["empirical"]=tall_error_list_sq
            dico[round(time.item(),1)]=dico_tmp
            ax3.loglog(list_error**2,tall_error_list_sq, color=colors[j], label=fr"$t={round(time.item(),1)}, \epsilon={round(eps_max.item(),3)}$")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
            ax3.loglog(list_error**2,error_bound_list_sq, ls="dashed",color=colors[j])#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
            # ax3.loglog(list_error**2,error_bound_list, ls="dashed",color=colors[j])#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
            # ax3.loglog(list_error**2,list_error**2, ls="dotted",color="orange")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
            # ax3.loglog(list_error**2,list_error+list_error**2, ls="dotted",color="brown")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
            ax3.set_xlabel(r"$\epsilon_{\text{DSM}}^2$")

            # ax2[j].loglog(list_error,tall_error_list_sq, color="blue", label="empirical squared error")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
            # ax2[j].loglog(list_error,error_bound_list_sq, ls="dotted",color="blue", label="theoretical squared bound")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")

            # ax2[j].loglog(list_error,tall_error_list, color="red", label="empirical error")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
            # ax2[j].loglog(list_error,error_bound_list, ls="dotted",color="red", label="theoretical bound")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
            # ax2[j].loglog(list_error,list_error**2, ls="dotted",color="green", label=r"$\epsilon_\text{DSM}^4$")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
            # ax2[j].loglog(list_error,list_error, ls="dotted",color="orange", label=r"$\epsilon_\text{DSM}^2$")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")

            # ax2[j].set_xlabel(r"$\epsilon_\text{DSM}^2$")
            # ax2[j].legend()
            # ax2[j].set_title(fr"$t={round(time.item(),3)}$")
            # ax1[j].plot(list_error**2,tall_error_list, color="red", label="error")#label=r"$E_{\nu_t}||\nabla \log \nu_t(\theta\mid x_{1:n})-s(\theta,x_{1:n},t)||$")
        logger.log_artifacts(dico, artifact_name=f"evol_epsilon_dsm_seed_{cfg.seed}",
                            artifact_type='pickle')
        ax3.plot([], [], c='k', ls="dashed", label='theoretical')
        ax3.plot([], [], c='k', ls="solid", label='empirical')
        handles, labels = ax3.get_legend_handles_labels()
        fig3.legend(handles, labels, loc="center right", bbox_to_anchor=(1,0.5))
        plt.tight_layout(rect=[0, 0, 0.8, 1])
        plt.show()


if __name__ == "__main__":
    main()

    # import pickle
    # samples = pickle.load(open(f"/home/ctouron/codedev/sbi_hackathon/HNPE_diff/logs/gaussian_ex_ve_nocorrect_11_obs/artifacts/pickle/11_obs_alpha_0.5_beta_0.5_samples","rb"))
    # samples_gauss = samples["samples_beta_gauss"]
    # samples_auto = samples["samples_beta_auto"]
    # samples_hunch = samples["samples_beta_hunch"]
    # samples_fnpe = samples["samples_beta_fnpe"]
    # samples_jac = pickle.load(open(f"/home/ctouron/codedev/sbi_hackathon/HNPE_diff/logs/4/artifacts/pickle/11_obs_alpha_0.5_beta_0.5_samples","rb"))["samples_beta_jac"]
    # fig = plt.figure(figsize=(12,8))
    # plt.subplot(121)
    # sns.kdeplot(samples_gauss[:,0], color="orange", label="GAUSS")
    # sns.kdeplot(samples_auto[:,0], color="green", label="auto")
    # sns.kdeplot(samples_fnpe[:,0], color="grey", label="Geffner")
    # sns.kdeplot(samples_jac[:,0], color="pink", label="JAC")
    # sns.kdeplot(samples_hunch[:,0], color="blue", label="Hunch")

    # # sns.kdeplot(true_samples[:,0], color="red", label="true")
    # plt.legend()
    # # plt.title(fr"$p(\beta_1|x_0,...,x_{{{n_obs}}})$")

    # plt.subplot(122)
    # sns.kdeplot(samples_gauss[:,1], color="orange", label="GAUSS")
    # sns.kdeplot(samples_auto[:,1], color="green", label="auto")
    # sns.kdeplot(samples_fnpe[:,1], color="grey", label="Geffner")
    # sns.kdeplot(samples_jac[:,1], color="pink", label="JAC")
    # sns.kdeplot(samples_hunch[:,1], color="blue", label="Hunch")

    # # sns.kdeplot(true_samples[:,1], color="red", label="true")
    # plt.legend()
    # # plt.title(fr"$p(\beta_2|x_0,...,x_{{{n_obs}}})$")
    # plt.show()
