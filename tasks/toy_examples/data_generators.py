import torch
import numpy as np

from scipy.stats import norm
from torch.distributions.multivariate_normal import MultivariateNormal

class SBIGaussian2d:
    def __init__(self, prior, x_correlation_factor=0.8) -> None:
        """2d Gaussian.
        Inference of mean under given prior.
        """
        self.prior = prior
        self.x_correlation_factor = x_correlation_factor

    def simulator(self, theta):
        # Distribution parameters
        rho = self.x_correlation_factor
        cov = torch.FloatTensor([[1, rho], [rho, 1]])
        # Sample X
        samples_x = MultivariateNormal(loc=theta, covariance_matrix=cov).sample((1,))[0]
        return samples_x

    def get_joint_data(self, n):
        samples_prior = self.prior.sample((n,))
        samples_x = self.simulator(samples_prior)
        return samples_prior, samples_x

    def true_posterior_pdf(self, x_obs):
        def true_posterior_prob(samples):
            log_p_prior = self.prior.log_prob(samples)
            rho = self.x_correlation_factor
            cov = torch.FloatTensor([[1, rho], [rho, 1]])
            if x_obs.ndim > 1:
                log_p_x = MultivariateNormal(
                    loc=samples, covariance_matrix=cov
                ).log_prob(torch.mean(x_obs, axis=0))
            else:
                log_p_x = MultivariateNormal(
                    loc=samples, covariance_matrix=cov
                ).log_prob(x_obs)
            log_p = log_p_prior + log_p_x
            return get_proba(log_p)
        return true_posterior_prob

    def true_posterior(self, x_obs):
        rho = self.x_correlation_factor
        cov = torch.FloatTensor([[1, rho], [rho, 1]])
        return MultivariateNormal(loc=x_obs, covariance_matrix=cov)


def get_proba(log_p):
    """Compute probability from log_prob in a safe way.

    Parameters
    ----------
    log_p: ndarray, shape (n_samples*n_samples,)
        Values of log_p for each point (a,b) of the samples (n_samples, n_samples).
    """
    if isinstance(log_p, np.ndarray):
        log_p = torch.tensor(log_p)
    log_p = log_p.to(dtype=torch.float64)
    log_p -= torch.logsumexp(log_p, dim=-1)
    return torch.exp(log_p)
