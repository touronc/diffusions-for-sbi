import torch

from sbi.utils import BoxUniform
from vp_diffused_priors import get_vpdiff_uniform_score

bounds = {
    0: [-10, 10],
    1: [100, 250],
}


class ToyPrior:
    def __init__(self, bounds=bounds):
        self.bounds = bounds
        self.low = torch.tensor([self.bounds[k][0] for k in [0, 1]])
        self.high = torch.tensor([self.bounds[k][1] for k in [0, 1]])
        self.prior = BoxUniform(
            low=self.low,
            high=self.high,
        )

    def sample(self, sample_shape):
        return self.prior.sample(sample_shape)


if __name__ == "__main__":
    prior = ToyPrior()
    theta_t = prior.sample((10000,))
    t = torch.zeros(10000, 1)
    diffused_prior_score = get_vpdiff_uniform_score(prior.low, prior.high)
    prior_score_t = diffused_prior_score(theta_t, t)
    print(prior_score_t.shape)
    print(prior_score_t.dtype)
