import torch
from torch import Tensor
from zuko.utils import broadcast



class PositionalEncodingVector(torch.nn.Module):
     # for the time variable t
     # inspired from the sinusoidal time embedding in DDPM+
    def __init__(self, d_model: int, M: int):
        super().__init__()
        div_term = 1 / M ** (2 * torch.arange(0, d_model, 2) / d_model) #size of torch.arange
        self.register_buffer("div_term", div_term)
    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        exp_div_term = self.div_term.reshape(*(1,) * (len(x.shape) - 1), -1) #reshape divterm as (1,dim(divterm))
        tmp = exp_div_term * x[...,None]
        #ATTENTION modification
        return torch.cat(
            (torch.sin(tmp), torch.cos(tmp)), dim=-1
        )
        # return torch.cat(
        #     (torch.sin(x * exp_div_term), torch.cos(x * exp_div_term)), dim=-1
        # ) #size = (nsample of x, dim divterm *2)


class _FBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        emb_channels,
        dropout=0,
        eps=1e-5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.dropout = dropout

        self.norm0 = torch.nn.GroupNorm(
            num_groups=16, num_channels=in_channels, eps=eps
        )
        self.conv0 = torch.nn.Linear(in_features=in_channels, out_features=out_channels)
        self.affine = torch.nn.Linear(
            in_features=emb_channels, out_features=out_channels
        )
        self.norm1 = torch.nn.GroupNorm(
            num_groups=16, num_channels=out_channels, eps=eps
        )
        self.conv1 = torch.nn.Linear(
            in_features=out_channels, out_features=out_channels
        )

        self.skip = None
        if out_channels != in_channels:
            self.skip = torch.nn.Linear(
                in_features=in_channels, out_features=out_channels
            )
        self.skip_scale = 0.5**0.5

    def forward(self, x, emb):
        silu = torch.nn.functional.silu
        orig = x
        x = self.conv0(silu(self.norm0(x)))

        params = self.affine(emb)

        x = silu(self.norm1(x.add_(params)))

        x = self.conv1(
            torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        )
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale
        return x


class FNet(torch.nn.Module):
    def __init__(self, dim_input, dim_cond, dim_embedding=512, n_layers=3):
        super().__init__()
        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(dim_input, dim_embedding),
            # torch.nn.SiLU(),
            # torch.nn.Linear
        )
        self.cond_layer = torch.nn.Linear(dim_cond, dim_embedding)
        self.embedding_map = torch.nn.Sequential(
            torch.nn.Linear(dim_embedding, 2 * dim_embedding),
            torch.nn.GroupNorm(num_groups=16, num_channels=2 * dim_embedding),
            torch.nn.SiLU(),
            torch.nn.Linear(2 * dim_embedding, dim_embedding),
        )
        self.res_layers = torch.nn.ModuleList(
            [
                _FBlock(
                    in_channels=dim_embedding // 2**i,
                    out_channels=dim_embedding // 2 ** (i + 1),
                    emb_channels=dim_embedding,
                    dropout=0.1,
                )
                for i in range(n_layers)
            ]
        )
        self.final_layer = torch.nn.Sequential(
            torch.nn.GroupNorm(
                num_groups=16, num_channels=dim_embedding // 2 ** (n_layers)
            ),
            torch.nn.SiLU(),
            torch.nn.Linear(dim_embedding // 2 ** (n_layers), dim_input, bias=False),
        )
        self.time_embedding = PositionalEncodingVector(d_model=dim_embedding, M=1000)

    def forward(self, theta, x, t):
        if isinstance(t, int):
            t = torch.tensor([t], device=theta.device)
        theta_emb = self.input_layer(theta)
        x_emb = self.cond_layer(x)
        t_emb = self.time_embedding(t)
        emb = self.embedding_map(t_emb + x_emb)
        for lr in self.res_layers:
            theta_emb = lr(x=theta_emb, emb=emb)
        return self.final_layer(theta_emb).reshape(*theta.shape)  # - theta


# Noisy network for experiments of section 4.2


class EpsilonNet(torch.nn.Module):
    def __init__(self, DIM):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2 * DIM + 1, 5 * DIM),
            torch.nn.ReLU(),
            torch.nn.Linear(5 * DIM, DIM),
            torch.nn.Tanh(),
        )

    def forward(self, theta, x, t):
        return self.net(torch.cat((theta, x, t), dim=-1))


class FakeFNet(torch.nn.Module):
    def __init__(self, real_eps_fun, eps_net, eps_net_max):
        r"""Fake score network that returns the real score plus a perturbation

        Args:
            real_eps_fun (callable): function that returns the real score network (analytic or trained)
            eps_net (torch.nn.Module): perturbation network (randomly initialized, not trained)
            eps_net_max (float): scaling factor for the perturbation
        """
        super().__init__()
        self.real_eps_fun = real_eps_fun
        self.eps_net = eps_net
        self.eps_net_max = eps_net_max

    def forward(self, theta, x, t):
        if len(t.shape) == 0:
            t = t[None, None].repeat(theta.shape[0], 1)
        real_eps = self.real_eps_fun(theta, x, t)
        perturb = self.eps_net(theta, x, t)
        return real_eps + self.eps_net_max * perturb


class GaussianNet(torch.nn.Module):
    """
    Score network that mimics the true shape of the posterior score and learn 3 matrices A_t, B_t, C_t
    The forward path looks like A_t(theta + B_t x + C_t)
    """
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.output_dim=output_dim
        self.mu_prior = torch.ones(2)
        self.inv_cov_prior=torch.eye(2)*1.0/3
        self.inv_cov_lik=torch.eye(2)*1.0/2
        self.cov_post=torch.linalg.inv(self.inv_cov_prior+self.inv_cov_lik)
        
        self.A_t = torch.nn.Sequential(
            torch.nn.Linear(1, hidden_dim, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim*output_dim, bias=True),
            
        )
        self.B_t = torch.nn.Sequential(
            torch.nn.Linear(1, hidden_dim, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim*output_dim, bias=True),
        )
        self.C_t = torch.nn.Sequential(
            torch.nn.Linear(1, hidden_dim, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim, bias=True),        
        )

    def est_matrices(self,t):
        return self.A_t(t).detach(), self.B_t(t).detach(), self.C_t(t).detach()

    def alpha_t(self,t):
        log_alpha = 0.5 * 39.9 * (t**2) + 0.1 * t
        return torch.exp(-log_alpha)
    
    def forward(self, theta, x, t):
        t_tmp=t
        cut=False # to adapt the size of the return score to the input variables
        if t.ndim==0:
            t_tmp=torch.tensor([t])
        t_tmp=t_tmp.unsqueeze(1)

        if theta.shape[0]!=x.shape[0]:
            theta, x = broadcast(theta, x, ignore=1)
        
        if theta.ndim==1:
            theta=theta.unsqueeze(0)
            x=x.unsqueeze(0)
            cut=True

        alpha=self.alpha_t(t_tmp)
        A=(self.A_t(t_tmp)).reshape(t_tmp.shape[0],self.output_dim,self.output_dim)
        #size (batch_size,2,2)
        #A=(1-alpha)[...,None]**0.5*torch.linalg.inv((1-alpha)[...,None]*(torch.eye(2).repeat(t_tmp.shape[0],1,1))+alpha[...,None]*(self.cov_post.repeat(t_tmp.shape[0],1,1)))
        B=(self.B_t(t_tmp)).reshape(t_tmp.shape[0],self.output_dim,self.output_dim)
        #size (batch_size,2,2)
        #B = -self.alpha_t(t_tmp)[...,None]**0.5 * (self.cov_post@self.inv_cov_lik).repeat(t_tmp.shape[0],1,1)
        C = self.C_t(t_tmp)
        #size (batch_size,2)
        #C = -self.alpha_t(t_tmp)**0.5*(self.cov_post@self.inv_cov_prior@self.mu_prior).repeat(t_tmp.shape[0],1)
        score = A@((theta[...,None])[:,:,0] + (B@x[...,None])[:,:,0] + C)[...,None]
        # size(batch size,2)
    
        if cut:
            return score[0,:,0]
        else:
            return score[:,:,0]


class GaussianNetAlpha(torch.nn.Module):
    """
    Score network that only learns the function alpha(t)
    and return the score with this learnt function
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.mu_prior = torch.ones(2)
        self.inv_cov_prior=torch.eye(2)*1.0/3
        self.inv_cov_lik=torch.eye(2)*1.0/2
        self.cov_post=torch.linalg.inv(self.inv_cov_prior+self.inv_cov_lik)
        
        self.scaling_alpha = torch.nn.Sequential(
            torch.nn.Linear(1, hidden_dim, bias=True),
            #torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim, bias=True),
            #torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1, bias=True),
            torch.nn.Sigmoid()
        )
        
    def forward(self, theta, x, t):
        t_tmp=t
        cut=False # to adapt the size of the return score to the input variables
        if t.ndim==0:
            t_tmp=torch.tensor([t])
        t_tmp=t_tmp.unsqueeze(1)

        if theta.shape[0]!=x.shape[0]:
            theta, x = broadcast(theta, x, ignore=1)
        
        if theta.ndim==1:
            theta=theta.unsqueeze(0)
            x=x.unsqueeze(0)
            cut=True

        alpha=self.scaling_alpha(t_tmp)
        #size (batch_size,2,2)
        A=(1-alpha)[...,None]**0.5*torch.linalg.inv((1-alpha)[...,None]*(torch.eye(2).repeat(t_tmp.shape[0],1,1))+alpha[...,None]*(self.cov_post.repeat(t_tmp.shape[0],1,1)))
        #size (batch_size,2,2)
        B = -alpha[...,None]**0.5 * (self.cov_post@self.inv_cov_lik).repeat(t_tmp.shape[0],1,1)
        #size (batch_size,2)
        C = -alpha**0.5*(self.cov_post@self.inv_cov_prior@self.mu_prior).repeat(t_tmp.shape[0],1)
        score = A@((theta[...,None])[:,:,0] + (B@x[...,None])[:,:,0] + C)[...,None]
        # size(batch size,2)
        if cut:
            return score[0,:,0]
        else:
            return score[:,:,0]