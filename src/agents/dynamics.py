import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as torch_dist
from src.agents.nn_models import EnsembleMLP
from src.agents.rl_utils import normalize, denormalize

def soft_clamp(x, _min, _max):
    x = _max - F.softplus(_max - x)
    x = _min + F.softplus(x - _min)
    return x

class EnsembleDynamics(nn.Module):
    def __init__(
        self,
        obs_dim,
        act_dim,
        out_dim,
        ensemble_dim, 
        topk,
        hidden_dim, 
        num_hidden, 
        activation, 
        decay=None, 
        clip_lv=False, 
        residual=False,
        termination_fn=None, 
        min_std=1e-5,
        max_std=1.6,
        device=torch.device("cpu")
        ):
        """
        Args:
            obs_dim (int): observation dimension
            act_dim (int): action dimension
            out_dim (int): outupt dimension. Use 1 for reward prediction
            ensemble_dim (int): number of ensemble models
            topk (int): top k models to perform rollout
            hidden_dim (int): value network hidden dim
            num_hidden (int): value network hidden layers
            activation (str): value network activation
            decay ([list, None], optional): weight decay for each dynamics and reward model layer. Default=None.
            clip_lv (bool, optional): whether to soft clip observation log variance. Default=False
            residual (bool, optional): whether to predict observation residuals. Default=False
            termination_fn (func, optional): termination function to output rollout done. Default=None
            min_std (float): minimum standard deviation. Default=1e-5
            max_std (float): maximum standard deviation. Default=1e-5
            device (torch.device): computing device. default=cpu
        """
        super().__init__()
        if decay is None:
            decay = [0. for _ in range(num_hidden + 2)]
        
        assert len(decay) == num_hidden + 2
        if residual:
            assert out_dim == obs_dim

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.out_dim = out_dim
        self.ensemble_dim = ensemble_dim
        self.topk = topk
        self.decay = decay
        self.clip_lv = clip_lv
        self.residual = residual
        self.termination_fn = termination_fn
        self.device = device
        
        self.mlp = EnsembleMLP(
            obs_dim + act_dim, 
            out_dim * 2, 
            ensemble_dim, 
            hidden_dim, 
            num_hidden, 
            activation
        )

        topk_dist = torch.ones(ensemble_dim) / ensemble_dim # model selection distribution
        self.topk_dist = nn.Parameter(topk_dist, requires_grad=False)
        self.min_lv = nn.Parameter(np.log(min_std) * torch.ones(out_dim), requires_grad=False)
        self.max_lv = nn.Parameter(np.log(max_std) * torch.ones(out_dim), requires_grad=False)

        self.obs_mean = nn.Parameter(torch.zeros(obs_dim), requires_grad=False)
        self.obs_variance = nn.Parameter(torch.ones(obs_dim), requires_grad=False)
        self.out_mean = nn.Parameter(torch.zeros(out_dim), requires_grad=False)
        self.out_variance = nn.Parameter(torch.ones(out_dim), requires_grad=False)
    
    def update_stats(self, obs_mean, obs_variance, out_mean, out_variance):
        self.obs_mean.data = torch.from_numpy(obs_mean).to(torch.float32).to(self.device)
        self.obs_variance.data = torch.from_numpy(obs_variance).to(torch.float32).to(self.device)
        self.out_mean.data = torch.from_numpy(out_mean).to(torch.float32).to(self.device)
        self.out_variance.data = torch.from_numpy(out_variance).to(torch.float32).to(self.device)

    def compute_dist(self, obs, act):
        """ Compute normalized output distribution class """
        obs_act = torch.cat([obs, act], dim=-1)
        mu_, lv = torch.chunk(self.mlp.forward(obs_act), 2, dim=-1)
        
        if self.residual:
            mu = obs.unsqueeze(-2) + mu_
        else:
            mu = mu_

        if self.clip_lv:
            std = torch.exp(soft_clamp(lv, self.min_lv, self.max_lv))
        else:
            std = torch.exp(lv.clip(self.min_lv.data, self.max_lv.data))
        return torch_dist.Normal(mu, std)
    
    def compute_log_prob(self, obs, act, target):
        """ Compute ensemble log probability 
        
        Args:
            obs (torch.tensor): normalized observations. size=[..., obs_dim]
            act (torch.tensor): actions. size=[..., act_dim]
            target (torch.tensor): normalized targets. size=[..., out_dim]

        Returns:
            out (torch.tensor): ensemble log probabilities of normalized targets. size=[..., ensemble_dim, 1]
        """
        return self.compute_dist(obs, act).log_prob(target.unsqueeze(-2)).sum(-1, keepdim=True)
    
    def compute_mixture_log_prob(self, obs, act, target):
        """ Compute log marginal probability 
        
        Args:
            obs (torch.tensor): normalized observations. size=[..., obs_dim]
            act (torch.tensor): actions. size=[..., act_dim]
            target (torch.tensor): normalized targets. size=[..., out_dim]

        Returns:
            mixture_log_prob (torch.tensor): log marginal probabilities of normalized targets. size=[..., 1]
        """
        log_elites = torch.log(self.topk_dist + 1e-6).unsqueeze(-1)
        log_prob = self.compute_log_prob(obs, act, target)
        mixture_log_prob = torch.logsumexp(log_prob + log_elites, dim=-2)
        return mixture_log_prob
    
    def sample_dist(self, obs, act):
        """ Sample from ensemble 
        
        Args:
            obs (torch.tensor): normalized observations. size=[..., obs_dim]
            act (torch.tensor): normaized actions. size=[..., act_dim]

        Returns:
            out (torch.tensor): normalized output sampled from ensemble member in topk_dist. size=[..., out_dim]
        """
        out_dist = self.compute_dist(obs, act)
        out = out_dist.rsample()
        
        # randomly select from top models
        ensemble_idx = torch_dist.Categorical(self.topk_dist).sample(obs.shape[:-1]).unsqueeze(-1)
        ensemble_idx_ = ensemble_idx.unsqueeze(-1).repeat_interleave(self.out_dim, dim=-1).to(self.device) # duplicate alone feature dim
        out = torch.gather(out, -2, ensemble_idx_).squeeze(-2)
        return out
    
    def step(self, obs, act):
        """ Simulate a step forward with normalization pre and post processing
        
        Args:
            obs (torch.tensor): unnormalized observations. size=[..., obs_dim]
            act (torch.tensor): unnormaized actions. size=[..., act_dim]

        Returns:
            out (torch.tensor): sampled unnormalized outputs. size=[..., out_dim]
            done (torch.tensor): done flag. If termination_fn is None, return all zeros. size=[..., 1]
        """
        obs_norm = normalize(obs, self.obs_mean, self.obs_variance)
        out_norm = self.sample_dist(obs_norm, act)
        out = denormalize(out_norm, self.out_mean, self.out_variance)

        if self.termination_fn is not None:
            done = self.termination_fn(
                obs.data.cpu().numpy(), act.data.cpu().numpy(), out.data.cpu().numpy()
            )
            done = torch.from_numpy(done).unsqueeze(-1).to(torch.float32)
        else:
            done = torch.zeros(list(obs.shape)[:-1] + [1]).to(torch.float32)
        return out, done
    
    def compute_loss(self, obs, act, target):
        """ Compute log likelihood for normalized data and weight decay loss """
        logp = self.compute_log_prob(obs, act, target).sum(-1)
        decay_loss = self.compute_decay_loss()
        loss = -logp.mean() + decay_loss
        return loss
    
    def compute_decay_loss(self):
        i, loss = 0, 0
        for layer in self.mlp.layers:
            if hasattr(layer, "weight"):
                loss += self.decay[i] * torch.sum(layer.weight ** 2) / 2.
                i += 1
        return loss
    
    def evaluate(self, obs, act, target):
        """ Compute mean average error of normalized data for each ensemble 
        
        Returns:
            stats (dict): MAE dict with fields [mae_0, ..., mae_{ensemble_dim}, mae]
        """
        with torch.no_grad():
            out_pred = self.compute_dist(obs, act).mean
        
        obs_mae = torch.abs(out_pred - target.unsqueeze(-2)).mean((0, 2))
        
        stats = {f"mae_{i}": obs_mae[i].cpu().data.item() for i in range(self.ensemble_dim)}
        stats["mae"] = obs_mae.mean().cpu().data.item()
        return stats
    
    def update_topk_dist(self, stats):
        """ Update top k model selection distribution """
        maes = [v for k, v in stats.items() if "mae_" in k]
        idx_topk = np.argsort(maes)[:self.topk]
        topk_dist = np.zeros(self.ensemble_dim)
        topk_dist[idx_topk] = 1./self.topk
        self.topk_dist.data = torch.from_numpy(topk_dist).to(torch.float32).to(self.device)

if __name__ == "__main__":
    torch.manual_seed(0)
    from src.env.gym_wrapper import get_termination_fn
    
    obs_dim = 11
    act_dim = 3
    ensemble_dim = 7
    topk = 5
    hidden_dim = 200
    num_hidden = 2
    activation = "relu"
    decay = [0.000025, 0.00005, 0.000075, 0.0001]
    min_std = 1e-5
    max_std = 0.6
    termination_fn = get_termination_fn("Hopper-v4")

    # synthetic data
    batch_size = 1000
    obs = torch.randn(batch_size, obs_dim)
    act = torch.randn(batch_size, act_dim)
    rwd = torch.randn(batch_size, 1)
    next_obs = torch.randn(batch_size, obs_dim)
    
    def test_soft_clamp():
        _min = torch.Tensor([-1.])
        _max = torch.Tensor([1.])
        x1 = torch.Tensor([0.])
        x2 = torch.Tensor([1.5])
        x3 = torch.Tensor([-1.5])
        assert _min < soft_clamp(x1, _min, _max) < _max
        assert _min < soft_clamp(x2, _min, _max) < _max
        assert _min < soft_clamp(x3, _min, _max) < _max

    def test_ensemble(clip_lv=False, residual=False, termination_fn=None, test_reward=False):
        out_dim = obs_dim
        target = next_obs
        if test_reward:
            out_dim = 1
            target = rwd
            termination_fn = None

        dynamics = EnsembleDynamics(
            obs_dim,
            act_dim, 
            out_dim,
            ensemble_dim,
            topk,
            hidden_dim,
            num_hidden,
            activation,
            decay=decay,
            clip_lv=clip_lv,
            residual=residual,
            termination_fn=termination_fn,
            min_std=min_std,
            max_std=max_std
        )

        # test internal variable shapes
        assert list(dynamics.topk_dist.shape) == [ensemble_dim]
        assert list(dynamics.min_lv.shape) == [out_dim]
        assert list(dynamics.max_lv.shape) == [out_dim]
        assert list(dynamics.obs_mean.shape) == [obs_dim]
        assert list(dynamics.obs_variance.shape) == [obs_dim]
        assert list(dynamics.out_mean.shape) == [out_dim]
        assert list(dynamics.out_variance.shape) == [out_dim]
        assert torch.isclose(dynamics.min_lv.exp(), min_std * torch.ones(1), atol=1e-5).all()
        assert torch.isclose(dynamics.max_lv.exp(), max_std * torch.ones(1), atol=1e-5).all()

        out_dist = dynamics.compute_dist(obs, act)
        logp =  dynamics.compute_log_prob(obs, act, target)
        mix_logp = dynamics.compute_mixture_log_prob(obs, act, target)
        out_sample = dynamics.sample_dist(obs, act)
        out_step, done = dynamics.step(obs, act)
        
        # test method output shapes
        assert list(out_dist.mean.shape) == [batch_size, ensemble_dim, out_dim]
        assert list(out_dist.variance.shape) == [batch_size, ensemble_dim, out_dim]
        assert list(logp.shape) == [batch_size, ensemble_dim, 1]
        assert list(mix_logp.shape) == [batch_size, 1]
        assert list(out_sample.shape) == [batch_size, out_dim]
        assert list(out_step.shape) == [batch_size, out_dim]
        assert list(done.shape) == [batch_size, 1]
        if test_reward or termination_fn is None:
            assert done.sum() == 0
        
        # test backward
        loss = dynamics.compute_loss(obs, act, target)
        loss.backward()
        for n, p in dynamics.named_parameters():
            if "weight" in n or "bias" in n:
                assert p.grad is not None
                p.grad *= 0.
        
        # test eval
        stats = dynamics.evaluate(obs, act, target)
        dynamics.update_topk_dist(stats)
        assert sum(dynamics.topk_dist == 0) == (dynamics.ensemble_dim - dynamics.topk)

    test_soft_clamp()
    print("soft_clamp passed")
    
    # test dynamics
    test_ensemble(
        clip_lv=True, 
        residual=True, 
        termination_fn=termination_fn,
        test_reward=False
    )
    
    test_ensemble(
        clip_lv=True, 
        residual=False, 
        termination_fn=termination_fn,
        test_reward=False
    )

    test_ensemble(
        clip_lv=False, 
        residual=False, 
        termination_fn=None,
        test_reward=False
    )

    print("dynamics passed")

    # test dynamics
    test_ensemble(
        clip_lv=True, 
        residual=False, 
        termination_fn=termination_fn,
        test_reward=True
    )

    test_ensemble(
        clip_lv=False, 
        residual=False, 
        termination_fn=None,
        test_reward=True
    )

    print("reward passed")