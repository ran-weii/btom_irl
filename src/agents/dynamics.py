import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as torch_dist
from src.agents.nn_models import EnsembleMLP
from src.utils.data import normalize, denormalize
from src.utils.logger import Logger

def soft_clamp(x, _min, _max):
    x = _max - F.softplus(_max - x)
    x = _min + F.softplus(x - _min)
    return x

class EnsembleDynamics(nn.Module):
    def __init__(
        self,
        obs_dim,
        act_dim,
        pred_rwd,
        ensemble_dim, 
        topk,
        hidden_dim, 
        num_hidden, 
        activation, 
        decay=None, 
        clip_lv=False, 
        residual=False,
        termination_fn=None, 
        min_std=0.04,
        max_std=1.6,
        device=torch.device("cpu")
        ):
        """
        Args:
            obs_dim (int): observation dimension
            act_dim (int): action dimension
            pred_rwd (bool): whether to predict reward. used to set loss function
            ensemble_dim (int): number of ensemble models
            topk (int): top k models to perform rollout
            hidden_dim (int): value network hidden dim
            num_hidden (int): value network hidden layers
            activation (str): value network activation
            decay ([list, None], optional): weight decay for each dynamics and reward model layer. Default=None.
            clip_lv (bool, optional): whether to soft clip observation log variance. Default=False
            residual (bool, optional): whether to predict observation residuals. Default=False
            termination_fn (func, optional): termination function to output rollout done. Default=None
            min_std (float, optional): minimum standard deviation. Default=0.04
            max_std (float, optional): maximum standard deviation. Default=1.6
            device (torch.device): computing device. default=cpu
        """
        super().__init__()
        if decay is None:
            decay = [0. for _ in range(num_hidden + 2)]
        
        assert len(decay) == num_hidden + 2
        
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.out_dim = obs_dim + 1
        self.ensemble_dim = ensemble_dim
        self.topk = topk
        self.decay = decay
        self.pred_rwd = pred_rwd
        self.clip_lv = clip_lv
        self.residual = residual
        self.termination_fn = termination_fn
        self.device = device
        
        self.mlp = EnsembleMLP(
            obs_dim + act_dim, 
            self.out_dim * 2, 
            ensemble_dim, 
            hidden_dim, 
            num_hidden, 
            activation
        )

        topk_dist = torch.ones(ensemble_dim) / ensemble_dim # model selection distribution
        self.topk_dist = nn.Parameter(topk_dist, requires_grad=False)
        self.min_lv = nn.Parameter(np.log(min_std**2) * torch.ones(self.out_dim), requires_grad=clip_lv)
        self.max_lv = nn.Parameter(np.log(max_std**2) * torch.ones(self.out_dim), requires_grad=clip_lv)
        
        # normalization stats
        self.obs_mean = nn.Parameter(torch.zeros(obs_dim), requires_grad=False)
        self.obs_variance = nn.Parameter(torch.ones(obs_dim), requires_grad=False)
        self.rwd_mean = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.rwd_variance = nn.Parameter(torch.ones(1), requires_grad=False)
    
    def update_stats(self, obs_mean, obs_variance, rwd_mean, rwd_variance):
        assert obs_mean.shape == self.obs_mean.shape
        assert obs_variance.shape == self.obs_variance.shape
        assert rwd_mean.shape == self.rwd_mean.shape
        assert rwd_variance.shape == self.rwd_variance.shape

        self.obs_mean.data = torch.from_numpy(obs_mean).to(torch.float32).to(self.device)
        self.obs_variance.data = torch.from_numpy(obs_variance).to(torch.float32).to(self.device)
        if self.pred_rwd:
            self.rwd_mean.data = torch.from_numpy(rwd_mean).to(torch.float32).to(self.device)
            self.rwd_variance.data = torch.from_numpy(rwd_variance).to(torch.float32).to(self.device)

    def compute_dists(self, obs, act):
        """ Compute normalized next observation and reward distribution classes 
        
        Returns:
            next_obs_dist (torch_dist.Normal): normalized next observation distribution
            rwd_dist (torch_dist.Normal): normalized reward distribution
        """
        obs_act = torch.cat([obs, act], dim=-1)
        mu_, lv = torch.chunk(self.mlp.forward(obs_act), 2, dim=-1)
        
        if self.residual:
            mu = torch.cat([obs.unsqueeze(-2) + mu_[..., :-1], mu_[..., -1:]], dim=-1)
        else:
            mu = mu_

        if self.clip_lv:
            std = torch.exp(0.5 * soft_clamp(lv, self.min_lv, self.max_lv))
        else:
            std = torch.exp(0.5 * lv.clip(self.min_lv.data, self.max_lv.data))

        next_obs_mu, rwd_mu = torch.split(mu, [self.obs_dim, 1], dim=-1)
        next_obs_std, rwd_std = torch.split(std, [self.obs_dim, 1], dim=-1)
        return torch_dist.Normal(next_obs_mu, next_obs_std), torch_dist.Normal(rwd_mu, rwd_std)
    
    def compute_dists_separate(self, obs, act):
        """ Compute normalized next observation and reward distribution classes 
            separately for each ensemble member
        
        Args:
            obs (torch.tensor): normalized observations. size=[..., ensemble_dim, obs_dim]
            act (torch.tensor): actions. size=[..., ensemble_dim, act_dim]
        
        Returns:
            next_obs_dist (torch_dist.Normal): normalized next observation distribution
            rwd_dist (torch_dist.Normal): normalized reward distribution
        """
        obs_act = torch.cat([obs, act], dim=-1)
        mu_, lv = torch.chunk(self.mlp.forward_separete(obs_act), 2, dim=-1)
        
        if self.residual:
            mu = torch.cat([obs + mu_[..., :-1], mu_[..., -1:]], dim=-1)
        else:
            mu = mu_

        if self.clip_lv:
            std = torch.exp(0.5 * soft_clamp(lv, self.min_lv, self.max_lv))
        else:
            std = torch.exp(0.5 * lv.clip(self.min_lv.data, self.max_lv.data))

        next_obs_mu, rwd_mu = torch.split(mu, [self.obs_dim, 1], dim=-1)
        next_obs_std, rwd_std = torch.split(std, [self.obs_dim, 1], dim=-1)
        return torch_dist.Normal(next_obs_mu, next_obs_std), torch_dist.Normal(rwd_mu, rwd_std)

    def compute_log_prob(self, obs, act, next_obs, rwd):
        """ Compute ensemble log probability 
        
        Args:
            obs (torch.tensor): normalized observations. size=[..., obs_dim]
            act (torch.tensor): actions. size=[..., act_dim]
            next_obs (torch.tensor): normalized next observations. size=[..., obs_dim]
            rwd ([torch.tensor, None]): normalized reward. size=[..., 1]

        Returns:
            logp_obs (torch.tensor): ensemble log probabilities of normalized next observations. size=[..., ensemble_dim, 1]
            logp_rwd (torch.tensor): ensemble log probabilities of normalized rewards. size=[..., ensemble_dim, 1]
        """
        next_obs_dist, rwd_dist = self.compute_dists(obs, act)
        logp_obs = next_obs_dist.log_prob(next_obs.unsqueeze(-2)).sum(-1, keepdim=True)
        logp_rwd = rwd_dist.log_prob(rwd.unsqueeze(-2)).sum(-1, keepdim=True)
        return logp_obs, logp_rwd
    
    def compute_log_prob_separate(self, obs, act, next_obs, rwd):
        """ Compute ensemble log probability separately for each ensemble member 
        
        Args:
            obs (torch.tensor): normalized observations. size=[..., ensemble_dim, obs_dim]
            act (torch.tensor): actions. size=[..., ensemble_dim, act_dim]
            next_obs (torch.tensor): normalized next observations. size=[..., ensemble_dim, obs_dim]
            rwd ([torch.tensor, None]): normalized reward. size=[..., ensemble_dim, 1]

        Returns:
            logp_obs (torch.tensor): ensemble log probabilities of normalized next observations. size=[..., ensemble_dim, 1]
            logp_rwd (torch.tensor): ensemble log probabilities of normalized rewards. size=[..., ensemble_dim, 1]
        """
        next_obs_dist, rwd_dist = self.compute_dists_separate(obs, act)
        logp_obs = next_obs_dist.log_prob(next_obs).sum(-1, keepdim=True)
        logp_rwd = rwd_dist.log_prob(rwd).sum(-1, keepdim=True)
        return logp_obs, logp_rwd
    
    def compute_mixture_log_prob(self, obs, act, next_obs, rwd):
        """ Compute log marginal probability 
        
        Args:
            obs (torch.tensor): normalized observations. size=[..., obs_dim]
            act (torch.tensor): actions. size=[..., act_dim]
            next_obs (torch.tensor): normalized next observations. size=[..., obs_dim]
            rwd ([torch.tensor, None]): normalized reward. size=[..., 1]

        Returns:
            mixture_logp_obs (torch.tensor): log marginal probabilities of normalized next observations. size=[..., 1]
            mixture_logp_rwd (torch.tensor): log marginal probabilities of normalized rewards. size=[..., 1]
        """
        log_elites = torch.log(self.topk_dist + 1e-6).unsqueeze(-1)
        logp_obs, logp_rwd = self.compute_log_prob(obs, act, next_obs, rwd)
        mixture_logp_obs = torch.logsumexp(logp_obs + log_elites, dim=-2)
        mixture_logp_rwd = torch.logsumexp(logp_rwd + log_elites, dim=-2)
        return mixture_logp_obs, mixture_logp_rwd
    
    def sample_dist(self, obs, act, sample_mean=False):
        """ Sample from ensemble 
        
        Args:
            obs (torch.tensor): normalized observations. size=[..., obs_dim]
            act (torch.tensor): normaized actions. size=[..., act_dim]
            sample_mean (bool, optional): whether to sample mean. Default=False

        Returns:
            next_obs (torch.tensor): normalized next observations sampled from ensemble member in topk_dist. size=[..., obs_dim]
            rwd (torch.tensor): normalized reward sampled from ensemble member in topk_dist. size=[..., 1]
        """
        next_obs_dist, rwd_dist = self.compute_dists(obs, act)
        if not sample_mean:
            next_obs = next_obs_dist.rsample()
            rwd = rwd_dist.rsample()
        else:
            next_obs = next_obs_dist.mean
            rwd = rwd_dist.mean
        
        # randomly select from top models
        ensemble_idx = torch_dist.Categorical(self.topk_dist).sample(obs.shape[:-1]).unsqueeze(-1)
        ensemble_idx_obs = ensemble_idx.unsqueeze(-1).repeat_interleave(self.obs_dim, dim=-1).to(self.device) # duplicate alone feature dim
        ensemble_idx_rwd = ensemble_idx.unsqueeze(-1).repeat_interleave(1, dim=-1).to(self.device) # duplicate alone feature dim

        next_obs = torch.gather(next_obs, -2, ensemble_idx_obs).squeeze(-2)
        rwd = torch.gather(rwd, -2, ensemble_idx_rwd).squeeze(-2)
        return next_obs, rwd
    
    def step(self, obs, act, sample_mean=False):
        """ Simulate a step forward with normalization pre and post processing
        
        Args:
            obs (torch.tensor): unnormalized observations. size=[..., obs_dim]
            act (torch.tensor): unnormaized actions. size=[..., act_dim]
            sample_mean (bool, optional): whether to sample mean. Default=False

        Returns:
            next_obs (torch.tensor): sampled unnormalized next observations. size=[..., obs_dim]
            rwd (torch.tensor): sampled unnormalized reward. size=[..., 1]
            done (torch.tensor): done flag. If termination_fn is None, return all zeros. size=[..., 1]
        """
        obs_norm = normalize(obs, self.obs_mean, self.obs_variance)
        next_obs_norm, rwd_norm = self.sample_dist(obs_norm, act, sample_mean=sample_mean)
        next_obs = denormalize(next_obs_norm, self.obs_mean, self.obs_variance)
        rwd = denormalize(rwd_norm, self.rwd_mean, self.rwd_variance)

        if self.termination_fn is not None:
            done = self.termination_fn(
                obs.data.cpu().numpy(), act.data.cpu().numpy(), next_obs.data.cpu().numpy()
            )
            done = torch.from_numpy(done).unsqueeze(-1).to(torch.float32).to(self.device)
        else:
            done = torch.zeros(list(obs.shape)[:-1] + [1]).to(torch.float32).to(self.device)
        return next_obs, rwd, done

    def compute_loss(self, obs, act, next_obs, rwd, use_decay=True):
        """ Compute ensemble log likelihood for normalized data and weight decay loss """
        logp_obs, logp_rwd = self.compute_log_prob_separate(obs, act, next_obs, rwd)
        logp = logp_obs + self.pred_rwd * logp_rwd
        clip_lv_loss = 0.001 * (
            self.max_lv[:-1].sum() + self.pred_rwd * self.max_lv[-1:].sum() \
            - self.min_lv[:-1].sum() - self.pred_rwd * self.min_lv[-1:].sum()
        )
        decay_loss = self.compute_decay_loss()
        loss = (
            -logp.mean(-1).mean(0).sum(-1) / (self.obs_dim + 1 * self.pred_rwd)
            + clip_lv_loss \
            + decay_loss * use_decay
        )
        return loss
    
    def compute_decay_loss(self):
        i, loss = 0, 0
        for layer in self.mlp.layers:
            if hasattr(layer, "weight"):
                loss += self.decay[i] * torch.sum(layer.weight ** 2) / 2.
                i += 1
        return loss
    
    def evaluate(self, obs, act, next_obs, rwd):
        """ Compute mean average error of normalized data for each ensemble 
        
        Returns:
            stats (dict): MAE dict with fields [mae_0, ..., mae_{ensemble_dim}, mae]
        """
        with torch.no_grad():
            next_obs_dist, rwd_dist = self.compute_dists(obs, act)
        
        obs_mae = torch.abs(next_obs_dist.mean - next_obs.unsqueeze(-2)).mean((0, 2))
        mae = obs_mae
        
        mae_stats = {f"mae_{i}": mae[i].cpu().data.item() for i in range(self.ensemble_dim)}
        obs_mae_stats = {f"obs_mae_{i}": obs_mae[i].cpu().data.item() for i in range(self.ensemble_dim)}
        obs_mae_stats["obs_mae"] = obs_mae.mean().cpu().data.item()
        rwd_mae_stats = {}

        if self.pred_rwd:
            rwd_mae = torch.abs(rwd_dist.mean - rwd.unsqueeze(-2)).mean((0, 2))
            mae = obs_mae * self.obs_dim / (self.obs_dim + 1) + rwd_mae / (self.obs_dim + 1)
            
            rwd_mae_stats = {f"rwd_mae_{i}": rwd_mae[i].cpu().data.item() for i in range(self.ensemble_dim)}
            rwd_mae_stats["rwd_mae"] = rwd_mae.mean().cpu().data.item()

        stats = {**mae_stats, **obs_mae_stats, **rwd_mae_stats}
        stats["mae"] = mae.mean().cpu().data.item()
        return stats
    
    def update_topk_dist(self, stats):
        """ Update top k model selection distribution """
        maes = [stats[f"mae_{i}"] for i in range(self.ensemble_dim)]
        idx_topk = np.argsort(maes)[:self.topk]
        topk_dist = np.zeros(self.ensemble_dim)
        topk_dist[idx_topk] = 1./self.topk
        self.topk_dist.data = torch.from_numpy(topk_dist).to(torch.float32).to(self.device)


def parse_ensemble_params(ensemble):
    """ Parse ensemble parameters into list of size ensemble_dim 
    
    Returns:
        params_list (list): list of named parameters dictionaries
    """
    ensemble_dim = ensemble.ensemble_dim
    params_list = [{} for _ in range(ensemble_dim)]
    for n, p in ensemble.named_parameters():
        if "weight" in n or "bias" in n:
            for i in range(ensemble_dim):
                params_list[i].update({n: p[i].data.clone()})
    return params_list

def set_ensemble_params(ensemble, params_list):
    """ Set ensemble parameters from list of size ensemble_dim 
    
    Args:
        ensemble (EnsembleDyanmics): EnsembleDynamics object
        params_list (list): list of named parameters dictionaries
    """
    ensemble_dim = ensemble.ensemble_dim
    for n, p in ensemble.named_parameters():
        if "weight" in n or "bias" in n:
            p.data = torch.stack([params_list[i][n] for i in range(ensemble_dim)])

def get_random_index(batch_size, ensemble_dim, bootstrap=True):
    if bootstrap:
        return np.stack([np.random.choice(np.arange(batch_size), batch_size, replace=False) for _ in range(ensemble_dim)]).T
    else:
        idx = np.random.choice(np.arange(batch_size), batch_size, replace=False)
        return np.stack([idx for _ in range(ensemble_dim)]).T

def train_ensemble(
        data, agent, optimizer, eval_ratio, batch_size, epochs, bootstrap=True, grad_clip=None, 
        update_stats=True, update_elites=True, max_epoch_since_update=10, 
        verbose=1, callback=None, debug=False
    ):
    """
    Args:
        data (list): list of [obs, act, rwd, next_obs]
        agent (nn.Module): agent with dynamics property
        optimizer (torch.optim): optimizer
        eval_ratio (float): evaluation ratio
        batch_size (int): batch size
        epochs (int): max training epochs
        bootstrap (bool): whether to use different minibatch ordering for each ensemble member. Default=True
        grad_clip (float): gradient norm clipping. Default=None
        update_stats (bool): whether to normalize data and update reward and dynamics model stats. Default=True
        update_elites (bool): whether to update reward and dynamics topk_dist. Default=True
        max_epoch_since_update (int): max epoch for termination condition. Default=10
        verbose (int): verbose interval. Default=1
        callback (object): callback object. Default=None
        debug (bool): debug flag. If True will print data stats. Default=None

    Returns:
        logger (Logger): logger class with training history
    """
    obs, act, rwd, next_obs = data

    # train test split
    num_eval = int(len(obs) * eval_ratio)
    obs_train = obs[:-num_eval]
    act_train = act[:-num_eval]
    rwd_train = rwd[:-num_eval]
    next_obs_train = next_obs[:-num_eval]

    obs_eval = obs[-num_eval:]
    act_eval = act[-num_eval:]
    rwd_eval = rwd[-num_eval:]
    next_obs_eval = next_obs[-num_eval:]
    
    # normalize data
    obs_mean, obs_var = 0., 1.
    rwd_mean, rwd_var = 0., 1.
    if update_stats:
        obs_mean = obs_train.mean(0)
        obs_var = obs_train.var(0)
        rwd_mean = rwd_train.mean(0)
        rwd_var = rwd_train.var(0)
        agent.dynamics.update_stats(obs_mean, obs_var, rwd_mean, rwd_var)

    obs_train = normalize(obs_train, obs_mean, obs_var)
    obs_eval = normalize(obs_eval, obs_mean, obs_var)
    next_obs_train = normalize(next_obs_train, obs_mean, obs_var)
    next_obs_eval = normalize(next_obs_eval, obs_mean, obs_var)
    rwd_train = normalize(rwd_train, rwd_mean, rwd_var)
    rwd_eval = normalize(rwd_eval, rwd_mean, rwd_var)

    if debug:
        print("obs_train_mean", obs_train.mean(0).round(2))
        print("obs_train_std", obs_train.std(0).round(2))
        print("obs_eval_mean", obs_eval.mean(0).round(2))
        print("obs_eval_std", obs_eval.std(0).round(2))
        print("next_obs_train_mean", next_obs_train.mean(0).round(2))
        print("next_obs_train_std", next_obs_train.std(0).round(2))
        print("next_obs_eval_mean", next_obs_eval.mean(0).round(2))
        print("next_obs_eval_std", next_obs_eval.std(0).round(2))
        
        print()
        print("rwd_train_mean", rwd_train.mean(0).round(2))
        print("rwd_train_std", rwd_train.std(0).round(2))
        print("rwd_eval_mean", rwd_eval.mean(0).round(2))
        print("rwd_eval_std", rwd_eval.std(0).round(2))
        print()
    
    # pack eval data
    obs_eval = torch.from_numpy(obs_eval).to(torch.float32).to(agent.device)
    act_eval = torch.from_numpy(act_eval).to(torch.float32).to(agent.device)
    rwd_eval = torch.from_numpy(rwd_eval).to(torch.float32).to(agent.device)
    next_obs_eval = torch.from_numpy(next_obs_eval).to(torch.float32).to(agent.device)
    
    logger = Logger()
    start_time = time.time()
    
    ensemble_dim = agent.dynamics.ensemble_dim

    best_eval = [1e6] * ensemble_dim
    best_params_list = parse_ensemble_params(agent.dynamics)
    epoch_since_last_update = 0
    for e in range(epochs):
        # shuffle train data
        idx_train = get_random_index(obs_train.shape[0], ensemble_dim, bootstrap=bootstrap)
        
        train_stats_epoch = []
        for i in range(0, obs_train.shape[0], batch_size):
            idx_batch = idx_train[i:i+batch_size]
            obs_batch = torch.from_numpy(obs_train[idx_batch]).to(torch.float32).to(agent.dynamics.device)
            act_batch = torch.from_numpy(act_train[idx_batch]).to(torch.float32).to(agent.dynamics.device)
            rwd_batch = torch.from_numpy(rwd_train[idx_batch]).to(torch.float32).to(agent.dynamics.device)
            next_obs_batch = torch.from_numpy(next_obs_train[idx_batch]).to(torch.float32).to(agent.dynamics.device)
            
            loss = agent.dynamics.compute_loss(obs_batch, act_batch, next_obs_batch, rwd_batch)

            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(agent.dynamics.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()

            stats = {"loss": loss.cpu().data.item()}
            train_stats_epoch.append(stats)
            logger.push(stats)
        train_stats_epoch = pd.DataFrame(train_stats_epoch).mean(0).to_dict()
        
        # evaluate
        eval_stats_epoch = agent.dynamics.evaluate(obs_eval, act_eval, next_obs_eval, rwd_eval)
        if update_elites:
            agent.dynamics.update_topk_dist(eval_stats_epoch)

        # log stats
        stats_epoch = {**train_stats_epoch, **eval_stats_epoch}
        logger.push(stats_epoch)
        logger.push({"epoch": e + 1})
        logger.push({"time": time.time() - start_time})
        logger.log(silent=True)
        
        if callback is not None:
            callback(agent, pd.DataFrame(logger.history))
        
        if (e + 1) % verbose == 0:
            print("e: {}, loss: {:.4f}, obs_mae: {:.4f}, rwd_mae: {:.4f}, terminate: {}/{}".format(
                e + 1, 
                stats_epoch["loss"], 
                stats_epoch["obs_mae"],
                0. if not agent.dynamics.pred_rwd else stats_epoch["rwd_mae"],
                epoch_since_last_update,
                max_epoch_since_update,
                ))
        
        # termination condition based on eval performance
        updated = False
        current_params_list = parse_ensemble_params(agent.dynamics)
        for m in range(ensemble_dim):
            current_eval = stats_epoch[f"mae_{m}"]
            improvement = (best_eval[m] - current_eval) / (best_eval[m] + 1e-6)
            if improvement > 0.01:
                best_eval[m] = min(best_eval[m], current_eval)
                best_params_list[m] = current_params_list[m]
                updated = True

        if updated:
            epoch_since_last_update = 0
        else:
            epoch_since_last_update += 1
         
        if epoch_since_last_update > max_epoch_since_update:
            break
    
    set_ensemble_params(agent.dynamics, best_params_list)
    return logger

if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    from src.env.gym_wrapper import get_termination_fn
    
    obs_dim = 11
    act_dim = 3
    ensemble_dim = 7
    topk = 5
    hidden_dim = 200
    num_hidden = 2
    activation = "silu"
    decay = [0.000025, 0.00005, 0.000075, 0.0001]
    min_std = 1e-5
    max_std = 100
    termination_fn = get_termination_fn("Hopper-v4")

    # synthetic data
    batch_size = 1000
    obs = torch.randn(batch_size, obs_dim)
    act = torch.randn(batch_size, act_dim)
    rwd = torch.randn(batch_size, 1)
    next_obs = torch.randn(batch_size, obs_dim)
    
    obs_sep = torch.randn(batch_size, ensemble_dim, obs_dim)
    act_sep = torch.randn(batch_size, ensemble_dim, act_dim)
    rwd_sep = torch.randn(batch_size, ensemble_dim, 1)
    next_obs_sep = torch.randn(batch_size, ensemble_dim, obs_dim)

    def test_soft_clamp():
        _min = torch.Tensor([-1.])
        _max = torch.Tensor([1.])
        x1 = torch.Tensor([0.])
        x2 = torch.Tensor([1.5])
        x3 = torch.Tensor([-1.5])
        assert _min < soft_clamp(x1, _min, _max) < _max
        assert _min < soft_clamp(x2, _min, _max) < _max
        assert _min < soft_clamp(x3, _min, _max) < _max

    def test_ensemble(clip_lv=False, residual=False, termination_fn=None, pred_rwd=False):
        dynamics = EnsembleDynamics(
            obs_dim,
            act_dim, 
            pred_rwd,
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
        assert list(dynamics.min_lv.shape) == [obs_dim + 1]
        assert list(dynamics.max_lv.shape) == [obs_dim + 1]
        assert list(dynamics.obs_mean.shape) == [obs_dim]
        assert list(dynamics.obs_variance.shape) == [obs_dim]
        assert list(dynamics.rwd_mean.shape) == [1]
        assert list(dynamics.rwd_variance.shape) == [1]
        assert torch.isclose(dynamics.min_lv.exp() ** 0.5, min_std * torch.ones(1), atol=1e-5).all()
        assert torch.isclose(dynamics.max_lv.exp() ** 0.5, max_std * torch.ones(1), atol=1e-5).all()
        
        # test method output shapes on regular batch
        next_obs_dist, rwd_dist = dynamics.compute_dists(obs, act)
        logp_obs, logp_rwd =  dynamics.compute_log_prob(obs, act, next_obs, rwd)
        mix_logp_obs, mix_logp_rwd = dynamics.compute_mixture_log_prob(obs, act, next_obs, rwd)
        next_obs_sample, rwd_sample = dynamics.sample_dist(obs, act)
        next_obs_step, rwd_step, done = dynamics.step(obs, act)

        assert list(next_obs_dist.mean.shape) == [batch_size, ensemble_dim, obs_dim]
        assert list(rwd_dist.mean.shape) == [batch_size, ensemble_dim, 1]
        assert list(next_obs_dist.variance.shape) == [batch_size, ensemble_dim, obs_dim]
        assert list(rwd_dist.variance.shape) == [batch_size, ensemble_dim, 1]
        assert list(logp_obs.shape) == [batch_size, ensemble_dim, 1]
        assert list(logp_rwd.shape) == [batch_size, ensemble_dim, 1]
        assert list(mix_logp_obs.shape) == [batch_size, 1]
        assert list(mix_logp_rwd.shape) == [batch_size, 1]
        assert list(next_obs_sample.shape) == [batch_size, obs_dim]
        assert list(rwd_sample.shape) == [batch_size, 1]
        assert list(next_obs_step.shape) == [batch_size, obs_dim]
        assert list(rwd_step.shape) == [batch_size, 1]
        assert list(done.shape) == [batch_size, 1]
        
        # test method output shapes on separate batch
        next_obs_dist, rwd_dist = dynamics.compute_dists_separate(obs_sep, act_sep)
        logp_obs, logp_rwd =  dynamics.compute_log_prob_separate(obs_sep, act_sep, next_obs_sep, rwd_sep)
        
        assert list(next_obs_dist.mean.shape) == [batch_size, ensemble_dim, obs_dim]
        assert list(rwd_dist.mean.shape) == [batch_size, ensemble_dim, 1]
        assert list(next_obs_dist.variance.shape) == [batch_size, ensemble_dim, obs_dim]
        assert list(rwd_dist.variance.shape) == [batch_size, ensemble_dim, 1]
        assert list(logp_obs.shape) == [batch_size, ensemble_dim, 1]
        assert list(logp_rwd.shape) == [batch_size, ensemble_dim, 1]
        
        # test eval
        stats = dynamics.evaluate(obs, act, next_obs, rwd)
        dynamics.update_topk_dist(stats)
        assert sum(dynamics.topk_dist == 0) == (dynamics.ensemble_dim - dynamics.topk)

        # test gradients: drop one member and check no gradients
        logp_obs, logp_rwd = dynamics.compute_log_prob_separate(obs_sep, act_sep, next_obs_sep, rwd_sep)
        loss = torch.mean(logp_obs[:, :-1] + logp_rwd[:, :-1])
        loss.backward()
        
        head_weight = dict(dynamics.named_parameters())["mlp.layers.6.weight"]
        head_bias = dict(dynamics.named_parameters())["mlp.layers.6.bias"]

        assert torch.all(head_weight.grad[-1] == 0)
        assert torch.all(head_bias.grad[-1] == 0)

        for n, p in dynamics.named_parameters():
            if p.grad is not None:
                p.grad.zero_()

        # test gradients: check reward head gradients
        loss = dynamics.compute_loss(obs_sep, act_sep, next_obs_sep, rwd_sep, use_decay=False)
        loss.backward()
        
        head_weight = dict(dynamics.named_parameters())["mlp.layers.6.weight"]
        head_bias = dict(dynamics.named_parameters())["mlp.layers.6.bias"]
        
        if pred_rwd:
            assert torch.all(head_weight.grad[:, :, obs_dim] != 0)
            assert torch.all(head_weight.grad[:, :, obs_dim * 2 + 1] != 0)
            assert torch.all(head_bias.grad[:, obs_dim] != 0)
            assert torch.all(head_bias.grad[:, obs_dim * 2 + 1] != 0)
            if clip_lv:
                assert torch.all(dynamics.min_lv.grad[obs_dim] != 0)
                assert torch.all(dynamics.max_lv.grad[obs_dim] != 0)
        else:
            assert torch.all(head_weight.grad[:, :, obs_dim] == 0)
            assert torch.all(head_weight.grad[:, :, obs_dim * 2 + 1] == 0)
            assert torch.all(head_bias.grad[:, obs_dim] == 0)
            assert torch.all(head_bias.grad[:, obs_dim * 2 + 1] == 0)
            if clip_lv:
                assert torch.all(dynamics.min_lv.grad[obs_dim] == 0)
                assert torch.all(dynamics.max_lv.grad[obs_dim] == 0)
    
    def test_ensemble_parsing():
        def ensemble_equal(m1, m2):
            """ Check whether two ensembles have equal parameters """
            are_equal = True
            for ((n1, p1), (n2, p2)) in zip(m1.named_parameters(), m2.named_parameters()):
                if ("weight" in n1 or "bias" in n1) and "weight" in n2 or "bias" in n2:
                    if torch.all(p1 != p2):
                        are_equal = False
                        break
            return are_equal
        
        pred_rwd = True
        ensemble_1 = EnsembleDynamics(
            obs_dim,
            act_dim, 
            pred_rwd,
            ensemble_dim,
            topk,
            hidden_dim,
            num_hidden,
            activation,
        )
        ensemble_2 = EnsembleDynamics(
            obs_dim,
            act_dim, 
            pred_rwd,
            ensemble_dim,
            topk,
            hidden_dim,
            num_hidden,
            activation,
        )
        assert not ensemble_equal(ensemble_1, ensemble_2)

        params_list = parse_ensemble_params(ensemble_1)
        set_ensemble_params(ensemble_2, params_list)
        assert ensemble_equal(ensemble_1, ensemble_2)
    
    def test_get_random_index():
        idx = get_random_index(batch_size, ensemble_dim, bootstrap=True)
        obs_batch = obs[idx]

        assert all([len(np.unique(idx[:, i])) == batch_size for i in range(ensemble_dim)])
        assert np.any(idx[:, 0] != idx[:, 1]) # at least one is not the same

        for i in range(ensemble_dim):
            assert torch.isclose(obs_batch[:, i], obs[idx[:, i]]).all()

    # start testing
    test_soft_clamp()
    print("soft_clamp passed")
    
    test_ensemble(
        clip_lv=False, 
        residual=False, 
        termination_fn=None,
        pred_rwd=False
    )
    
    test_ensemble(
        clip_lv=False, 
        residual=False, 
        termination_fn=termination_fn,
        pred_rwd=True
    )

    test_ensemble(
        clip_lv=True, 
        residual=False, 
        termination_fn=termination_fn,
        pred_rwd=True
    )

    test_ensemble(
        clip_lv=True, 
        residual=False, 
        termination_fn=termination_fn,
        pred_rwd=False
    )

    test_ensemble(
        clip_lv=True, 
        residual=True, 
        termination_fn=termination_fn,
        pred_rwd=True
    )
    print("test_ensemble passed")

    test_get_random_index()
    print("test_get_random_index passed")

    test_ensemble_parsing()
    print("test_ensemble_parsing passed")