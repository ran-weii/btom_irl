import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as torch_dist
from src.agents.nn_models import EnsembleMLP
from src.utils.data import normalize, denormalize
from src.utils.logging import Logger

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
        clip_mu=False,
        clip_lv=False, 
        residual=False,
        termination_fn=None, 
        max_mu=3.,
        min_std=0.04,
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
            clip_mu (bool, optional): whether to soft clip observation mean. Default=False
            clip_lv (bool, optional): whether to soft clip observation log variance. Default=False
            residual (bool, optional): whether to predict observation residuals. Default=False
            termination_fn (func, optional): termination function to output rollout done. Default=None
            max_mu (float, optional): maximum mean prediction. Default=3.
            min_std (float, optional): minimum standard deviation. Default=0.04
            max_std (float, optional): maximum standard deviation. Default=1.6
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
        self.clip_mu = clip_mu
        self.clip_lv = clip_lv
        self.residual = residual
        self.termination_fn = termination_fn
        self.max_mu = max_mu
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
        self.max_lm = nn.Parameter(np.log(max_mu) * torch.ones(out_dim), requires_grad=clip_mu)
        self.min_lv = nn.Parameter(np.log(min_std**2) * torch.ones(out_dim), requires_grad=clip_lv)
        self.max_lv = nn.Parameter(np.log(max_std**2) * torch.ones(out_dim), requires_grad=clip_lv)

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
        
        if self.clip_mu:
            mu_ = soft_clamp(mu_, -self.max_lm.exp(), self.max_lm.exp())
        else:
            mu_ = torch.clip(mu_, -self.max_mu, self.max_mu)

        if self.residual:
            mu = obs.unsqueeze(-2) + mu_
        else:
            mu = mu_

        if self.clip_lv:
            std = torch.exp(0.5 * soft_clamp(lv, self.min_lv, self.max_lv))
        else:
            std = torch.exp(0.5 * lv.clip(self.min_lv.data, self.max_lv.data))
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
    
    def sample_dist(self, obs, act, sample_mean=False):
        """ Sample from ensemble 
        
        Args:
            obs (torch.tensor): normalized observations. size=[..., obs_dim]
            act (torch.tensor): normaized actions. size=[..., act_dim]
            sample_mean (bool, optional): whether to sample mean. Default=False

        Returns:
            out (torch.tensor): normalized output sampled from ensemble member in topk_dist. size=[..., out_dim]
        """
        out_dist = self.compute_dist(obs, act)
        if not sample_mean:
            out = out_dist.rsample()
        else:
            out = out_dist.mean
        
        # randomly select from top models
        ensemble_idx = torch_dist.Categorical(self.topk_dist).sample(obs.shape[:-1]).unsqueeze(-1)
        ensemble_idx_ = ensemble_idx.unsqueeze(-1).repeat_interleave(self.out_dim, dim=-1).to(self.device) # duplicate alone feature dim
        out = torch.gather(out, -2, ensemble_idx_).squeeze(-2)
        return out
    
    def step(self, obs, act, sample_mean=False):
        """ Simulate a step forward with normalization pre and post processing
        
        Args:
            obs (torch.tensor): unnormalized observations. size=[..., obs_dim]
            act (torch.tensor): unnormaized actions. size=[..., act_dim]
            sample_mean (bool, optional): whether to sample mean. Default=False

        Returns:
            out (torch.tensor): sampled unnormalized outputs. size=[..., out_dim]
            done (torch.tensor): done flag. If termination_fn is None, return all zeros. size=[..., 1]
        """
        obs_norm = normalize(obs, self.obs_mean, self.obs_variance)
        out_norm = self.sample_dist(obs_norm, act, sample_mean=sample_mean)
        out = denormalize(out_norm, self.out_mean, self.out_variance)

        if self.termination_fn is not None:
            done = self.termination_fn(
                obs.data.cpu().numpy(), act.data.cpu().numpy(), out.data.cpu().numpy()
            )
            done = torch.from_numpy(done).unsqueeze(-1).to(torch.float32).to(self.device)
        else:
            done = torch.zeros(list(obs.shape)[:-1] + [1]).to(torch.float32).to(self.device)
        return out, done
    
    def compute_loss(self, obs, act, target):
        """ Compute log likelihood for normalized data and weight decay loss """
        logp = self.compute_log_prob(obs, act, target).sum(-1)
        clip_mu_loss = 0.001 * self.max_lm.sum()
        clip_lv_loss = 0.001 * self.max_lv.sum() - 0.001 * self.min_lv.sum()
        decay_loss = self.compute_decay_loss()
        loss = -logp.mean() + clip_mu_loss + clip_lv_loss + decay_loss
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
        
        out_mae = torch.abs(out_pred - target.unsqueeze(-2)).mean((0, 2))
        
        stats = {f"mae_{i}": out_mae[i].cpu().data.item() for i in range(self.ensemble_dim)}
        stats["mae"] = out_mae.mean().cpu().data.item()
        return stats
    
    def update_topk_dist(self, stats):
        """ Update top k model selection distribution """
        maes = [v for k, v in stats.items() if "mae_" in k]
        idx_topk = np.argsort(maes)[:self.topk]
        topk_dist = np.zeros(self.ensemble_dim)
        topk_dist[idx_topk] = 1./self.topk
        self.topk_dist.data = torch.from_numpy(topk_dist).to(torch.float32).to(self.device)


def train_ensemble(
        data, agent, eval_ratio, batch_size, epochs, grad_clip=None, train_reward=True, 
        update_stats=True, update_elites=True, max_epoch_since_update=10, 
        verbose=1, callback=None, debug=False
    ):
    """
    Args:
        data (list): list of [obs, act, rwd, next_obs]
        agent (nn.Module): agent with reward, dynamics, and optimizers properties
        eval_ratio (float): evaluation ratio
        epochs (int): max training epochs
        grad_clip (float): gradient norm clipping. Default=None
        train_reward (bool): whether to train reward. Default=True
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
        agent.dynamics.update_stats(obs_mean, obs_var, obs_mean, obs_var)
        if train_reward:
            agent.reward.update_stats(obs_mean, obs_var, rwd_mean, rwd_var)

    obs_train = normalize(obs_train, obs_mean, obs_var)
    obs_eval = normalize(obs_eval, obs_mean, obs_var)
    next_obs_train = normalize(next_obs_train, obs_mean, obs_var)
    next_obs_eval = normalize(next_obs_eval, obs_mean, obs_var)
    if train_reward:
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
    
    # pack eval data
    obs_eval = torch.from_numpy(obs_eval).to(torch.float32).to(agent.device)
    act_eval = torch.from_numpy(act_eval).to(torch.float32).to(agent.device)
    rwd_eval = torch.from_numpy(rwd_eval).to(torch.float32).to(agent.device)
    next_obs_eval = torch.from_numpy(next_obs_eval).to(torch.float32).to(agent.device)
    
    logger = Logger()
    start_time = time.time()
    # best_eval = 1e6
    ensemble_dim = agent.dynamics.ensemble_dim
    best_eval = [1e6] * ensemble_dim
    epoch_since_last_update = 0
    for e in range(epochs):
        # shuffle train data
        idx_train = np.arange(len(obs_train))
        np.random.shuffle(idx_train)

        train_stats_epoch = []
        for i in range(0, obs_train.shape[0], batch_size):
            idx_batch = idx_train[i:i+batch_size]
            obs_batch = torch.from_numpy(obs_train[idx_batch]).to(torch.float32).to(agent.device)
            act_batch = torch.from_numpy(act_train[idx_batch]).to(torch.float32).to(agent.device)
            rwd_batch = torch.from_numpy(rwd_train[idx_batch]).to(torch.float32).to(agent.device)
            next_obs_batch = torch.from_numpy(next_obs_train[idx_batch]).to(torch.float32).to(agent.device)
            
            obs_loss = agent.dynamics.compute_loss(obs_batch, act_batch, next_obs_batch)
            total_loss = obs_loss
            stats = {"obs_loss": obs_loss.cpu().data.item()}
            if train_reward:
                rwd_loss = agent.reward.compute_loss(obs_batch, act_batch, rwd_batch)
                total_loss += rwd_loss
                stats["rwd_loss"] = rwd_loss.cpu().data.item()

            total_loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(agent.parameters(), grad_clip)
            agent.optimizers["dynamics"].step()
            agent.optimizers["dynamics"].zero_grad()
            if train_reward:
                agent.optimizers["reward"].step()
                agent.optimizers["reward"].zero_grad()

            train_stats_epoch.append(stats)
            logger.push(stats)
        train_stats_epoch = pd.DataFrame(train_stats_epoch).mean(0).to_dict()
        
        # evaluate
        obs_eval_stats_epoch = agent.dynamics.evaluate(obs_eval, act_eval, next_obs_eval)
        obs_eval_stats_epoch = {"obs_" + k: v for k, v in obs_eval_stats_epoch.items()}
        eval_stats_epoch = obs_eval_stats_epoch
        if update_elites:
            agent.dynamics.update_topk_dist(obs_eval_stats_epoch)
        if train_reward:
            rwd_eval_stats_epoch = agent.reward.evaluate(obs_eval, act_eval, rwd_eval)
            rwd_eval_stats_epoch = {"rwd_" + k: v for k, v in rwd_eval_stats_epoch.items()}
            eval_stats_epoch = {**eval_stats_epoch, **rwd_eval_stats_epoch}
            if update_elites:
                agent.reward.update_topk_dist(rwd_eval_stats_epoch)

        # log stats
        stats_epoch = {**train_stats_epoch, **eval_stats_epoch}
        logger.push(stats_epoch)
        logger.push({"epoch": e + 1})
        logger.push({"time": time.time() - start_time})
        logger.log(silent=True)
        
        if callback is not None:
            callback(agent, pd.DataFrame(logger.history))
        
        if (e + 1) % verbose == 0:
            print("e: {}, obs_loss: {:.4f}, obs_mae: {:.4f}, rwd_loss: {:.4f}, rwd_mae: {:.4f}, terminate: {}/{}".format(
                e + 1, 
                stats_epoch["obs_loss"], 
                stats_epoch["obs_mae"],
                0. if not train_reward else stats_epoch["rwd_loss"],
                0. if not train_reward else stats_epoch["rwd_mae"],
                epoch_since_last_update,
                max_epoch_since_update,
                ))
        
        # termination condition based on eval performance
        updated = False
        for m in range(ensemble_dim):
            current_eval = stats_epoch[f"obs_mae_{m}"]
            if train_reward:
                current_eval += stats_epoch[f"rwd_mae_{m}"]
            improvement = (best_eval[m] - current_eval) / (best_eval[m] + 1e-6)
            if improvement > 0.01:
                best_eval[m] = min(best_eval[m], current_eval)
                updated = True

        if updated:
            epoch_since_last_update = 0
        else:
            epoch_since_last_update += 1
         
        if epoch_since_last_update > max_epoch_since_update:
            break
    return logger

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
        assert torch.isclose(dynamics.min_lv.exp() ** 0.5, min_std * torch.ones(1), atol=1e-5).all()
        assert torch.isclose(dynamics.max_lv.exp() ** 0.5, max_std * torch.ones(1), atol=1e-5).all()

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