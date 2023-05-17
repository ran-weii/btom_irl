import time
import numpy as np
import pandas as pd
from tqdm import tqdm
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

class StandardScaler(nn.Module):
    """ Input normalizer """
    def __init__(self, input_dim, device=torch.device("cpu")):
        super().__init__()
        self.input_dim = input_dim
        self.device = device
        self.mean = nn.Parameter(torch.zeros(input_dim), requires_grad=False)
        self.variance = nn.Parameter(torch.ones(input_dim), requires_grad=False)
    
    def __repr__(self):
        s = "{}(input_dim={})".format(self.__class__.__name__, self.input_dim)
        return s
    
    def fit(self, x):
        """ Update stats from torch tensor """
        mean = x.mean(0)
        variance = x.var(0)
        self.mean.data = mean.to(torch.float32).to(self.device)
        self.variance.data = variance.to(torch.float32).to(self.device)
        
    def transform(self, x):
        """ Normalize inputs torch tensor """
        return normalize(x, self.mean, self.variance)
    
    def inverse_transform(self, x_norm):
        """ Denormalize inputs torch tensor """
        return denormalize(x_norm, self.mean, self.variance)


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
        self.out_dim = obs_dim + pred_rwd
        self.ensemble_dim = ensemble_dim
        self.topk = topk
        self.decay = decay
        self.pred_rwd = pred_rwd
        self.residual = residual
        self.termination_fn = termination_fn
        self.device = device
        
        self.scaler = StandardScaler(self.obs_dim + self.act_dim)
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
        self.min_lv = nn.Parameter(np.log(min_std**2) * torch.ones(self.out_dim), requires_grad=True)
        self.max_lv = nn.Parameter(np.log(max_std**2) * torch.ones(self.out_dim), requires_grad=True)
    
    def compute_stats(self, inputs):
        """ Compute prediction stats 
        
        Args:
            inputs (torch.tensor): inputs. size=[..., obs_dim + act_dim]

        Returns:
            mu (torch.tensor): prediction mean. size=[..., ensemble_dim, out_dim]
            lv (torch.tensor): prediction log variance. size=[..., ensemble_dim, out_dim]
        """
        inputs_norm = self.scaler.transform(inputs)
        mu, lv_ = torch.chunk(self.mlp.forward(inputs_norm), 2, dim=-1)
        lv = soft_clamp(lv_, self.min_lv, self.max_lv)
        return mu, lv
    
    def compute_stats_separate(self, inputs):
        """ Compute prediction stats for each ensemble member 
        
        Args:
            inputs (torch.tensor): inputs. size=[..., ensemble_dim, obs_dim + act_dim]

        Returns:
            mu (torch.tensor): prediction mean. size=[..., ensemble_dim, out_dim]
            lv (torch.tensor): prediction log variance. size=[..., ensemble_dim, out_dim]
        """
        inputs_norm = self.scaler.transform(inputs)
        mu, lv_ = torch.chunk(self.mlp.forward_separete(inputs_norm), 2, dim=-1)
        lv = soft_clamp(lv_, self.min_lv, self.max_lv)
        return mu, lv
    
    def compute_dists(self, obs, act):
        """ Compute prediction distribution classes 
        
        Returns:
            dist (torch_dist.Normal): prediction distribution
        """
        obs_act = torch.cat([obs, act], dim=-1)
        mu, lv = self.compute_stats(obs_act)
        std = torch.exp(0.5 * lv)
        return torch_dist.Normal(mu, std)
    
    def compute_log_prob(self, obs, act, next_obs, rwd):
        """ Compute ensemble log probability 
        
        Args:
            obs (torch.tensor): observations. size=[..., obs_dim]
            act (torch.tensor): actions. size=[..., act_dim]
            next_obs (torch.tensor): next observations. size=[..., obs_dim]
            rwd ([torch.tensor, None]): reward. size=[..., 1]

        Returns:
            logp_obs (torch.tensor): ensemble log probabilities of targets. size=[..., ensemble_dim, 1]
        """
        if self.residual:
            target = next_obs - obs
        else:
            target = next_obs

        if self.pred_rwd:
            target = torch.cat([target, rwd], dim=-1)

        dist = self.compute_dists(obs, act)
        logp = dist.log_prob(target.unsqueeze(-2)).sum(-1, keepdim=True)
        return logp
    
    def compute_mixture_log_prob(self, obs, act, next_obs, rwd):
        """ Compute log marginal probability 
        
        Args:
            obs (torch.tensor): observations. size=[..., obs_dim]
            act (torch.tensor): actions. size=[..., act_dim]
            next_obs (torch.tensor): next observations. size=[..., obs_dim]
            rwd ([torch.tensor, None]): reward. size=[..., 1]

        Returns:
            mixture_logp (torch.tensor): log marginal probabilities of targets. size=[..., 1]
        """
        log_elites = torch.log(self.topk_dist + 1e-6).unsqueeze(-1)
        logp = self.compute_log_prob(obs, act, next_obs, rwd)
        mixture_logp = torch.logsumexp(logp + log_elites, dim=-2)
        return mixture_logp
    
    def sample_dist(self, obs, act, sample_mean=False):
        """ Sample from ensemble
        
        Args:
            obs (torch.tensor): normalized observations. size=[..., obs_dim]
            act (torch.tensor): normaized actions. size=[..., act_dim]
            sample_mean (bool, optional): whether to sample mean. Default=False

        Returns:
            out (torch.tensor): predictions sampled from ensemble member in topk_dist. size=[..., out_dim]
        """
        dist = self.compute_dists(obs, act)
        if not sample_mean:
            out = dist.rsample()
        else:
            out = dist.mean
        
        # randomly select from top models
        ensemble_idx = torch_dist.Categorical(self.topk_dist).sample(obs.shape[:-1]).unsqueeze(-1)
        ensemble_idx_obs = ensemble_idx.unsqueeze(-1).repeat_interleave(self.obs_dim + self.pred_rwd, dim=-1).to(self.device) # duplicate alone feature dim

        out = torch.gather(out, -2, ensemble_idx_obs).squeeze(-2)
        return out
    
    def step(self, obs, act, sample_mean=False):
        """ Simulate a step forward
        
        Args:
            obs (torch.tensor): observations. size=[..., obs_dim]
            act (torch.tensor): actions. size=[..., act_dim]
            sample_mean (bool, optional): whether to sample mean. Default=False

        Returns:
            next_obs (torch.tensor): sampled next observations. size=[..., obs_dim]
            rwd (torch.tensor): sampled reward, set to zeros if pred_rwd=False. size=[..., 1]
            done (torch.tensor): done flag. If termination_fn is None, return all zeros. size=[..., 1]
        """
        out = self.sample_dist(obs, act, sample_mean=sample_mean)

        if self.pred_rwd:
            next_obs = out[..., :-1]
            rwd = out[..., -1:]
        else:
            next_obs = out
            rwd = torch.zeros_like(obs[..., -1:])
        
        if self.residual:
            next_obs += obs

        if self.termination_fn is not None:
            done = self.termination_fn(
                obs.data.cpu().numpy(), act.data.cpu().numpy(), next_obs.data.cpu().numpy()
            )
            done = torch.from_numpy(done).unsqueeze(-1).to(torch.float32).to(self.device)
        else:
            done = torch.zeros(list(obs.shape)[:-1] + [1]).to(torch.float32).to(self.device)
        return next_obs, rwd, done
    
    def compute_loss(self, inputs, targets):
        mu, lv = self.compute_stats_separate(inputs)
        inv_var = torch.exp(-lv)

        mse_loss = torch.mean(torch.pow(mu - targets, 2) * inv_var, dim=-1).mean(0)
        var_loss = torch.mean(lv, dim=-1).mean(0)

        clip_lv_loss = 0.001 * (self.max_lv.sum() - self.min_lv.sum())
        decay_loss = self.compute_decay_loss()

        loss = (
            mse_loss.sum() + var_loss.sum() \
            + clip_lv_loss \
            + decay_loss
        )
        return loss
    
    def compute_decay_loss(self):
        i, loss = 0, 0
        for layer in self.mlp.layers:
            if hasattr(layer, "weight"):
                loss += self.decay[i] * torch.sum(layer.weight ** 2) / 2.
                i += 1
        return loss

    def evaluate(self, inputs, targets):
        """ Compute mean average error for each ensemble 
        
        Returns:
            stats (dict): MAE dict with fields [mae_0, ..., mae_{ensemble_dim}, mae]
        """
        with torch.no_grad():
            mu, _ = self.compute_stats(inputs)
        
        mae = torch.abs(mu - targets.unsqueeze(-2)).mean((0, 2))
        
        stats = {f"mae_{i}": mae[i].cpu().data.item() for i in range(self.ensemble_dim)}
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

def remove_reward_head(state_dict, obs_dim, num_hidden):
    weight_key = "dynamics.mlp.layers.{}.weight".format(2 * (num_hidden + 1))
    bias_key = "dynamics.mlp.layers.{}.bias".format(2 * (num_hidden + 1))
    head_weight = state_dict["model_state_dict"][weight_key]
    head_bias = state_dict["model_state_dict"][bias_key]
    
    if head_weight.shape[-1] > (obs_dim * 2):
        head_weight_mu, head_weight_lv = torch.chunk(head_weight, 2, dim=-1)
        head_bias_mu, head_bias_lv = torch.chunk(head_bias, 2, dim=-1)

        head_weight_mu = head_weight_mu[..., :-1]
        head_weight_lv = head_weight_lv[..., :-1]
        head_bias_mu = head_bias_mu[..., :-1]
        head_bias_lv = head_bias_lv[..., :-1]

        head_weight = torch.cat([head_weight_mu, head_weight_lv], dim=-1)
        head_bias = torch.cat([head_bias_mu, head_bias_lv], dim=-1)

        state_dict["model_state_dict"][weight_key] = head_weight
        state_dict["model_state_dict"][bias_key] = head_bias
        state_dict["model_state_dict"]["dynamics.min_lv"] = state_dict["model_state_dict"]["dynamics.min_lv"][:-1]
        state_dict["model_state_dict"]["dynamics.max_lv"] = state_dict["model_state_dict"]["dynamics.max_lv"][:-1]
    return state_dict
        
def format_samples_for_training(batch, residual=True, pred_rwd=True):
    """ Formate transition samples into inputs and targets 
    
    Args:
        batch (dist): dict of transition torch tensors
        residual (bool, optional): whether to predict residual. Default=True
        pred_rwd (bool, optional): whether to predict reward

    Returns:
        inputs (torch.tensor): model inputs
        targets (torch.tensor): model targets. 
            If residual=True, compute observation difference. 
            If pred_rwd=True, add reward to last dimension.
    """
    obs = batch["obs"]
    act = batch["act"]
    rwd = batch["rwd"]
    next_obs = batch["next_obs"]

    inputs = torch.cat([obs, act], dim=-1)

    if residual:
        targets = next_obs - obs
    else:
        targets = next_obs

    if pred_rwd:
        targets = torch.cat([targets, rwd], dim=-1)
    return inputs, targets

def get_random_index(batch_size, ensemble_dim, bootstrap=True):
    if bootstrap:
        return np.stack([np.random.choice(np.arange(batch_size), batch_size, replace=False) for _ in range(ensemble_dim)]).T
    else:
        idx = np.random.choice(np.arange(batch_size), batch_size, replace=False)
        return np.stack([idx for _ in range(ensemble_dim)]).T

def train_ensemble(
        data, agent, optimizer, eval_ratio, batch_size, epochs, bootstrap=True, grad_clip=None, 
        update_elites=True, max_eval_num=1000, max_epoch_since_update=10, callback=None, debug=False
    ):
    """
    Args:
        data (dict): dict of transitions torch tensors
        agent (nn.Module): agent with dynamics property
        optimizer (torch.optim): optimizer
        eval_ratio (float): evaluation ratio
        batch_size (int): batch size
        epochs (int): max training epochs
        bootstrap (bool): whether to use different minibatch ordering for each ensemble member. Default=True
        grad_clip (float): gradient norm clipping. Default=None
        update_elites (bool): whether to update reward and dynamics topk_dist. Default=True
        max_epoch_since_update (int): max epoch for termination condition. Default=10
        callback (object): callback object. Default=None
        debug (bool): debug flag. If True will print data stats. Default=None

    Returns:
        logger (Logger): logger class with training history
    """
    inputs, targets = format_samples_for_training(
        data, 
        residual=agent.dynamics.residual,
        pred_rwd=agent.dynamics.pred_rwd,
    )

    # train test split
    num_eval = min(int(len(inputs) * eval_ratio), max_eval_num)
    permutation = np.random.permutation(inputs.shape[0])
    train_inputs = inputs[permutation[:-num_eval]]
    train_targets = targets[permutation[:-num_eval]]
    eval_inputs = inputs[permutation[-num_eval:]]
    eval_targets = targets[permutation[-num_eval:]]
    
    # normalize data
    agent.dynamics.scaler.fit(train_inputs)

    if debug:
        print("\ntrain ensemble")
        print("data size train: {}, eval: {}".format(train_inputs.shape, eval_inputs.shape))
        print("train mean", train_inputs.mean(0).numpy().round(3))
        print("train std", train_inputs.std(0).numpy().round(3))
        print("eval mean", eval_inputs.mean(0).numpy().round(3))
        print("eval std", eval_inputs.std(0).numpy().round(3))
    
    logger = Logger()
    start_time = time.time()
    
    ensemble_dim = agent.dynamics.ensemble_dim

    best_eval = [1e6] * ensemble_dim
    best_params_list = parse_ensemble_params(agent.dynamics)
    epoch_since_last_update = 0
    bar = tqdm(range(epochs))
    for e in bar:
        # shuffle train data
        idx_train = get_random_index(train_inputs.shape[0], ensemble_dim, bootstrap=bootstrap)
        
        train_stats_epoch = []
        for i in range(0, train_inputs.shape[0], batch_size):
            idx_batch = idx_train[i:i+batch_size]
            if len(idx_batch) < batch_size: # drop last batch
                continue

            inputs_batch = train_inputs[idx_batch]
            targets_batch = train_targets[idx_batch]

            loss = agent.dynamics.compute_loss(inputs_batch, targets_batch)

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
        eval_stats_epoch = agent.dynamics.evaluate(eval_inputs, eval_targets)
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
        
        bar.set_description("ensemble loss: {:.4f}, mae: {:.4f}, terminate: {}/{}".format(
            stats_epoch["loss"], 
            stats_epoch["mae"],
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