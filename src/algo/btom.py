import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

# model imports
from src.agents.mbpo import MBPO
from src.utils.data import normalize, denormalize
from src.utils.logging import Logger

class BTOM(MBPO):
    """ Bayesian theory of mind """
    def __init__(
        self, 
        reward,
        dynamics,
        obs_dim, 
        act_dim, 
        act_lim,  
        hidden_dim, 
        num_hidden, 
        activation, 
        gamma=0.9, 
        beta=0.2, 
        polyak=0.995, 
        tune_beta=True,
        state_only=False,
        rwd_clip_max=10.,
        adv_clip_max=6.,
        obs_penalty=10., 
        adv_penalty=1., 
        adv_rollout_steps=5,
        norm_advantage=False,
        update_critic_adv=False,
        buffer_size=1e6, 
        d_batch_size=200, 
        a_batch_size=200,  
        rollout_batch_size=10000, 
        rollout_min_steps=1,
        rollout_max_steps=10,
        rollout_min_epoch=20, 
        rollout_max_epoch=100, 
        model_retain_epochs=5,
        real_ratio=0., 
        eval_ratio=0.2,
        m_steps=100, 
        d_steps=3,
        a_steps=1, 
        lr_d=3e-4,
        lr_a=3e-4, 
        lr_c=3e-4, 
        lr_m=3e-4, 
        grad_clip=None,
        grad_penalty=1., 
        grad_target=1.,
        device=torch.device("cpu")
        ):
        """
        Args:
            reward (EnsembleDynamics): reward function as an EnsembleDynamics object
            dynamics (EnsembleDynamics): transition function as an EnsembleDynamics object
            obs_dim (int): observation dimension
            act_dim (int): action dimension
            act_lim (torch.tensor): action limits
            hidden_dim (int): value network hidden dim
            num_hidden (int): value network hidden layers
            activation (str): value network activation
            gamma (float, optional): discount factor. Default=0.9
            beta (float, optional): softmax temperature. Default=0.2
            polyak (float, optional): target network polyak averaging factor. Default=0.995
            tune_beta (bool, optional): whether to automatically tune temperature. Default=True
            state_only (bool, optional): whether to use state only reward. Default=False
            rwd_clip_max (float, optional): clip reward max value. Default=10.
            adv_clip_max (float, optional): clip advantage max value. Default=6.
            obs_penalty (float, optional): transition likelihood penalty. Default=1..
            adv_penalty (float, optional): model advantage penalty. Default=3e-4.
            adv_rollout_steps (int, optional): model rollout steps for computing adversarial loss. Default=5
            norm_advantage (bool, optional): whether to normalize advantage. Default=False
            update_adv_critic (bool, optional): whether to udpate critic during model update. Default=False
            buffer_size (int, optional): replay buffer size. Default=1e6
            d_batch_size (int, optional): reward batch size. Default=200
            a_batch_size (int, optional): actor and critic batch size. Default=200
            rollout_batch_size (int, optional): model_rollout batch size. Default=10000
            rollout_min_steps (int, optional): initial model rollout steps. Default=1
            rollout_max_steps (int, optional): maximum model rollout steps. Default=10
            rollout_min_epoch (int, optional): epoch to start increasing rollout length. Default=20
            rollout_max_epoch (int, optional): epoch to stop increasing rollout length. Default=100
            model_retain_epochs (int, optional): number of epochs to keep model samples. Default=5
            real_ratio (float, optional): ratio of real samples for policy training. Default=0.
            eval_ratio (float, optional): ratio of real samples for model evaluation. Default=0.2
            m_steps (int, optional): model update steps per training step. Default=100
            d_steps (int, optional): reward update steps per training step. Default=50
            a_steps (int, optional): policy update steps per training step. Default=50
            lr_d (float, optional): reward learning rate. Default=3e-4
            lr_a (float, optional): actor learning rate. Default=3e-4
            lr_c (float, optional): critic learning rate. Default=3e-4
            lr_m (float, optional): model learning rate. Default=3e-4
            grad_clip (float, optional): gradient clipping. Default=None
            grad_penalty (float, optional): gradient penalty weight. Default=1.
            grad_target (float, optional): gradient penalty target. Default1.
            device (optional): training device. Default=cpu
        """
        super().__init__(
            reward, dynamics, obs_dim, act_dim, act_lim, hidden_dim, num_hidden, activation, 
            gamma, beta, polyak, tune_beta, False, buffer_size, a_batch_size, 
            rollout_batch_size, rollout_min_steps, rollout_max_steps, 
            rollout_min_epoch, rollout_max_epoch, model_retain_epochs,
            real_ratio, eval_ratio, m_steps, a_steps, lr_a, lr_c, lr_m, grad_clip, device
        )
        self.state_only = state_only
        self.rwd_clip_max = rwd_clip_max
        self.adv_clip_max = adv_clip_max
        self.obs_penalty = obs_penalty
        self.adv_penalty = adv_penalty
        self.adv_rollout_steps = adv_rollout_steps
        self.norm_advantage = norm_advantage
        self.update_critic_adv = update_critic_adv
        
        self.d_batch_size = d_batch_size
        self.a_batch_size = a_batch_size
        self.d_steps = d_steps
        self.grad_clip = grad_clip
        self.grad_penalty = grad_penalty
        self.grad_target = grad_target
        
        self.optimizers["reward"] = torch.optim.Adam(
            self.reward.parameters(), lr=lr_d
        )

        self.plot_keys = [
            "eval_eps_return_avg", "eval_eps_len_avg", "rwd_loss_avg", "adv_loss_avg", 
            "obs_mae_avg", "log_pi_avg", "critic_loss_avg", "actor_loss_avg",  "beta_avg"
        ]
    
    def update_stats(self):
        obs_mean = self.real_buffer.obs_mean
        obs_variance = self.real_buffer.obs_variance

        self.dynamics.update_stats(obs_mean, obs_variance, obs_mean, obs_variance)
    
    def compute_reward(self, obs, act, done):
        if self.state_only:
            rwd_inputs = torch.cat([obs, done], dim=-1)
        else:
            rwd_inputs = torch.cat([obs, act, done], dim=-1)
        return self.reward(rwd_inputs).clip(-self.rwd_clip_max, self.rwd_clip_max)
    
    def compute_dynamics_adversarial_loss(self, obs, act):
        # sample next obs and rwd
        with torch.no_grad():
            next_obs, done = self.dynamics.step(obs, act)
            obs_norm = normalize(obs, self.dynamics.out_mean, self.dynamics.out_variance)
            next_obs_norm = normalize(next_obs, self.dynamics.out_mean, self.dynamics.out_variance)

            rwd = self.compute_reward(obs * (1 - done), act * (1 - done), done)
            rwd_done = self.compute_reward(
                torch.zeros(1, self.obs_dim).to(torch.float32).to(self.device),
                torch.zeros(1, self.act_dim).to(torch.float32).to(self.device),
                torch.ones(1, 1).to(torch.float32).to(self.device)
            ) # special handle terminal state
        
        # compute advantage
        q_1, q_2 = self.critic(obs, act)
        q = torch.min(q_1, q_2)
        with torch.no_grad():
            next_act, logp = self.sample_action(next_obs)
            q_next_1, q_next_2 = self.critic(next_obs, next_act)
            q_next = torch.min(q_next_1, q_next_2)
            v_next = q_next - self.beta * logp
            v_done = self.gamma / (1 - self.gamma) * rwd_done # special handle terminal state
            advantage = rwd + (1 - done) * self.gamma * v_next + done * v_done - q.data
            
            if self.norm_advantage:
                advantage_norm = (advantage - advantage.mean(0)) / (advantage.std(0) + 1e-6)
            else:
                advantage_norm = advantage
            advantage_norm = advantage_norm.clip(-self.adv_clip_max, self.adv_clip_max)
        
        # compute ensemble mixture log likelihood
        logp_obs = self.dynamics.compute_mixture_log_prob(obs_norm, act, next_obs_norm)
        adv_loss = torch.mean(advantage_norm * logp_obs)

        # update model aware q loss
        with torch.no_grad():
            q_target_next_1, q_target_next_2 = self.critic_target(next_obs, next_act)
            q_target_next = torch.min(q_target_next_1, q_target_next_2)
            v_target_next = q_target_next - self.beta * logp
            q_target = rwd + (1 - done) * self.gamma * v_target_next + done * v_done

        q1_loss = torch.pow(q_1 - q_target, 2).mean()
        q2_loss = torch.pow(q_2 - q_target, 2).mean()
        adv_q_loss = (q1_loss + q2_loss) / 2 

        stats = {
            "v_next_mean": v_next.cpu().mean().data.item(),
            "adv_mean": advantage.cpu().mean().data.item(),
            "adv_std": advantage.cpu().std().data.item(),
            "logp_obs_mean": logp_obs.cpu().mean().data.item(),
            "logp_obs_std": logp_obs.cpu().std().data.item(),
        }
        return adv_loss, adv_q_loss, next_obs, done, stats

    def compute_grad_penalty(self, real_obs, real_act, real_done):
        """ Score matching gradient penalty """
        real_inputs = torch.cat([real_obs, real_act, real_done], dim=-1)
        real_var = Variable(real_inputs, requires_grad=True)
        obs_var, act_var, done_var = torch.split(real_var, [self.obs_dim, self.act_dim, 1], dim=-1)

        rwd = self.compute_reward(obs_var, act_var, done_var)
        
        grad = torch_grad(
            outputs=rwd, inputs=real_var, 
            grad_outputs=torch.ones_like(rwd),
            create_graph=True, retain_graph=True
        )[0]

        grad_norm = torch.linalg.norm(grad, dim=-1)
        grad_pen = torch.pow(grad_norm - self.grad_target, 2).mean()
        return grad_pen
    
    def take_btom_gradient_step(self, real_obs, real_act, real_done, fake_obs, fake_act, fake_done, sl_batch):
        self.reward.train()
        self.dynamics.train()
        
        real_rwd = self.compute_reward(real_obs * (1 - real_done), real_act * (1 - real_done), real_done)
        fake_rwd = self.compute_reward(fake_obs * (1 - fake_done), fake_act * (1 - fake_done), fake_done)
        gp = self.compute_grad_penalty(real_obs * (1 - real_done), real_act * (1 - real_done), real_done)

        real_adv_loss, real_adv_q_loss, real_next_obs, real_done, real_adv_stats = self.compute_dynamics_adversarial_loss(real_obs, real_act)
        fake_adv_loss, fake_adv_q_loss, fake_next_obs, fake_done, fake_adv_stats = self.compute_dynamics_adversarial_loss(fake_obs, fake_act)
        dynamics_loss = self.dynamics.compute_loss(sl_batch["obs"], sl_batch["act"], sl_batch["next_obs"])

        rwd_loss = -(real_rwd.mean() - fake_rwd.mean())
        adv_loss = -(real_adv_loss - fake_adv_loss)
        adv_q_loss = (real_adv_q_loss + fake_adv_q_loss) / 2
        total_loss = (
            rwd_loss + \
            self.grad_penalty * gp + \
            self.adv_penalty * self.gamma * adv_loss + \
            self.obs_penalty * dynamics_loss
        )
        total_loss.backward()
        if self.update_critic_adv:
            adv_q_loss.backward()
        
        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.optimizers["reward"].step()
        self.optimizers["dynamics"].step()
        if self.update_critic_adv:
            self.optimizers["critic"].step()
        self.optimizers["reward"].zero_grad()
        self.optimizers["dynamics"].zero_grad()
        self.optimizers["critic"].zero_grad()
        self.optimizers["actor"].zero_grad()
        
        # update target networks
        if self.update_critic_adv:
            with torch.no_grad():
                for p, p_target in zip(
                    self.critic.parameters(), self.critic_target.parameters()
                ):
                    p_target.data.mul_(self.polyak)
                    p_target.data.add_((1 - self.polyak) * p.data)

        real_adv_stats = {"real_" + k: v for k, v in real_adv_stats.items()}
        fake_adv_stats = {"fake_" + k: v for k, v in fake_adv_stats.items()}
        stats = {
            "rwd_loss": rwd_loss.cpu().data.item(),
            "adv_loss": adv_loss.cpu().data.item(),
            "obs_loss": dynamics_loss.cpu().data.item(),
            "critic_adv_loss": adv_q_loss.cpu().data.item(),
            **real_adv_stats, **fake_adv_stats
        }
        
        self.reward.eval()
        self.dynamics.eval()
        return stats, real_next_obs, real_done, fake_next_obs, fake_done
    
    def train_btom_epoch(self, steps, rollout_steps, logger=None):
        # train test split supervised learning data
        num_total = min(int(self.batch_size * steps * (1 + self.eval_ratio)), self.real_buffer.size)
        num_eval = int(self.eval_ratio * num_total)
        data = self.real_buffer.sample(num_total)

        # normalize data
        data["obs"] = normalize(data["obs"], self.dynamics.obs_mean.cpu(), self.dynamics.obs_variance.cpu())
        data["next_obs"] = normalize(data["next_obs"], self.dynamics.obs_mean.cpu(), self.dynamics.obs_variance.cpu())

        train_data = {k:v[:-num_eval].to(self.device) for k, v in data.items()}
        eval_data = {k:v[-num_eval:].to(self.device) for k, v in data.items()}

        # shuffle train data
        idx_train = np.arange(len(train_data["obs"]))
        np.random.shuffle(idx_train)

        train_stats_epoch = []
        counter = 0
        idx_start_sl = 0
        while counter < steps:
            batch = self.real_buffer.sample(int(self.batch_size/2))
            with torch.no_grad():
                real_done = fake_done = batch["done"].to(self.device)
                real_obs = fake_obs = batch["obs"].to(self.device)
                real_act = batch["act"].to(self.device)
                fake_act = self.choose_action(fake_obs)
            for t in range(rollout_steps):
                # get supervised batch
                idx_batch = idx_train[idx_start_sl:idx_start_sl+self.batch_size]
                train_batch = {k:v[idx_batch] for k, v in train_data.items()}
                
                if t > 0:
                    real_act = self.choose_action(real_obs)
                    fake_act = self.choose_action(fake_obs)
                model_train_stats, real_next_obs, real_done, fake_next_obs, fake_done = self.take_btom_gradient_step(
                    real_obs, real_act, real_done, fake_obs, fake_act, fake_done, train_batch
                )
                train_stats_epoch.append(model_train_stats)
                real_obs = real_next_obs.clone()
                fake_obs = fake_next_obs.clone()
                
                if logger is not None:
                    logger.push(model_train_stats)

                counter += 1
                if counter == steps:
                    break

                # update sl batch counter
                idx_start_sl += self.batch_size
                if idx_start_sl + self.batch_size >= len(train_data["obs"]):
                    idx_start_sl = 0
                    idx_train = np.arange(len(train_data["obs"]))
                    np.random.shuffle(idx_train)

        train_stats_epoch = pd.DataFrame(train_stats_epoch).mean(0).to_dict()

        # evaluate
        dynamics_eval_stats = self.dynamics.evaluate(eval_data["obs"], eval_data["act"], eval_data["next_obs"])
        self.dynamics.update_topk_dist(dynamics_eval_stats)
        with torch.no_grad():
            log_pi = self.compute_action_likelihood(
                denormalize(eval_data["obs"], self.dynamics.obs_mean, self.dynamics.obs_variance), 
                eval_data["act"]
            )
        eval_stats = {
            "obs_mae": dynamics_eval_stats["mae"],
            "log_pi": log_pi.cpu().mean().data.item()
        }
        if logger is not None:
            logger.push(eval_stats)
        
        stats_epoch = {**train_stats_epoch, **eval_stats}
        return stats_epoch
    
    def compute_critic_loss(self, batch, rwd_fn=None):
        obs = batch["obs"].to(self.device)
        act = batch["act"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        done = batch["done"].to(self.device)

        with torch.no_grad():
            if rwd_fn is not None:
                r = rwd_fn(obs, act, done)
                r_done = rwd_fn(
                    torch.zeros(1, self.obs_dim).to(torch.float32).to(self.device),
                    torch.zeros(1, self.act_dim).to(torch.float32).to(self.device),
                    torch.ones(1, 1).to(torch.float32).to(self.device)
                ) # special handle terminal state

            # sample next action
            next_act, logp = self.sample_action(next_obs)

            # compute value target
            q1_next, q2_next = self.critic_target(next_obs, next_act)
            q_next = torch.min(q1_next, q2_next)
            v_next = q_next - self.beta * logp
            v_done = self.gamma / (1 - self.gamma) * r_done # special handle terminal state
            q_target = r + (1 - done) * self.gamma * v_next + done * v_done
        
        q1, q2 = self.critic(obs, act)
        q1_loss = torch.pow(q1 - q_target, 2).mean()
        q2_loss = torch.pow(q2 - q_target, 2).mean()
        q_loss = (q1_loss + q2_loss) / 2 
        return q_loss
    
    def rollout_dynamics(self, obs, done, rollout_steps):
        """ Rollout dynamics model

        Args:
            obs (torch.tensor): observations. size=[batch_size, obs_dim]
            done (torch.tensor): done flag. size=[batch_size, 1]
            rollout_steps (int): number of rollout steps.

        Returns:
            data (dict): size=[rollout_steps, batch_size, dim]
        """
        self.reward.eval()
        self.dynamics.eval()
        
        obs = obs.clone()
        done = done.clone()
        data = {"obs": [], "act": [], "next_obs": [], "rwd": [], "done": []}
        for t in range(rollout_steps):
            with torch.no_grad():
                act = self.choose_action(obs)
                rwd = self.compute_reward(obs, act, done)
                next_obs, done = self.dynamics.step(obs, act)

            data["obs"].append(obs)
            data["act"].append(act)
            data["next_obs"].append(next_obs)
            data["rwd"].append(rwd)
            data["done"].append(done)
            
            obs = next_obs[done.flatten() == 0].clone()
            done = done[done.flatten() == 0].clone()
            if len(obs) == 0:
                break
        
        data["obs"] = torch.cat(data["obs"], dim=0)
        data["act"] = torch.cat(data["act"], dim=0)
        data["next_obs"] = torch.cat(data["next_obs"], dim=0)
        data["rwd"] = torch.cat(data["rwd"], dim=0)
        data["done"] = torch.cat(data["done"], dim=0)
        return data

    def train(
        self, 
        eval_env, 
        epochs, 
        steps_per_epoch, 
        sample_model_every, 
        update_model_every,
        eval_steps=1000,
        num_eval_eps=0, 
        eval_deterministic=True, 
        callback=None, 
        verbose=10
        ):
        logger = Logger()
        start_time = time.time()
        total_steps = epochs * steps_per_epoch
        
        epoch = 0
        for t in range(total_steps):
            # train reward and dynamics
            if (t + 1) % update_model_every == 0:
                btom_stats_dict = self.train_btom_epoch(self.m_steps, self.adv_rollout_steps, logger=logger)
                if (t + 1) % verbose == 0:
                    round_loss_dict = {k: round(v, 3) for k, v in btom_stats_dict.items()}
                    print(f"e: {epoch + 1}, t model: {t + 1}, {round_loss_dict}")
                    
            # sample model
            if t == 0 or (t + 1) % sample_model_every == 0:
                rollout_steps = self.compute_rollout_steps(epoch + 1)
                self.replay_buffer.max_size = min(
                    self.buffer_size, int(self.model_retain_epochs * self.rollout_batch_size * rollout_steps)
                )
                self.sample_imagined_data(
                    self.rollout_batch_size, rollout_steps, mix=False
                )
                print("rollout steps", rollout_steps, "replay buffer size", self.replay_buffer.size)

            # train policy
            policy_stats_dict = self.train_policy_epoch(self.compute_reward, logger=logger)
            if (t + 1) % verbose == 0:
                round_loss_dict = {k: round(v, 3) for k, v in policy_stats_dict.items()}
                print(f"e: {epoch + 1}, t policy: {t + 1}, {round_loss_dict}")

            # end of epoch handeling
            if (t + 1) % steps_per_epoch == 0: 
                epoch = (t + 1) // steps_per_epoch

                # evaluate episodes
                if num_eval_eps > 0:
                    eval_eps = []
                    for i in range(num_eval_eps):
                        eval_eps.append(self.rollout(eval_env, eval_steps, sample_mean=eval_deterministic))
                        logger.push({"eval_eps_return": sum(eval_eps[-1]["rwd"])})
                        logger.push({"eval_eps_len": sum(1 - eval_eps[-1]["done"])})

                logger.push({"epoch": epoch + 1})
                logger.push({"time": time.time() - start_time})
                logger.log()
                print()
            
                if callback is not None:
                    callback(self, pd.DataFrame(logger.history))
        
        return logger

