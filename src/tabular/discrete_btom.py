import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscreteBTOM(nn.Module):
    """ Discrete environment BTOM with state-only reward """
    def __init__(self, agent, rollout_steps, exact=True, obs_penalty=1., lr=1e-3, decay=0.):
        """
        Args:
            agent (DiscreteAgent): discrete agent
            rollout_steps (int): number of rollout steps. 
                Default to horizon for finite horizon agent
            exact (bool): whether to compute occupancy exactly. 
                Default to horizon for finite horizon agent. Default=True
            obs_penalty (float): transition likelihood penalty. Default=1.
            lr (float): learning rate. Default=1e-3
            decay (float): weight decay. Default=0.
        """
        super().__init__()
        self.state_dim = agent.state_dim
        self.act_dim = agent.act_dim
        self.gamma = agent.gamma
        self.finite_horizon = agent.finite_horizon
        self.rollout_steps = rollout_steps # max rollout steps
        self.exact = exact
        self.obs_penalty = obs_penalty

        if agent.finite_horizon:
            self.rollout_steps = agent.horizon 
            self.exact = True

        self.agent = agent
        self.optimizer = torch.optim.Adam(
            self.agent.parameters(), lr=lr, weight_decay=decay
        )
    
    def rollout(self, s0, a0, pi, transition, rollout_steps):
        s = torch.multinomial(transition[a0, s0], 1).flatten()
        
        data = {"s": [s0], "a": [a0]}
        for i in range(rollout_steps):
            a = torch.multinomial(pi[s], 1).flatten()
            s_next = torch.multinomial(transition[a, s], 1).flatten()

            data["s"].append(s)
            data["a"].append(a)

            s = s_next

        data["s"] = torch.stack(data["s"]).T
        data["a"] = torch.stack(data["a"]).T
        return data
    
    def compute_state_action_marginal(self, s, pi0, pi, transition, rollout_steps):
        """ Compute marginal state-action sequence using forward propagation 
        
        Args:
            s (torch.tensor): initial state. size=[batch_size]
            p0 (torch.tenosr): initial action distribution. size=[batch_size, act_dim]
            pi (torch.tensor): policy. size=[horizon, state_dim, act_dim] for finit horizon
                or size=[state_dim, act_dim] for infinite horizon
            transition (torch.tensor): transition matrix. size=[act_dim, state_dim, state_dim]
            rollout_steps (int): rollout steps

        Returns:
            traj (torch.tensor): marginal state-action distribution. size=[horizon, batch_size, state_dim, act_dim]
        """
        traj0 = torch.zeros(len(s), self.state_dim, self.act_dim)
        traj0[torch.arange(len(s)), s] = pi0
        traj = [traj0] + [torch.empty(0)] * rollout_steps
        for h in range(rollout_steps):
            s_next = torch.einsum("kij, nik -> nj", transition, traj[h])
            if self.finite_horizon:
                traj[h+1] = s_next.unsqueeze(-1) * pi[-h-2].unsqueeze(0)
            else:
                traj[h+1] = s_next.unsqueeze(-1) * pi.unsqueeze(0)
        
        traj = torch.stack(traj)
        return traj

    def compute_reward_cumulents_from_marginal(self, traj, reward):
        r = torch.einsum("hnik, ik -> nh", traj, reward)
        gamma = self.gamma ** torch.arange(r.shape[1]).view(1, -1)
        rho = torch.sum(gamma * r, dim=1)
        return rho
    
    def compute_reward_cumulents_from_rollout(self, traj, reward):
        s = traj["s"]
        a = traj["a"]

        r = reward[s, a]
        gamma = self.gamma ** torch.arange(r.shape[1]).view(1, -1)
        rho = torch.sum(gamma * r, dim=1)
        return rho

    def compute_value_cumulents_from_marginal(self, traj, transition, value):
        if self.finite_horizon:
            ev = torch.einsum("kij, hj -> hik", transition, value)
            traj_ev = torch.einsum("hnik, hik -> nh", traj[:-1], ev[1:])
        else:
            ev = torch.einsum("kij, j -> ik", transition, value)
            traj_ev = torch.einsum("hnik, ik -> nh", traj, ev)
        gamma = self.gamma ** (1 + torch.arange(traj_ev.shape[1])).view(1, -1)
        rho = torch.sum(gamma * traj_ev, dim=-1)
        return rho

    def compute_value_cumulents_from_rollout(self, traj, transition, value):
        s = traj["s"]
        a = traj["a"]
        
        traj_ev = torch.sum(transition[a, s] * value.view(1, 1, -1), dim=-1)
        gamma = self.gamma ** (1 + torch.arange(self.rollout_steps + 1)).view(1, -1)
        rho = torch.sum(gamma * traj_ev, dim=-1)
        return rho

    def compute_transition_loss(self, s, a, s_next, transition):
        logp = torch.log(transition[a, s, s_next] + 1e-6)
        loss = -logp.mean()
        return loss
    
    def compute_action_loss(self, s, a, pi):
        if self.finite_horizon:
            pi = pi[-1]
        logp = torch.log(pi[s, a] + 1e-6)
        loss = -logp.mean()
        return loss

    def fit(self, dataset, epochs, verbose=1):
        """
        Args:
            dataset (dict[np.array]): dict with keys ["s", "a"]. size=[batch_size, seq_len]
        """
        # unpack data
        s = torch.from_numpy(dataset["s"][:, :-1]).flatten()
        a = torch.from_numpy(dataset["a"]).flatten()
        s_next = torch.from_numpy(dataset["s"][:, 1:]).flatten()

        history = {
            "epoch": [], "total_loss": [], "r_loss": [], "ev_loss": [], 
            "pi_loss": [], "p_loss": [], "time": []
        }
        start = time.time()
        for e in range(epochs):
            transition = self.agent.transition()
            reward = self.agent.reward()
            r = reward.view(-1, 1).repeat_interleave(self.act_dim, -1)
            
            with torch.no_grad():
                self.agent.plan()
                pi = self.agent.pi
                v = self.agent.v

                pi_data = F.one_hot(a, num_classes=self.act_dim).to(torch.float32)
                if self.finite_horizon:
                    pi_fake = pi[-1][s]
                else:
                    pi_fake = pi[s]
                a_fake = self.agent.choose_action(s)
                
                if self.exact:
                    real_traj = self.compute_state_action_marginal(s, pi_data, pi, transition.data, self.rollout_steps)
                    fake_traj = self.compute_state_action_marginal(s, pi_fake, pi, transition.data, self.rollout_steps)
                else:
                    real_traj = self.rollout(s, a, pi, transition.data, self.rollout_steps)
                    fake_traj = self.rollout(s, a_fake, pi, transition.data, self.rollout_steps)
            
            if self.exact:
                r_cum_real = self.compute_reward_cumulents_from_marginal(real_traj.data, r)
                r_cum_fake = self.compute_reward_cumulents_from_marginal(fake_traj.data, r)
                ev_cum_real = self.compute_value_cumulents_from_marginal(real_traj.data, transition, v.data)
                ev_cum_fake = self.compute_value_cumulents_from_marginal(fake_traj.data, transition, v.data)
            else:
                r_cum_real = self.compute_reward_cumulents_from_rollout(real_traj, r)
                r_cum_fake = self.compute_reward_cumulents_from_rollout(fake_traj, r)
                ev_cum_real = self.compute_value_cumulents_from_rollout(real_traj, transition, v.data)
                ev_cum_fake = self.compute_value_cumulents_from_rollout(fake_traj, transition, v.data)
            
            transition_loss = self.compute_transition_loss(s, a, s_next, transition)
            r_loss = -(r_cum_real.mean() - r_cum_fake.mean())
            ev_loss = -(ev_cum_real.mean() - ev_cum_fake.mean())
            
            total_loss = (
                r_loss + ev_loss + self.obs_penalty * transition_loss
            )

            total_loss.backward()

            # # grad check
            # for n, p in self.agent.named_parameters():
            #     if p.grad is not None:
            #         print(n, p.grad.data.norm())
            #     else:
            #         print(n, None)

            self.optimizer.step()
            self.optimizer.zero_grad()

            with torch.no_grad():
                action_loss = self.compute_action_loss(s, a, pi.data)
            
            history["epoch"].append(e + 1)
            history["total_loss"].append(total_loss.data.item())
            history["r_loss"].append(r_loss.data.item())
            history["ev_loss"].append(ev_loss.data.item())
            history["pi_loss"].append(action_loss.data.item())
            history["p_loss"].append(transition_loss.data.item())
            history["time"].append(time.time() - start)
            
            if (e + 1) % verbose == 0:
                print("e: {}, loss: {:.4f}, r: {:.4f}, ev: {:.4f}, pi:{:.4f}, p: {:.4f}, t: {:.4f}".format(
                    e + 1,
                    history["total_loss"][-1],
                    history["r_loss"][-1],
                    history["ev_loss"][-1],
                    history["pi_loss"][-1],
                    history["p_loss"][-1],
                    history["time"][-1],
                ))

        return history
        