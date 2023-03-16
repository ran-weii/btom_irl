import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscreteMCEIRL(nn.Module):
    """ Discrete environment MCEIRL with state-only reward """
    def __init__(self, agent, state_marginal, rollout_steps, pess_penalty=0., exact=True, lr=1e-3, decay=0.):
        """
        Args:
            agent (DiscreteAgent): discrete agent with infinite horizon
            state_marginal (torch.tensor): empirical state marginal density. size=[state_dim]
            rollout_steps (int): number of rollout steps. 
                Default to horizon for finite horizon agent
            pess_penalty (float): pessimistic penalty. Default=0.
            exact (bool): whether to compute occupancy exactly. 
                Default to horizon for finite horizon agent. Default=True
            obs_penalty (float): transition likelihood penalty. Default=1.
            lr (float): learning rate. Default=1e-3
            decay (float): weight decay. Default=0.
        """
        super().__init__()
        assert agent.finite_horizon == False

        self.state_dim = agent.state_dim
        self.act_dim = agent.act_dim
        self.gamma = agent.gamma
        self.finite_horizon = agent.finite_horizon
        self.rollout_steps = rollout_steps # max rollout steps
        self.exact = exact
        self.pess_penalty = pess_penalty

        self.agent = agent
        self.state_marginal = nn.Parameter(state_marginal, requires_grad=False)
        self.optimizer = torch.optim.Adam(
            self.agent.parameters(), lr=lr, weight_decay=decay
        )
    
    def rollout(self, s, pi, transition, rollout_steps):
        data = {"s": [], "a": []}
        for i in range(rollout_steps):
            a = torch.multinomial(pi[s], 1).flatten()
            s_next = torch.multinomial(transition[a, s], 1).flatten()

            data["s"].append(s)
            data["a"].append(a)

            s = s_next

        data["s"] = torch.stack(data["s"]).T
        data["a"] = torch.stack(data["a"]).T
        return data
    
    def compute_state_action_marginal(self, s, pi, transition, rollout_steps):
        """ Compute marginal state-action sequence using forward propagation 
        
        Args:
            s (torch.tensor): initial state. size=[batch_size]
            pi (torch.tensor): policy. size=[horizon, state_dim, act_dim] for finit horizon
                or size=[state_dim, act_dim] for infinite horizon
            transition (torch.tensor): transition matrix. size=[act_dim, state_dim, state_dim]
            rollout_steps (int): rollout steps

        Returns:
            traj (torch.tensor): marginal state-action distribution. size=[horizon, batch_size, state_dim, act_dim]
        """
        traj0 = torch.zeros(len(s), self.state_dim, self.act_dim)
        traj0[torch.arange(len(s)), s] = pi[s]
        traj = [traj0] + [torch.empty(0)] * rollout_steps
        for h in range(rollout_steps):
            s_next = torch.einsum("kij, nik -> nj", transition, traj[h])
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

    def compute_transition_loss(self, s, a, s_next, transition):
        logp = torch.log(transition[a, s, s_next] + 1e-6)
        loss = -logp.mean()
        return loss
    
    def compute_action_loss(self, s, a, pi):
        logp = torch.log(pi[s, a] + 1e-6)
        loss = -logp.mean()
        return loss

    def fit(self, dataset, epochs, verbose=1):
        """
        Args:
            dataset (dict[np.array]): dict with keys ["s", "a"]. size=[batch_size, seq_len]
        """
        # unpack data
        self.rollout_steps = min(self.rollout_steps, dataset["a"].shape[1])
        s0 = torch.from_numpy(dataset["s"][:, 0])
        s = torch.from_numpy(dataset["s"][:, :-1]).flatten()
        a = torch.from_numpy(dataset["a"]).flatten()
        s_next = torch.from_numpy(dataset["s"][:, 1:]).flatten()
        
        real_traj = {
            "s": torch.from_numpy(dataset["s"][:, :self.rollout_steps]),
            "a": torch.from_numpy(dataset["a"][:, :self.rollout_steps])
        }

        history = {
            "epoch": [], "r_loss": [], "pi_loss": [], "p_loss": [], "time": []
        }
        start = time.time()
        for e in range(epochs):
            transition = self.agent.transition()
            reward = self.agent.reward() + self.pess_penalty * torch.log(self.state_marginal + 1e-6)
            r = reward.view(-1, 1).repeat_interleave(self.act_dim, -1)
            
            with torch.no_grad():
                self.agent.plan(bonus=self.pess_penalty * torch.log(self.state_marginal + 1e-6))
                pi = self.agent.pi
                
                if self.exact:
                    fake_traj = self.compute_state_action_marginal(s0, pi, transition.data, self.rollout_steps)
                else:
                    fake_traj = self.rollout(s0, pi, transition.data, self.rollout_steps)
            
            r_cum_real = self.compute_reward_cumulents_from_rollout(real_traj, r)
            if self.exact:
                r_cum_fake = self.compute_reward_cumulents_from_marginal(fake_traj.data, r)
            else:
                r_cum_fake = self.compute_reward_cumulents_from_rollout(fake_traj, r)
            
            r_loss = -(r_cum_real.mean() - r_cum_fake.mean())
            r_loss.backward()

            # grad check
            # for n, p in self.named_parameters():
            #     if p.grad is not None:
            #         print(n, p.grad.data.norm())
            #     else:
            #         print(n, None)

            self.optimizer.step()
            self.optimizer.zero_grad()

            with torch.no_grad():
                action_loss = self.compute_action_loss(s, a, pi.data)
                transition_loss = self.compute_transition_loss(s, a, s_next, transition)
            
            history["epoch"].append(e + 1)
            history["r_loss"].append(r_loss.data.item())
            history["pi_loss"].append(action_loss.data.item())
            history["p_loss"].append(transition_loss.data.item())
            history["time"].append(time.time() - start)
            
            if (e + 1) % verbose == 0:
                print("e: {}, r: {:.4f}, pi:{:.4f}, p: {:.4f}, t: {:.4f}".format(
                    e + 1,
                    history["r_loss"][-1],
                    history["pi_loss"][-1],
                    history["p_loss"][-1],
                    history["time"][-1],
                ))

        return history
        