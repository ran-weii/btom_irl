import time
import torch
import torch.nn as nn
import torch.distributions as torch_dist

class LQRBTOM(nn.Module):
    """ Linear quadratic gaussian environment BTOM """
    def __init__(self, agent, rollout_steps, obs_penalty=1., lr=1e-3, decay=0.):
        """
        Args:
            agent (DiscreteAgent): discrete agent
            rollout_steps (int): number of rollout steps. 
                Default to horizon for finite horizon agent
            obs_penalty (float): transition likelihood penalty. Default=1.
            lr (float): learning rate. Default=1e-3
            decay (float): weight decay. Default=0.
        """
        super().__init__()
        assert agent.horizon == 0 # only support infinite horizon agents
        self.state_dim = agent.state_dim
        self.act_dim = agent.act_dim
        self.gamma = agent.gamma
        self.finite_horizon = agent.finite_horizon
        self.rollout_steps = rollout_steps # max rollout steps
        self.obs_penalty = obs_penalty

        if agent.finite_horizon:
            self.rollout_steps = agent.horizon 

        self.agent = agent
        self.optimizer = torch.optim.Adam(
            self.agent.parameters(), lr=lr, weight_decay=decay
        )
    
    def sample_transition(self, s, a, A, B, I):
        mu_s = torch.einsum("ij, ...j -> ...i", A, s)
        mu_a = torch.einsum("ij, ...j -> ...i", B, a)
        s_dist = torch_dist.MultivariateNormal(mu_s + mu_a, I)
        s_next = s_dist.rsample()
        return s_next

    def rollout(self, s0, a0, agent, A, B, I, rollout_steps):
        s = self.sample_transition(s0, a0, A, B, I)
        
        data = {"s": [s0], "a": [a0]}
        for i in range(rollout_steps):
            a = agent.choose_action(s)
            s_next = self.sample_transition(s, a, A, B, I)

            data["s"].append(s)
            data["a"].append(a)

            s = s_next

        data["s"] = torch.stack(data["s"]).transpose(0, 1)
        data["a"] = torch.stack(data["a"]).transpose(0, 1)
        return data

    def compute_reward_cumulents_from_rollout(self, traj, Q, R):
        s = traj["s"]
        a = traj["a"]
        
        r = -self.agent.compute_cost(s, a, Q, R)
        gamma = self.gamma ** torch.arange(r.shape[1]).view(1, -1)
        rho = torch.sum(gamma * r, dim=1)
        return rho

    def compute_value_cumulents_from_rollout(self, traj, A, B, I, Q_t, c_t):
        s = traj["s"]
        a = traj["a"]

        s_next = self.sample_transition(s, a, A, B, I)
        ev = -self.agent.compute_value(s_next, Q_t, c_t)
        gamma = self.gamma ** (1 + torch.arange(self.rollout_steps + 1)).view(1, -1)
        rho = torch.sum(gamma * ev, dim=-1)
        return rho

    def compute_transition_loss(self, s, a, s_next, A, B, I):
        mu = (A.matmul(s.T) + B.matmul(a.T)).T
        p = torch_dist.MultivariateNormal(mu, I)
        logp = p.log_prob(s_next)
        loss = -logp.mean()
        return loss
    
    def compute_action_loss(self, s, a, K, Sigma):
        mu = s.matmul(K.T)
        pi = torch_dist.MultivariateNormal(mu, Sigma)

        logp = pi.log_prob(a)
        loss = -logp.mean()
        return loss

    def fit(self, dataset, epochs, verbose=1):
        """
        Args:
            dataset (dict[np.array]): dict with keys ["s", "a"]. size=[batch_size, seq_len]
        """
        # unpack data 
        s = torch.from_numpy(dataset["s"][:, :-1]).flatten(0, 1).to(torch.float32)
        a = torch.from_numpy(dataset["a"]).flatten(0, 1).to(torch.float32)
        s_next = torch.from_numpy(dataset["s"][:, 1:]).flatten(0, 1).to(torch.float32)

        history = {
            "epoch": [], "total_loss": [], "r_loss": [], "ev_loss": [], 
            "pi_loss": [], "p_loss": [], "time": []
        }
        start = time.time()
        for e in range(epochs):
            A = self.agent.A()
            B = self.agent.B()
            I = self.agent.I()
            Q = self.agent.Q()
            R = self.agent.R()
            
            with torch.no_grad():
                self.agent.plan()
                Q_t = self.agent.Q_t.data
                c_t = self.agent.c_t.data
                K = self.agent.K.data
                Sigma = self.agent.Sigma.data
                a_fake = self.agent.choose_action(s)
                
                real_traj = self.rollout(s, a, self.agent, A, B, I, self.rollout_steps)
                fake_traj = self.rollout(s, a_fake, self.agent, A, B, I, self.rollout_steps)
            
            r_cum_real = self.compute_reward_cumulents_from_rollout(real_traj, Q, R)
            r_cum_fake = self.compute_reward_cumulents_from_rollout(fake_traj, Q, R)
            ev_cum_real = self.compute_value_cumulents_from_rollout(real_traj, A, B, I, Q_t, c_t)
            ev_cum_fake = self.compute_value_cumulents_from_rollout(fake_traj, A, B, I, Q_t, c_t)
            
            transition_loss = self.compute_transition_loss(s, a, s_next, A, B, I)
            r_loss = -(r_cum_real.mean() - r_cum_fake.mean())
            ev_loss = -(ev_cum_real.mean() - ev_cum_fake.mean())
            
            total_loss = (
                r_loss + ev_loss + self.obs_penalty * transition_loss
            )

            total_loss.backward()

            # grad check
            # for n, p in self.agent.named_parameters():
            #     if p.grad is not None:
            #         print(n, p.grad.data.norm())
            #     else:
            #         print(n, None)

            self.optimizer.step()
            self.optimizer.zero_grad()
            
            with torch.no_grad():
                action_loss = self.compute_action_loss(s, a, K, Sigma)
            
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