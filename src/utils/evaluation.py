import numpy as np
import torch

def rollout(env, agent, max_steps, sample_mean=False):
    obs = env.reset()[0]

    data = {"obs": [], "act": [], "next_obs": [], "rwd": [], "done": []}
    for t in range(max_steps):
        with torch.no_grad():
            act = agent.choose_action(
                torch.from_numpy(obs).to(torch.float32).to(agent.device),
                sample_mean=sample_mean
            ).cpu().numpy()
        next_obs, rwd, terminated, _, _ = env.step(act)
        
        data["obs"].append(obs)
        data["act"].append(act)
        data["next_obs"].append(next_obs)
        data["rwd"].append(rwd)
        data["done"].append(terminated)

        if terminated:
            break
        
        obs = next_obs

    data["obs"] = torch.from_numpy(np.stack(data["obs"])).to(torch.float32)
    data["act"] = torch.from_numpy(np.stack(data["act"])).to(torch.float32)
    data["next_obs"] = torch.from_numpy(np.stack(data["next_obs"])).to(torch.float32)
    data["rwd"] = torch.from_numpy(np.stack(data["rwd"])).to(torch.float32)
    data["done"] = torch.from_numpy(np.stack(data["done"])).to(torch.float32)
    return data

def evaluate_episodes(eval_env, agent, num_eval_eps, max_steps, eval_deterministic=True, logger=None):
    eval_eps = []
    eval_returns = []
    eval_lens = []
    for i in range(num_eval_eps):
        eval_eps.append(rollout(eval_env, agent, max_steps, sample_mean=eval_deterministic))
        eval_returns.append(sum(eval_eps[-1]["rwd"]))
        eval_lens.append(sum(1 - eval_eps[-1]["done"]))
    
    if logger is not None:
        logger.push({"eval_eps_return_mean": np.mean(eval_returns)})
        logger.push({"eval_eps_return_std": np.std(eval_returns)})
        logger.push({"eval_eps_return_max": np.max(eval_returns)})
        logger.push({"eval_eps_return_min": np.min(eval_returns)})
        
        logger.push({"eval_eps_len_mean": np.mean(eval_lens)})
        logger.push({"eval_eps_len_std": np.mean(eval_lens)})
        logger.push({"eval_eps_len_max": np.max(eval_lens)})
        logger.push({"eval_eps_len_min": np.min(eval_lens)})
    return eval_eps, eval_returns, eval_lens

def evaluate_policy(batch, policy, logger=None):
    """ Evaluate policy likelihood and mae """
    obs = batch["obs"].to(policy.device)
    act = batch["act"].to(policy.device)
    
    with torch.no_grad():
        log_pi = policy.compute_action_likelihood(obs, act).mean()
        mu, _ = policy.sample_action(obs, sample_mean=True)
        mae = torch.abs(mu - act).mean()

    if logger is not None:
        logger.push({"log_pi": log_pi.cpu().data.item()})
        logger.push({"pi_mae": mae.cpu().data.item()})
    return log_pi, mae