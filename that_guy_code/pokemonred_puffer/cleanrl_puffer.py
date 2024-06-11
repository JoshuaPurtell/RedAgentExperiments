import os
import random
import time
import uuid
from collections import defaultdict
from datetime import timedelta
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pufferlib
import pufferlib.emulation
import pufferlib.frameworks.cleanrl
import pufferlib.policy_pool
import pufferlib.utils
import pufferlib.vectorization
import torch
import torch.nn as nn
import torch.optim as optim
from that_guy_code.pokemonred_puffer.eval import make_pokemon_red_overlay


@pufferlib.dataclass
class Performance:
    total_uptime = 0
    total_updates = 0
    total_agent_steps = 0
    epoch_time = 0
    epoch_sps = 0
    evaluation_time = 0
    evaluation_sps = 0
    evaluation_memory = 0
    evaluation_pytorch_memory = 0
    env_time = 0
    env_sps = 0
    inference_time = 0
    inference_sps = 0
    train_time = 0
    train_sps = 0
    train_memory = 0
    train_pytorch_memory = 0


@pufferlib.dataclass
class Losses:
    policy_loss = 0
    value_loss = 0
    entropy = 0
    old_approx_kl = 0
    approx_kl = 0
    clipfrac = 0
    explained_variance = 0


@pufferlib.dataclass
class Charts:
    global_step = 0
    SPS = 0
    learning_rate = 0


def rollout(
    env_creator,
    env_kwargs,
    agent_creator,
    agent_kwargs,
    model_path=None,
    device="cuda",
    verbose=True,
):
    env = env_creator(**env_kwargs)
    if model_path is None:
        agent = agent_creator(env, **agent_kwargs)
    else:
        agent = torch.load(model_path, map_location=device)

    terminal = truncated = True

    while True:
        if terminal or truncated:
            if verbose:
                print("---  Reset  ---")

            ob, info = env.reset()
            state = None
            step = 0
            return_val = 0

        ob = torch.tensor(ob, device=device).unsqueeze(0)
        with torch.no_grad():
            if hasattr(agent, "lstm"):
                action, _, _, _, state = agent.get_action_and_value(ob, state)
            else:
                action, _, _, _ = agent.get_action_and_value(ob)

        ob, reward, terminal, truncated, _ = env.step(action[0].item())
        return_val += reward

        chars = env.render()
        print("\033c", end="")
        print(chars)

        if verbose:
            print(f"Step: {step} Reward: {reward:.4f} Return: {return_val:.2f}")

        time.sleep(0.5)
        step += 1


def seed_everything(seed, torch_deterministic):
    random.seed(seed)
    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic


def unroll_nested_dict(d):
    if not isinstance(d, dict):
        return d

    for k, v in d.items():
        if isinstance(v, dict):
            for k2, v2 in unroll_nested_dict(v):
                yield f"{k}/{k2}", v2
        else:
            yield k, v


def print_dashboard(stats, init_performance, performance):
    output = []
    data = {**stats, **init_performance, **performance}

    grouped_data = defaultdict(dict)

    for k, v in data.items():
        if k == "total_uptime":
            v = timedelta(seconds=v)
        if "memory" in k:
            v = pufferlib.utils.format_bytes(v)
        elif "time" in k:
            try:
                v = f"{v:.2f} s"
            except:
                pass

        first_word, *rest_words = k.split("_")
        rest_words = " ".join(rest_words).title()

        grouped_data[first_word][rest_words] = v

    for main_key, sub_dict in grouped_data.items():
        output.append(f"{main_key.title()}")
        for sub_key, sub_value in sub_dict.items():
            output.append(f"    {sub_key}: {sub_value}")

    print("\033c", end="")
    print("\n".join(output))
    time.sleep(1 / 20)


# TODO: Make this an unfrozen dataclass with a post_init?
class CleanPuffeRL:
    def __init__(
        self,
        config: SimpleNamespace | None = None,
        exp_name: str | None = None,
        track: bool = False,
        # Agent
        agent: nn.Module | None = None,
        agent_creator: Callable[..., Any] | None = None,
        agent_kwargs: dict = None,
        # Environment
        env_creator: Callable[..., Any] | None = None,
        env_creator_kwargs: dict | None = None,
        vectorization: ... = pufferlib.vectorization.Serial,
        # Policy Pool options
        policy_selector: Callable[
            [list[Any], int], list[Any]
        ] = pufferlib.policy_pool.random_selector,
    ):
        self.config = config
        if self.config is None:
            self.config = pufferlib.args.CleanPuffeRL()

        self.exp_name = exp_name
        if exp_name is None:
            exp_name = str(uuid.uuid4())[:8]

        self.wandb = None
        if track:
            import wandb

            self.wandb = wandb

        self.start_time = time.time()
        seed_everything(config.seed, config.torch_deterministic)
        self.total_updates = config.total_timesteps // config.batch_size

        self.device = config.device
        self.obs_device = "cpu" if config.cpu_offload else self.device

        # Create environments, agent, and optimizer
        init_profiler = pufferlib.utils.Profiler(memory=True)
        with init_profiler:
            self.pool = vectorization(
                env_creator,
                env_kwargs=env_creator_kwargs,
                num_envs=config.num_envs,
                envs_per_worker=config.envs_per_worker,
                envs_per_batch=config.envs_per_batch,
                env_pool=config.env_pool,
            )

        obs_shape = self.pool.single_observation_space.shape
        atn_shape = self.pool.single_action_space.shape
        self.num_agents = self.pool.agents_per_env
        total_agents = self.num_agents * config.num_envs

        # If data_dir is provided, load the resume state
        resume_state = {}
        path = os.path.join(config.data_dir, exp_name)
        if os.path.exists(path):
            trainer_path = os.path.join(path, "trainer_state.pt")
            resume_state = torch.load(trainer_path)
            model_path = os.path.join(path, resume_state["model_name"])
            self.agent = torch.load(model_path, map_location=self.device)
            print(
                f'Resumed from update {resume_state["update"]} '
                f'with policy {resume_state["model_name"]}'
            )
        else:
            self.agent = pufferlib.emulation.make_object(
                agent, agent_creator, [self.pool.driver_env], agent_kwargs
            )

        self.global_step = resume_state.get("global_step", 0)
        self.agent_step = resume_state.get("agent_step", 0)
        self.update = resume_state.get("update", 0)

        self.optimizer = optim.Adam(self.agent.parameters(), lr=config.learning_rate, eps=1e-5)
        self.opt_state = resume_state.get("optimizer_state_dict", None)
        if self.opt_state is not None:
            self.optimizer.load_state_dict(resume_state["optimizer_state_dict"])

        # Create policy pool
        pool_agents = self.num_agents * self.pool.envs_per_batch
        self.policy_pool = pufferlib.policy_pool.PolicyPool(
            self.agent,
            pool_agents,
            atn_shape,
            self.device,
            path,
            self.config.pool_kernel,
            policy_selector,
        )

        # Allocate Storage
        storage_profiler = pufferlib.utils.Profiler(memory=True, pytorch_memory=True).start()
        self.pool.async_reset(config.seed)
        self.next_lstm_state = None
        if hasattr(self.agent, "lstm"):
            shape = (self.agent.lstm.num_layers, total_agents, self.agent.lstm.hidden_size)
            self.next_lstm_state = (
                torch.zeros(shape, device=self.device),
                torch.zeros(shape, device=self.device),
            )
        self.obs = torch.zeros(config.batch_size + 1, *obs_shape, device=self.obs_device)
        self.actions = torch.zeros(config.batch_size + 1, *atn_shape, dtype=int, device=self.device)
        self.logprobs = torch.zeros(config.batch_size + 1, device=self.device)
        self.rewards = torch.zeros(config.batch_size + 1, device=self.device)
        self.dones = torch.zeros(config.batch_size + 1, device=self.device)
        self.truncateds = torch.zeros(config.batch_size + 1, device=self.device)
        self.values = torch.zeros(config.batch_size + 1, device=self.device)
        storage_profiler.stop()

        # "charts/actions": wandb.Histogram(b_actions.cpu().numpy()),
        self.init_performance = pufferlib.namespace(
            init_time=time.time() - self.start_time,
            init_env_time=init_profiler.elapsed,
            init_env_memory=init_profiler.memory,
            tensor_memory=storage_profiler.memory,
            tensor_pytorch_memory=storage_profiler.pytorch_memory,
        )

        self.sort_keys = []
        self.learning_rate = (config.learning_rate,)
        self.losses = Losses()
        self.performance = Performance()

    @pufferlib.utils.profile
    def evaluate(self):
        config = self.config
        # TODO: Handle update on resume
        if self.wandb is not None and self.performance.total_uptime > 0:
            self.wandb.log(
                {
                    "SPS": self.SPS,
                    "global_step": self.global_step,
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                    **{f"losses/{k}": v for k, v in self.losses.items()},
                    **{f"performance/{k}": v for k, v in self.performance.items()},
                    **{f"stats/{k}": v for k, v in self.stats.items()},
                    **{f"max_stats/{k}": v for k, v in self.max_stats.items()},
                    **{
                        f"skillrank/{policy}": elo
                        for policy, elo in self.policy_pool.ranker.ratings.items()
                    },
                }
            )

        self.policy_pool.update_policies()
        performance = defaultdict(list)
        env_profiler = pufferlib.utils.Profiler()
        inference_profiler = pufferlib.utils.Profiler()
        with pufferlib.utils.Profiler(memory=True, pytorch_memory=True) as eval_profiler:
            ptr = step = padded_steps_collected = agent_steps_collected = 0
            infos = defaultdict(lambda: defaultdict(list))
            while True:
                step += 1
                if ptr == config.batch_size + 1:
                    break

                with env_profiler:
                    o, r, d, t, i, env_id, mask = self.pool.recv()

                i = self.policy_pool.update_scores(i, "return")

                with inference_profiler, torch.no_grad():
                    o = torch.as_tensor(o).to(device=self.device, non_blocking=True)
                    r = (
                        torch.as_tensor(r, dtype=torch.float32)
                        .to(device=self.device, non_blocking=True)
                        .view(-1)
                    )
                    d = (
                        torch.as_tensor(d, dtype=torch.float32)
                        .to(device=self.device, non_blocking=True)
                        .view(-1)
                    )

                agent_steps_collected += sum(mask)
                padded_steps_collected += len(mask)
                with inference_profiler, torch.no_grad():
                    # Multiple policies will not work with new envpool
                    next_lstm_state = self.next_lstm_state
                    if next_lstm_state is not None:
                        next_lstm_state = (
                            next_lstm_state[0][:, env_id],
                            next_lstm_state[1][:, env_id],
                        )

                    actions, logprob, value, next_lstm_state = self.policy_pool.forwards(
                        o, next_lstm_state
                    )

                    if next_lstm_state is not None:
                        h, c = next_lstm_state
                        self.next_lstm_state[0][:, env_id] = h
                        self.next_lstm_state[1][:, env_id] = c

                    value = value.flatten()

                # Index alive mask with policy pool idxs...
                # TODO: Find a way to avoid having to do this
                learner_mask = mask * self.policy_pool.mask

                for idx in np.where(learner_mask)[0]:
                    if ptr == config.batch_size + 1:
                        break
                    self.obs[ptr] = o[idx]
                    self.values[ptr] = value[idx]
                    self.actions[ptr] = actions[idx]
                    self.logprobs[ptr] = logprob[idx]
                    self.sort_keys.append((env_id[idx], step))
                    if len(d) != 0:
                        self.rewards[ptr] = r[idx]
                        self.dones[ptr] = d[idx]
                    ptr += 1

                for policy_name, policy_i in i.items():
                    for agent_i in policy_i:
                        for name, dat in unroll_nested_dict(agent_i):
                            infos[policy_name][name].append(dat)

                with env_profiler:
                    self.pool.send(actions.to(device="cpu", non_blocking=True).numpy())

        self.global_step += padded_steps_collected
        self.reward = torch.mean(self.rewards).float().item()
        self.SPS = int(padded_steps_collected / eval_profiler.elapsed)

        perf = self.performance
        perf.total_uptime = int(time.time() - self.start_time)
        perf.total_agent_steps = self.global_step
        perf.env_time = env_profiler.elapsed
        perf.env_sps = int(agent_steps_collected / env_profiler.elapsed)
        perf.inference_time = inference_profiler.elapsed
        perf.inference_sps = int(padded_steps_collected / inference_profiler.elapsed)
        perf.eval_time = eval_profiler.elapsed
        perf.eval_sps = int(padded_steps_collected / eval_profiler.elapsed)
        perf.eval_memory = eval_profiler.end_mem
        perf.eval_pytorch_memory = eval_profiler.end_torch_mem

        self.stats = {}
        self.max_stats = {}
        for k, v in infos["learner"].items():
            if "Task_eval_fn" in k:
                # Temporary hack for NMMO competition
                continue
            if "pokemon_exploration_map" in k:
                overlay = make_pokemon_red_overlay(np.stack(v, axis=0))
                if self.wandb is not None:
                    self.stats["Media/aggregate_exploration_map"] = self.wandb.Image(overlay)
            try:  # TODO: Better checks on log data types
                self.stats[k] = np.mean(v)
                self.max_stats[k] = np.max(v)
            except:
                continue

        if config.verbose:
            print_dashboard(self.stats, self.init_performance, self.performance)

        return self.stats, infos

    @pufferlib.utils.profile
    def train(self):
        if self.done_training():
            raise RuntimeError(f"Max training updates {self.total_updates} already reached")

        config = self.config
        # assert data.num_steps % bptt_horizon == 0, "num_steps must be divisible by bptt_horizon"
        with pufferlib.utils.Profiler(memory=True, pytorch_memory=True) as train_profiler:
            # Anneal learning rate
            frac = 1.0 - (self.update - 1.0) / self.total_updates
            lrnow = frac * config.learning_rate
            self.optimizer.param_groups[0]["lr"] = lrnow

            if config.anneal_lr:
                frac = 1.0 - (self.update - 1.0) / self.total_updates
                lrnow = frac * config.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            num_minibatches = config.batch_size // config.bptt_horizon // config.batch_rows
            assert (
                num_minibatches > 0
            ), "config.batch_size // config.bptt_horizon // config.batch_rows must be > 0"
            idxs = sorted(range(len(self.sort_keys)), key=self.sort_keys.__getitem__)
            self.sort_keys = []
            b_idxs = (
                torch.tensor(idxs, dtype=torch.long)[:-1]
                .reshape(config.batch_rows, num_minibatches, config.bptt_horizon)
                .transpose(0, 1)
            )

            # bootstrap value if not done
            with torch.no_grad():
                advantages = torch.zeros(config.batch_size, device=self.device)
                lastgaelam = 0
                for t in reversed(range(config.batch_size)):
                    i, i_nxt = idxs[t], idxs[t + 1]
                    nextnonterminal = 1.0 - self.dones[i_nxt]
                    nextvalues = self.values[i_nxt]
                    delta = (
                        self.rewards[i_nxt]
                        + config.gamma * nextvalues * nextnonterminal
                        - self.values[i]
                    )
                    advantages[t] = lastgaelam = (
                        delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
                    )

            # Flatten the batch
            self.b_obs = b_obs = self.obs[b_idxs]
            b_actions = self.actions[b_idxs]
            b_logprobs = self.logprobs[b_idxs]
            b_dones = self.dones[b_idxs]
            b_values = self.values[b_idxs]
            b_advantages = advantages.reshape(
                config.batch_rows, num_minibatches, config.bptt_horizon
            ).transpose(0, 1)
            b_returns = b_advantages + b_values

            # Optimizing the policy and value network
            train_time = time.time()
            pg_losses, entropy_losses, v_losses, clipfracs, old_kls, kls = [], [], [], [], [], []
            for epoch in range(config.update_epochs):
                lstm_state = None
                for mb in range(num_minibatches):
                    mb_obs = b_obs[mb].to(self.device, non_blocking=True)
                    mb_actions = b_actions[mb].contiguous()
                    mb_values = b_values[mb].reshape(-1)
                    mb_advantages = b_advantages[mb].reshape(-1)
                    mb_returns = b_returns[mb].reshape(-1)

                    if hasattr(self.agent, "lstm"):
                        (
                            _,
                            newlogprob,
                            entropy,
                            newvalue,
                            lstm_state,
                        ) = self.agent.get_action_and_value(
                            mb_obs, state=lstm_state, action=mb_actions
                        )
                        lstm_state = (lstm_state[0].detach(), lstm_state[1].detach())
                    else:
                        _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                            mb_obs.reshape(-1, *self.pool.single_observation_space.shape),
                            action=mb_actions,
                        )

                    logratio = newlogprob - b_logprobs[mb].reshape(-1)
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        old_kls.append(old_approx_kl.item())
                        approx_kl = ((ratio - 1) - logratio).mean()
                        kls.append(approx_kl.item())
                        clipfracs += [
                            ((ratio - 1.0).abs() > config.clip_coef).float().mean().item()
                        ]

                    mb_advantages = mb_advantages.reshape(-1)
                    if config.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - config.clip_coef, 1 + config.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    pg_losses.append(pg_loss.item())

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if config.clip_vloss:
                        v_loss_unclipped = (newvalue - mb_returns) ** 2
                        v_clipped = mb_values + torch.clamp(
                            newvalue - mb_values,
                            -config.vf_clip_coef,
                            config.vf_clip_coef,
                        )
                        v_loss_clipped = (v_clipped - mb_returns) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()
                    v_losses.append(v_loss.item())

                    entropy_loss = entropy.mean()
                    entropy_losses.append(entropy_loss.item())

                    loss = pg_loss - config.ent_coef * entropy_loss + v_loss * config.vf_coef
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), config.max_grad_norm)
                    self.optimizer.step()

                if config.target_kl is not None:
                    if approx_kl > config.target_kl:
                        break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        losses = self.losses
        losses.policy_loss = np.mean(pg_losses)
        losses.value_loss = np.mean(v_losses)
        losses.entropy = np.mean(entropy_losses)
        losses.old_approx_kl = np.mean(old_kls)
        losses.approx_kl = np.mean(kls)
        losses.clipfrac = np.mean(clipfracs)
        losses.explained_variance = explained_var

        perf = self.performance
        perf.total_uptime = int(time.time() - self.start_time)
        perf.total_updates = self.update + 1
        perf.train_time = time.time() - train_time
        perf.train_sps = int(config.batch_size / perf.train_time)
        perf.train_memory = train_profiler.end_mem
        perf.train_pytorch_memory = train_profiler.end_torch_mem
        perf.epoch_time = perf.eval_time + perf.train_time
        perf.epoch_sps = int(config.batch_size / perf.epoch_time)

        if config.verbose:
            print_dashboard(self.stats, self.init_performance, self.performance)

        self.update += 1
        if self.update % config.checkpoint_interval == 0 or self.done_training():
            self.save_checkpoint()

    def close(self):
        self.pool.close()

        if self.wandb is not None:
            artifact_name = f"{self.exp_name}_model"
            artifact = self.wandb.Artifact(artifact_name, type="model")
            model_path = self.save_checkpoint()
            artifact.add_file(model_path)
            self.wandb.run.log_artifact(artifact)
            self.wandb.finish()

    def save_checkpoint(self):
        path = os.path.join(self.config.data_dir, self.exp_name)
        if not os.path.exists(path):
            os.makedirs(path)

        model_name = f"model_{self.update:06d}.pt"
        model_path = os.path.join(path, model_name)

        # Already saved
        if os.path.exists(model_path):
            return model_path

        torch.save(self.agent, model_path)

        state = {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "agent_step": self.global_step,
            "update": self.update,
            "model_name": model_name,
        }

        if self.wandb:
            state["exp_name"] = self.exp_name

        state_path = os.path.join(path, "trainer_state.pt")
        torch.save(state, state_path + ".tmp")
        os.rename(state_path + ".tmp", state_path)

        return model_path

    def done_training(self):
        return self.update >= self.total_updates

    def __enter__(self):
        return self

    def __exit__(self, *args):
        print("Done training. Saving data...")
        self.close()
        print("Run complete")
