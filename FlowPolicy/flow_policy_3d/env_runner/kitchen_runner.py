import time

import numpy as np
import torch
import tqdm
from termcolor import cprint

import flow_policy_3d.common.logger_util as logger_util
from flow_policy_3d.common.pytorch_util import dict_apply
from flow_policy_3d.env.kitchen.kitchen_wrapper import KitchenEnvWrapper
from flow_policy_3d.env_runner.base_runner import BaseRunner
from flow_policy_3d.gym_util.gymnasium_multistep_wrapper import (
    GymnasiumMultiStepWrapper,
)
from flow_policy_3d.policy.base_policy import BasePolicy


class KitchenRunner(BaseRunner):
    def __init__(
        self,
        output_dir,
        eval_episodes=50,
        max_steps=280,
        n_obs_steps=2,
        n_action_steps=4,
        tqdm_interval_sec=5.0,
        tasks_to_complete=None,
        device="cuda",
        n_tasks=4,
    ):
        super().__init__(output_dir)
        if tasks_to_complete is None:
            tasks_to_complete = [
                "microwave",
                "kettle",
                "light switch",
                "slide cabinet",
            ]
        self.tasks_to_complete = tasks_to_complete
        self.n_tasks = n_tasks
        self.eval_episodes = eval_episodes
        self.max_steps = max_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.device = device

        def env_fn():
            base = KitchenEnvWrapper(
                tasks_to_complete=tasks_to_complete,
                max_episode_steps=max_steps,
            )
            return GymnasiumMultiStepWrapper(
                base,
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method="sum",
            )

        self.env = env_fn()
        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)

    def _warmup_policy(self, policy: BasePolicy, device):
        """Dummy forward pass to reduce CUDA init bias in latency (§5.2 doc)."""
        obs, _ = self.env.reset()
        np_obs = dict_apply(obs, lambda x: torch.from_numpy(x).to(device=device))
        obs_in = {"agent_pos": np_obs["agent_pos"].unsqueeze(0)}
        with torch.no_grad():
            policy.predict_action(obs_in)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def run(self, policy: BasePolicy, eval_episodes=None, warmup_gpu=True):
        if eval_episodes is not None:
            n_episodes = eval_episodes
        else:
            n_episodes = self.eval_episodes

        device = policy.device
        env = self.env

        if warmup_gpu:
            self._warmup_policy(policy, device)

        all_success = []
        all_success_k = {k: [] for k in range(1, self.n_tasks + 1)}
        all_latency = []

        for episode_idx in tqdm.tqdm(
            range(n_episodes),
            desc="Eval Franka Kitchen",
            leave=False,
            mininterval=self.tqdm_interval_sec,
        ):
            obs, info = env.reset()
            policy.reset()

            terminated = False
            truncated = False
            step_count = 0
            total_time = 0.0
            max_tasks_completed = 0

            while not (terminated or truncated):
                np_obs_dict = dict(obs)
                obs_dict = dict_apply(
                    np_obs_dict,
                    lambda x: torch.from_numpy(x).to(device=device),
                )
                obs_dict_input = {
                    "agent_pos": obs_dict["agent_pos"].unsqueeze(0),
                }

                with torch.no_grad():
                    t0 = time.time()
                    action_dict = policy.predict_action(obs_dict_input)
                    total_time += time.time() - t0

                np_action_dict = dict_apply(
                    action_dict,
                    lambda x: x.detach().to("cpu").numpy(),
                )
                action = np_action_dict["action"].squeeze(0)
                obs, reward, terminated, truncated, info = env.step(action)
                step_count += 1
                if isinstance(info, dict):
                    if "num_tasks_completed" in info:
                        val = info["num_tasks_completed"]
                        if hasattr(val, "__len__") and not isinstance(val, str):
                            n_step = int(np.max(val))
                        else:
                            n_step = int(val)
                        max_tasks_completed = max(max_tasks_completed, n_step)
                    elif "episode_task_completions" in info:
                        comp = info["episode_task_completions"]
                        if hasattr(comp, "__len__") and len(comp) > 0:
                            last = comp[-1] if hasattr(comp[-1], "__len__") else comp
                            max_tasks_completed = max(
                                max_tasks_completed, len(last)
                            )

            n_done = max_tasks_completed

            success = 1.0 if n_done >= self.n_tasks else 0.0
            all_success.append(success)
            for k in range(1, self.n_tasks + 1):
                all_success_k[k].append(1.0 if n_done >= k else 0.0)

            if step_count > 0:
                all_latency.append(total_time / step_count)

        log_data = dict()
        mean_sr = float(np.mean(all_success)) * 100.0
        log_data["success_rate"] = mean_sr
        log_data["mean_success_rates"] = mean_sr
        log_data["test_mean_score"] = mean_sr

        for k in range(1, self.n_tasks + 1):
            log_data[f"success_rate_k{k}"] = float(np.mean(all_success_k[k])) * 100.0

        mean_lat = float(np.mean(all_latency)) if all_latency else 0.0
        log_data["mean_inference_latency"] = mean_lat
        log_data["mean_time"] = mean_lat
        if mean_lat > 0:
            log_data["trade_off"] = mean_sr / mean_lat
        else:
            log_data["trade_off"] = 0.0

        cprint(f"success_rate: {mean_sr:.2f}%", "green")
        cprint(f"mean_inference_latency: {mean_lat:.6f}s", "green")

        self.logger_util_test.record(mean_sr)
        self.logger_util_test10.record(mean_sr)
        log_data["SR_test_L3"] = self.logger_util_test.average_of_largest_K()
        log_data["SR_test_L5"] = self.logger_util_test10.average_of_largest_K()

        return log_data
