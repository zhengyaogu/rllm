import asyncio
import json
import math
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import reduce
from pprint import pprint
from queue import Queue
from threading import Thread
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, List
import warnings

import numpy as np
import torch
from omegaconf import OmegaConf

from rllm.engine.agent_execution_engine import AsyncAgentExecutionEngine
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    RayWorkerGroup,
    ResourcePoolManager,
    Role,
    WorkerType,
    AdvantageEstimator,
    _timer,
    compute_advantage,
    compute_data_metrics,
    compute_response_mask,
    compute_timing_metrics,
    reduce_metrics,
)


@dataclass
class DISCDataPoint:
    # we opt to not include the prompt in the data point, since it is already prepared in batch
    steps: List[str]
    z_score: float
    alpha: float
    num_sampled: int


def split_str(s, fraction):
    '''
    Splits a string into two parts around the specified fraction of the string.
    Returns None if the string cannot be split at the specified fraction.
    '''
    # Find the index at which to split the string
    split_index = int(len(s) * fraction)
    return s[:split_index], s[split_index:]


def compute_z_score(
    batch,
    epsilon=1e-6
):
    index = batch.non_tensor_batch["uid"]
    if "rm_scores" in batch.batch:
        token_level_scores = batch.batch["token_level_scores"].sum(dim=-1)
        rm_scores = torch.sigmoid(batch.batch["rm_scores"].sum(dim=-1)).to(token_level_scores.dtype)
        correct_mask = token_level_scores == 1.
        rm_scores[correct_mask] = 1.
    elif "token_level_scores" in batch.batch:
        rm_scores = batch.batch["token_level_scores"].sum(dim=-1)
    else:
        raise ValueError("No reward scores found in batch")
    
    scores = rm_scores

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    uid2batch_idx = defaultdict(list)

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            uid2batch_idx[index[i]].append(i) # write down index of data point for each uid
        
        max_len = 0
        for idx in id2score:
            if len(id2score[idx]) > max_len:
                max_len = len(id2score[idx])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                if max_len == 1:
                    id2mean[idx] = torch.tensor(0.0)
                    id2std[idx] = torch.tensor(1.0)
                else:
                    id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                    id2std[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        
        for idx in id2score:
            max_i = max(range(len(id2score[idx])), key=lambda i: id2score[idx][i])
            id2score[idx] = (id2score[idx][max_i] - id2mean[idx]) / (id2std[idx] + epsilon)
            uid2batch_idx[idx] = uid2batch_idx[idx][max_i]


    return id2score, uid2batch_idx, id2mean, id2std


class AgentPPOTrainer(RayPPOTrainer):
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        reward_fn=None,
        val_reward_fn=None,
        env_class=None,
        agent_class=None,
        env_args=None,
        agent_args=None,
    ):
        super().__init__(config=config, tokenizer=tokenizer, role_worker_mapping=role_worker_mapping, resource_pool_manager=resource_pool_manager, ray_worker_group_cls=ray_worker_group_cls, reward_fn=reward_fn, val_reward_fn=val_reward_fn)
        self.env_class = env_class
        self.agent_class = agent_class
        self.env_args = env_args or {}
        self.agent_args = agent_args or {}

        if self.config.agent.use_stepwise_advantage:
            print("Using step-level advantage, max_prompt_length and max_response_length will be applied step-wise")
        else:
            print("Using trajectory-level advantage, max_prompt_length and max_response_length will be applied episode-wise")

    def init_workers(self):
        super().init_workers()

        # Initialize additional agent class
        # Number of agents is set to be 0 initially
        if self.hybrid_engine:
            agent_rollout_wg = self.actor_rollout_wg
        else:
            agent_rollout_wg = self.rollout_wg

        if self.config.actor_rollout_ref.rollout.mode == "async":
            rollout_engine = self.async_rollout_manager
        else:
            rollout_engine = agent_rollout_wg

        self.agent_execution_engine = AsyncAgentExecutionEngine(
            rollout_engine=rollout_engine,
            config=self.config,
            engine_name="verl",
            tokenizer=self.tokenizer,
            model_path=self.config.actor_rollout_ref.model.path,
            max_steps=self.config.agent.max_steps,
            max_response_length=self.config.data.max_response_length,
            max_prompt_length=self.config.data.max_prompt_length,
            agent_class=self.agent_class,
            agent_args=self.agent_args,
            env_class=self.env_class,
            env_args=self.env_args,
            enforce_max_prompt_length=self.config.agent.use_stepwise_advantage,
            trajectory_timeout=self.config.agent.trajectory_timeout,
            overlong_filter=self.config.agent.overlong_filter,
            **self.config.agent.get("engine_args", {}),
        )

    def init_envs_and_agents(self, batch):
        """
        Initialize environment depending on env_class with the necessary extra_info, also set uid of the batch.
        """
        env_args = batch.non_tensor_batch["extra_info"].tolist()

        full_agent_args = dict(self.config.agent.get("agent_args", {})) | self.agent_args
        base_env_args = dict(self.config.env.get("env_args", {})) | self.env_args

        def _create_env(i):
            if isinstance(env_args[i], str):
                env_args[i] = json.loads(env_args[i])
            return i, self.env_class.from_dict({**env_args[i], **base_env_args})

        def _create_agent(i):
            return i, self.agent_class(**full_agent_args)

        # Create environments in parallel while preserving order
        envs = [None] * len(env_args)
        with ThreadPoolExecutor(max_workers=64) as executor:
            env_futures = [executor.submit(_create_env, i) for i in range(len(env_args))]
            for future in as_completed(env_futures):
                idx, env = future.result()
                envs[idx] = env

        # Create agents in parallel while preserving order
        agents = [None] * len(envs)
        with ThreadPoolExecutor(max_workers=64) as executor:
            agent_futures = [executor.submit(_create_agent, i) for i in range(len(envs))]
            for future in as_completed(agent_futures):
                idx, agent = future.result()
                agents[idx] = agent
        self.agent_execution_engine.update_envs_and_agents(envs, agents)
        return envs

    def fit_agent(self):
        """
        The training loop of PPO. Adapted to train the underlying model of agent.
        """
        print("Start Training...")
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        import time

        start_time = time.time()
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate_agent()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return
        print(f"Time taken to validate agent: {time.time() - start_time}")
        # we start from step 1
        self.global_steps += 1

        for epoch in range(self.config.trainer.total_epochs):
            pprint(f"epoch {epoch}, step {self.global_steps} started")
            for batch_dict in self.train_dataloader:
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)

                sampling_method = self.config.actor_rollout_ref.rollout.get("sampling_method", "greedy")
                if sampling_method == "greedy":
                    batch = batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n,
                        interleave=True,
                    ) # only repeat batch for greedy sampling
                elif sampling_method == "disc":
                    # prepare variables for disc sampling
                    sample_budget = self.config.actor_rollout_ref.rollout.n
                    iteration_threshold = self.config.actor_rollout_ref.rollout.reward_threshold
                    alpha = self.config.actor_rollout_ref.rollout.alpha0

                    unique_uids = np.unique(batch.non_tensor_batch["uid"])

                    id2data_points = {}
                    for uid in unique_uids:
                        id2data_points[uid] = DISCDataPoint(
                            steps=[],
                            z_score=float("inf"),
                            alpha=alpha,
                            num_sampled=0
                        )
                    remaining_uids = unique_uids.tolist()

                    batch_total = None
                else:
                    raise NotImplementedError(f"Sampling method {sampling_method} not supported")

                metrics = {}
                timing_raw = {}

                batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"])
                batch.meta_info = {
                    "agent_rollout": True,  # no need to generate multiple ones since environment is repeated already
                }

                with _timer("step", timing_raw):
                    sampling_method = self.config.actor_rollout_ref.rollout.get("sampling_method", "greedy")
                    if sampling_method == "greedy":
                        self.init_envs_and_agents(batch)

                        if self.config.agent.use_stepwise_advantage:
                            final_gen_batch_output = self.generate_agent_steps(timing_raw=timing_raw, meta_info=batch.meta_info, uids=batch.non_tensor_batch["uid"])
                            repeat_counts = final_gen_batch_output.meta_info["repeat_counts"]
                            # need to repeat to make shape match
                            batch = batch.repeat_by_counts(repeat_counts, interleave=True)
                            final_gen_batch_output.meta_info.pop("repeat_counts", None)  # no longer needed after this
                            # batch needs to be padded to divisor of world size, we will pad with everything masked out
                            batch = batch.union(final_gen_batch_output)
                            batch = self._pad_dataproto_to_world_size(batch=batch)
                        else:
                            final_gen_batch_output, generate_metrics = self.generate_agent_trajectory(timing_raw=timing_raw, meta_info=batch.meta_info)
                            batch = batch.union(final_gen_batch_output)
                            metrics.update(generate_metrics)

                        # compute values
                        if self.use_critic:
                            with _timer("values", timing_raw):
                                values = self.critic_wg.compute_values(batch)
                                batch = batch.union(values)
                        with _timer("adv", timing_raw):
                            if self.use_rm:
                                reward_tensor = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(reward_tensor)

                            # reward tensor for env-based trajectory data can be obtained by processing the trajectories
                            if "token_level_scores" not in batch.batch:
                                reward_tensor = self.reward_fn(batch)
                                batch.batch["token_level_scores"] = reward_tensor
                            else:
                                reward_tensor = batch.batch["token_level_scores"]  # filled in by environment collected trajectory transformation
                        
                    elif sampling_method == "disc":
                        round_num = 0
                        
                        # wake up rollout engine if manual sleep wakeup is enabled
                        if self.config.actor_rollout_ref.manual_sleep_wakeup:
                            max_concurrency = self.agent_execution_engine.n_parallel_agents
                            self.agent_execution_engine.executor = ThreadPoolExecutor(max_workers=max_concurrency)

                            if self.agent_execution_engine.engine_name == "verl":
                                self.agent_execution_engine.rollout_engine.wake_up()

                        total_traj_remaining = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                        max_num_rounds = self.config.actor_rollout_ref.rollout.disc_max_num_rounds
                        rollout_size_per_round = self.config.actor_rollout_ref.rollout.disc_rollout_size_per_round
                        finished_uids = set()
                        while round_num < max_num_rounds and total_traj_remaining > 0:
                            print(f"NUM UIDS STILL TO SAMPLE THIS BATCH: {len(unique_uids) - len(finished_uids)}")
                            remaining_uids = np.array([uid for uid in unique_uids if uid not in finished_uids])
                            if remaining_uids.shape[0] == 0: # if all uids are finished, sample all uids to make the batch size
                                remaining_uids = unique_uids.copy()
                            actual_rollout_size_curr_round, rollout_remainder = total_traj_remaining // remaining_uids.shape[0], total_traj_remaining % remaining_uids.shape[0]

                            # if not the last round sample a pre-designated number of traj for each uid (if available)
                            # sample evenly (with remainder) if not enough traj for each uid
                            if round_num < max_num_rounds - 1 and actual_rollout_size_curr_round >= rollout_size_per_round: 
                                select_mask = np.isin(batch.non_tensor_batch["uid"], remaining_uids)
                                batch_curr_round = batch.select_idxs(select_mask)
                                batch_curr_round = batch_curr_round.repeat(
                                    repeat_times=rollout_size_per_round,
                                    interleave=True,
                                )
                            else: # sample evenly what's remaining, sample remainders randomly
                                remainder_uids = np.random.choice(remaining_uids, size=rollout_remainder, replace=False)
                                main_uids = np.array([uid for uid in remaining_uids if uid not in remainder_uids])
                                remainder_mask = np.isin(batch.non_tensor_batch["uid"], remainder_uids)
                                main_mask = np.isin(batch.non_tensor_batch["uid"], main_uids)
                                main_batch = batch.select_idxs(main_mask)
                                remainder_batch = batch.select_idxs(remainder_mask)
                                remainder_batch = remainder_batch.repeat(
                                    repeat_times=actual_rollout_size_curr_round + 1,
                                    interleave=True,
                                )
                                main_batch = main_batch.repeat(
                                    repeat_times=actual_rollout_size_curr_round,
                                    interleave=True,
                                )
                                batch_curr_round = DataProto.concat([main_batch, remainder_batch])
                            
                            # add partial solution if available
                            for i, uid in enumerate(batch_curr_round.non_tensor_batch["uid"]):
                                if len(id2data_points[uid].steps) > 0:
                                    split_last_step, _ = split_str(id2data_points[uid].steps[-1], id2data_points[uid].alpha)
                                    partial_solution = ''.join(id2data_points[uid].steps[:-1]) + split_last_step
                                    batch_curr_round.non_tensor_batch["extra_info"][i]["partial_solution"] = partial_solution
                                else:
                                    batch_curr_round.non_tensor_batch["extra_info"][i]["partial_solution"] = None
                            
                            self.init_envs_and_agents(batch_curr_round)
                            
                            # generate trajectory
                            print("GENERATING TRAJECTORIES...")
                            assert self.config.agent.use_stepwise_advantage == False, "stepwise advantage is not supported for disc sampling"
                            final_gen_batch_output, generate_metrics = self.generate_agent_trajectory(timing_raw=timing_raw, meta_info=batch_curr_round.meta_info)
                            batch_curr_round = batch_curr_round.union(final_gen_batch_output)
                            metrics.update(generate_metrics)

                            batch_curr_round, pad_size = pad_dataproto_to_divisor(batch_curr_round, self.actor_rollout_wg.world_size) # pad to world size for rm

                            with _timer("adv", timing_raw):
                                # compute scores using reward model and/or reward function
                                if self.use_rm:
                                    reward_tensor = self.rm_wg.compute_rm_score(batch_curr_round)
                                    batch_curr_round = batch_curr_round.union(reward_tensor)
                                
                                batch_curr_round = unpad_dataproto(batch_curr_round, pad_size)

                                # reward tensor for env-based trajectory data can be obtained by processing the trajectories
                                if "token_level_scores" not in batch_curr_round.batch:
                                    reward_tensor = self.reward_fn(batch_curr_round)
                                    batch_curr_round.batch["token_level_scores"] = reward_tensor
                                else:
                                    reward_tensor = batch_curr_round.batch["token_level_scores"]  # filled in by environment collected trajectory transformation
                                
                                token_level_scores = batch_curr_round.batch["token_level_scores"].sum(dim=-1)

                            for i, uid in enumerate(batch_curr_round.non_tensor_batch["uid"]):
                                if token_level_scores[i] >= 1:
                                    finished_uids.add(uid)

                            for uid in batch_curr_round.non_tensor_batch["uid"]:
                                id2data_points[uid].num_sampled += 1
                            
                            id2score, uid2best_batch_idx, id2mean, id2std = compute_z_score(batch_curr_round, epsilon=1e-6)
                            for uid in id2score:
                                print(f"ROUND: {round_num}, UID: {uid}, SCORE: {id2score[uid]}, MEAN: {id2mean[uid]}, STD: {id2std[uid]}")
                            
                            unique_uids_curr_round = np.unique(batch_curr_round.non_tensor_batch["uid"])
                            for uid in unique_uids_curr_round:
                                new_z_score = id2score[uid]
                                # if new z score is better, or if the last step is too short, update the data point
                                if new_z_score < id2data_points[uid].z_score or (
                                    len(id2data_points[uid].steps) > 0 and len(id2data_points[uid].steps[-1]) <= 1
                                ):
                                    id2data_points[uid].z_score = new_z_score
                                    if len(id2data_points[uid].steps) > 0:
                                        second_last_step, _ = split_str(id2data_points[uid].steps[-1], id2data_points[uid].alpha)
                                        id2data_points[uid].steps[-1] = second_last_step
                                        id2data_points[uid].steps.append(
                                            batch_curr_round.non_tensor_batch["response_strs"][uid2best_batch_idx[uid]]
                                        )
                                    else:
                                        id2data_points[uid].steps.append(
                                            batch_curr_round.non_tensor_batch["response_strs"][uid2best_batch_idx[uid]]
                                        )
                                    id2data_points[uid].alpha = self.config.actor_rollout_ref.rollout.alpha0
                                else:
                                    id2data_points[uid].alpha = id2data_points[uid].alpha * self.config.actor_rollout_ref.rollout.alpha0
                            
                            if batch_total is None:
                                batch_total = batch_curr_round
                            else:
                                batch_total = DataProto.concat([batch_total, batch_curr_round])
                            for uid in unique_uids:
                                print(f"UID: {uid}, NUM SAMPLED: {id2data_points[uid].num_sampled}")
                            
                            round_num += 1
                            total_traj_remaining -= batch_curr_round.non_tensor_batch["uid"].shape[0]
                        
                        expected_sample_total = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                        if batch_total.non_tensor_batch["uid"].shape[0] != expected_sample_total:
                            warnings.warn(f"Expected {expected_sample_total} samples, got {batch_total.non_tensor_batch['uid'].shape[0]}")
                    
                        if self.config.actor_rollout_ref.manual_sleep_wakeup:
                            if self.agent_execution_engine.engine_name == "verl":
                                self.agent_execution_engine.rollout_engine.sleep()

                            self.agent_execution_engine.executor.shutdown(wait=False, cancel_futures=True)
                        
                        batch = batch_total

                    with _timer("adv", timing_raw):
                        # Rejection sampling based on rewards
                        # Group rewards by uid
                        reward_tensor = batch.batch["token_level_scores"]
                        uids = batch.non_tensor_batch["uid"]
                        unique_uids = np.unique(uids)
                        valid_mask = torch.ones(len(uids), dtype=torch.bool)
                        solve_none = 0
                        solve_all = 0
                        for uid in unique_uids:
                            uid_mask = uids == uid
                            uid_rewards = reward_tensor[uid_mask].sum(-1)  # Sum rewards for each sequence

                            # Check if all rewards are <= 0 or all are 1 >= for this uid
                            if (uid_rewards <= 0).all():
                                valid_mask[uid_mask] = False
                                solve_none += 1
                            elif (uid_rewards >= 1).all():
                                valid_mask[uid_mask] = False
                                solve_all += 1

                        # Log to metrics
                        metrics["batch/solve_none"] = solve_none
                        metrics["batch/solve_all"] = solve_all
                        metrics["batch/solve_partial"] = len(unique_uids) - solve_none - solve_all

                        if self.config.trainer.rejection_sample:
                            # log the actual complete training rewards before rejection sampling
                            token_level_rewards = None  # for metrics calculation
                            if self.config.agent.use_stepwise_advantage:
                                is_pad_step = batch.non_tensor_batch["is_pad_step"]
                                non_pad_step_indices = np.where(is_pad_step == False)[0]
                                non_pad_steps = batch.select_idxs(non_pad_step_indices)
                                is_last_step = non_pad_steps.non_tensor_batch["is_last_step"]
                                valid_last_step_indices = np.where(is_last_step == True)[0]
                                last_step_batch = batch.select_idxs(valid_last_step_indices)
                                token_level_rewards = last_step_batch.batch["token_level_scores"]
                            else:
                                token_level_rewards = batch.batch["token_level_scores"]
                            full_sequence_score = token_level_rewards.sum(-1)
                            metrics["critic/full-score/mean"] = torch.mean(full_sequence_score).detach().item()
                            metrics["critic/full-score/max"] = torch.max(full_sequence_score).detach().item()
                            metrics["critic/full-score/min"] = torch.min(full_sequence_score).detach().item()

                            # If no valid samples remain, skip this batch and get a new one
                            if not valid_mask.any():
                                continue

                            # Filter batch to keep only valid samples
                            batch = batch[valid_mask]

                            if self.config.agent.use_stepwise_advantage and self.config.agent.stepwise_advantage_mode == "broadcast":
                                # batch now only contains steps with valid uids
                                # filter out padding steps
                                is_pad_step = batch.non_tensor_batch["is_pad_step"]
                                non_pad_step_indices = np.where(is_pad_step == False)[0]
                                batch = batch.select_idxs(non_pad_step_indices)  # This batch only has non_pad steps

                                # need to make sure both number of last steps (number of uids) and number of total steps in the batch (batch size after processing) are all multiples of world size
                                # separate out last step and intermediate steps
                                is_last_step = batch.non_tensor_batch["is_last_step"]
                                valid_last_step_indices = np.where(is_last_step == True)[0]
                                not_last_step_indices = np.where(is_last_step == False)[0]
                                last_step_batch = batch.select_idxs(valid_last_step_indices)  # This batch only has valid last steps
                                non_last_step_batch = batch.select_idxs(not_last_step_indices)

                                # filter last_step_batch to make sure its multiple of world size
                                num_trainer_replicas = self.actor_rollout_wg.world_size
                                max_batch_size = (
                                    last_step_batch.batch["input_ids"].shape[0]  # 1 per trajectory
                                    // num_trainer_replicas
                                ) * num_trainer_replicas
                                if not max_batch_size:
                                    # give up, you got everything either all wrong or right.
                                    continue

                                size_mask = torch.zeros(last_step_batch.batch["input_ids"].shape[0], dtype=torch.bool)
                                size_mask[:max_batch_size] = True
                                last_step_batch = last_step_batch[size_mask]  # filtered last steps

                                # now we go through all the non_last_step_batch and keep everything that has same idxs that exists in the filtered last steps
                                valid_last_step_idxs = last_step_batch.non_tensor_batch["idxs"]
                                non_last_step_idxs = non_last_step_batch.non_tensor_batch["idxs"]
                                non_last_step_mask = np.isin(non_last_step_idxs, valid_last_step_idxs)
                                non_last_step_batch = non_last_step_batch[non_last_step_mask]

                                # concatenate then pad
                                batch = DataProto.concat([last_step_batch, non_last_step_batch])
                                batch = self._pad_dataproto_to_world_size(batch)
                            else:
                                # Round down to the nearest multiple of world size
                                num_trainer_replicas = self.actor_rollout_wg.world_size
                                max_batch_size = (batch.batch["input_ids"].shape[0] // num_trainer_replicas) * num_trainer_replicas
                                if not max_batch_size:
                                    # give up, you got everything either all wrong or right.
                                    continue

                                size_mask = torch.zeros(batch.batch["input_ids"].shape[0], dtype=torch.bool)
                                size_mask[:max_batch_size] = True
                                batch = batch[size_mask]

                        # recompute old_log_probs
                        with _timer("old_log_prob", timing_raw):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            batch = batch.union(old_log_prob)

                        if self.use_reference_policy:
                            # compute reference log_prob
                            with _timer("ref", timing_raw):
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)

                        # compute rewards with KL penalty if needed

                        # Note: This kl penalty applied directly over the rewards is disabled for GRPO. The kl penalty is applied at dp_actor.py
                        # where it is subtracted directly from the policy loss

                        # if not self.config.actor_rollout_ref.actor.use_kl_loss:
                        #     batch, kl_metrics = apply_kl_penalty(batch,
                        #                                        kl_ctrl=self.kl_ctrl,
                        #                                        kl_penalty=self.config.algorithm.kl_penalty)
                        #     metrics.update(kl_metrics)
                        # else:
                        #     batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
                        if self.config.reward_model.enable:
                            token_level_scores = batch.batch["token_level_scores"]
                            rm_scores = batch.batch["rm_scores"].clone().to(dtype=token_level_scores.dtype)
                            #print(torch.count_nonzero(rm_scores, dim=-1))
                            # assert (torch.count_nonzero(rm_scores, dim=-1) <= 1).all(), \
                            #     "Each sequence should have at most one non-zero reward score"
                            # assert token_level_scores.shape == rm_scores.shape, \
                            #     f"token_level_scores and rm_scores should have the same shape, got {token_level_scores.shape} and {rm_scores.shape}"
                            # assert (torch.count_nonzero(token_level_scores, dim=-1) <= 1).all(), \
                            #     "Each sequence should have at most one non-zero reward score"
                            correct_mask = (token_level_scores.sum(dim=-1) == 1)
                            rm_scores[correct_mask] = token_level_scores[correct_mask]
                            #print(torch.count_nonzero(rm_scores, dim=-1).max())
                            # assert (torch.count_nonzero(rm_scores, dim=-1) <= 1).all(), \
                            #     "Each sequence should have at most one non-zero reward score"
                            batch.batch["token_level_rewards"] = rm_scores.to(dtype=token_level_scores.dtype)
                        
                        if self.config.reward_model.overlong_buffer.enable:
                            token_level_scores = batch.batch["token_level_scores"]
                            correct_idx = token_level_scores.sum(dim=-1) == 1
                            correct_last_valid_idx = torch.nonzero(token_level_scores, as_tuple=False)[:, 1]
                            prompt_length = batch.batch["prompts"].shape[1]
                            attention_mask = batch.batch["attention_mask"]
                            last_valid_idx = attention_mask[:, prompt_length:].sum(dim=-1) - 1

                            #print("last_valid_idx", last_valid_idx[correct_idx])
                            #print("correct_last_valid_idx", correct_last_valid_idx)
                            assert torch.equal(last_valid_idx[correct_idx], correct_last_valid_idx), "last_valid_idx and correct_last_valid_idx should be the same"

                            overlong_penalty = self.compute_overlong_penalty(batch)
                            #print("overlong_penalty.shape", overlong_penalty.shape)
                            token_level_rewards = batch.batch["token_level_rewards"]
                            token_level_rewards[torch.arange(last_valid_idx.shape[0]), last_valid_idx] += overlong_penalty.to(dtype=token_level_scores.dtype)
                            batch.batch["token_level_rewards"] = token_level_rewards

                        if self.config.agent.use_stepwise_advantage:
                            if self.config.agent.stepwise_advantage_mode == "mc_return":
                                batch.batch["token_level_rewards"] = batch.batch["mc_returns"]
                                batch.non_tensor_batch["uid"] = batch.non_tensor_batch["step_ids"]

                                is_pad_step = batch.non_tensor_batch["is_pad_step"]
                                non_pad_step_indices = np.where(is_pad_step == False)[0]
                                batch = batch.select_idxs(non_pad_step_indices)  # This batch only has non_pad steps
                            elif self.config.agent.stepwise_advantage_mode == "broadcast":
                                # In case of step-wise advantage broadcast, we would split out the final steps, then merge again
                                is_last_step = batch.non_tensor_batch["is_last_step"]
                                last_step_indices = np.where(is_last_step == True)[0]
                                other_step_indices = np.where(is_last_step == False)[0]
                                other_step_batch = batch.select_idxs(other_step_indices)
                                batch = batch.select_idxs(last_step_indices)  # This batch only has last steps
                            else:
                                raise ValueError(f"Stepwise advantage mode {self.config.agent.stepwise_advantage_mode} not supported")

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            mask_truncated_samples=self.config.algorithm.mask_truncated_samples,
                            clip_advantages=self.config.algorithm.clip_advantages,
                        )

                        if self.config.agent.use_stepwise_advantage and self.config.agent.stepwise_advantage_mode == "broadcast":
                            # remove the padded last steps
                            # Merging the separated out steps using the advantage from last steps
                            self._stepwise_advantage_broadcast(batch, other_step_batch=other_step_batch)
                            # batch = batch.merge(other_step_batch)
                            batch = DataProto.concat([batch, other_step_batch])

                    batch = self._pad_dataproto_to_world_size(batch=batch)
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate_agent()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:
                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate_agent()
                        pprint(f"Final validation metrics: {val_metrics}")
                        logger.log(data=val_metrics, step=self.global_steps)
                    return

    def _validate_agent(self):
        if self.config.actor_rollout_ref.manual_sleep_wakeup:
            max_concurrency = self.agent_execution_engine.n_parallel_agents
            self.agent_execution_engine.executor = ThreadPoolExecutor(max_workers=max_concurrency)

            if self.agent_execution_engine.engine_name == "verl":
                self.agent_execution_engine.rollout_engine.wake_up()
        rewards_lst = []
        data_source_lst = []
        uid_lst = []
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            test_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object)
            for i in range(len(test_batch.batch)):
                test_batch.non_tensor_batch["extra_info"][i]["partial_solution"] = None
            n_val_samples = self.config.actor_rollout_ref.rollout.val_kwargs.n
            test_batch = test_batch.repeat(repeat_times=n_val_samples, interleave=True)
            test_batch.pop(["input_ids", "attention_mask", "position_ids"])  # these are not needed for environment based interaction
            test_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": False,
                "validate": True,
                "agent_rollout": True,
            }
            self.init_envs_and_agents(test_batch)

            if self.config.agent.use_stepwise_advantage:
                test_output_gen_batch = self.generate_agent_steps(meta_info=test_batch.meta_info, uids=test_batch.non_tensor_batch["uid"])
                # for validation, we only need the last step
                is_last_step = test_output_gen_batch.non_tensor_batch["is_last_step"]
                last_step_indices = np.where(is_last_step == True)[0]
                test_output_gen_batch = test_output_gen_batch.select_idxs(last_step_indices)  # This batch only has last steps
            else:
                test_output_gen_batch, _ = self.generate_agent_trajectory(meta_info=test_batch.meta_info)

            test_batch = test_batch.union(test_output_gen_batch)

            reward_tensor = test_batch.batch["token_level_scores"]

            rewards_lst.append(reward_tensor.sum(-1).cpu())
            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))
            uid_lst.append(test_batch.non_tensor_batch["uid"])

        reward_tensor = torch.cat(rewards_lst, dim=0)  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        # evaluate test_score based on data source
        data_source_reward = {}

        # to group for pass@k
        uid_tensor = np.concatenate(uid_lst, axis=0)
        data_source_uid_pass_rates = {}  # data source to {uid: pass or not}

        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]

            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

            # pass@k
            if data_source not in data_source_uid_pass_rates:
                data_source_uid_pass_rates[data_source] = {}

            uid = uid_tensor[i]
            if uid not in data_source_uid_pass_rates[data_source]:
                data_source_uid_pass_rates[data_source][uid] = 0  # default to not pass
            # take highest score
            data_source_uid_pass_rates[data_source][uid] = max(data_source_uid_pass_rates[data_source][uid], reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            # clip rewards to be between 0 and 1
            rewards_array = np.array(rewards)
            rewards_array = np.clip(rewards_array, 0, 1)
            metric_dict[f"val/test_score/{data_source}"] = np.mean(rewards_array)

        for data_source, pass_rates in data_source_uid_pass_rates.items():
            pass_k_lst = []
            for uid, pass_score in pass_rates.items():
                pass_k_lst.append(pass_score >= 1)  # assuming 1 means passed
            metric_dict[f"val/test_score/pass@k/{data_source}"] = np.mean(pass_k_lst)
        
        if self.config.actor_rollout_ref.manual_sleep_wakeup:
            if self.agent_execution_engine.engine_name == "verl":
                self.agent_execution_engine.rollout_engine.sleep()

        return metric_dict

    def generate_agent_trajectory(self, timing_raw=None, meta_info=None):
        """
        Generates agent trajectories by interacting with the environment. Does not close or reset the environment afterwards

        Args:
            envs: The environments in which the agent interacts.
            agents: The agents to use for interation.
            timing_raw: Dictionary to store timing information for profiling.
            meta_info (optional): Metadata for veRL generation.

        Returns:
            DataProto: Representation of the agent's trajectories.
            Dict[str:float]: Metrics for the generation process.
        """
        if timing_raw is None:
            timing_raw = {}
        with _timer("collect_trajectory", timing_raw):
            trajectories = []
            if self.config.agent.async_engine:
                gen_seq_generator = self.generate_agent_trajectories_async(timing_raw=timing_raw, meta_info=meta_info, mode="Token")
                for _, trajectory in enumerate(gen_seq_generator):
                    trajectories.append(trajectory)
            else:
                # generate_trajectories returns list of trajectories.
                trajectories = self.agent_execution_engine.generate_trajectories(timing_raw=timing_raw, mode="Token", meta_info=meta_info)
        # Sort trajectories by their idx, to ensure they are in order.
        trajectories.sort(key=lambda x: x["idx"])

        with _timer("transform_trajectory", timing_raw):
            # Transform the raw trajectories into DataProto format.
            final_gen_batch_output, metrics = self._transform_agent_trajectories(trajectories)
        return final_gen_batch_output, metrics

    def generate_agent_steps(self, timing_raw=None, meta_info=None, uids=None):
        """
        Generates agent trajectories by interacting with the environment. Does not close or reset the environment afterwards.

        Returns:
            DataProto: Representation of the last step of agent's trajectories.
            Dict[str:List[DataProto]]: Index of the trajectory to the rest of the steps from the trajectory.
        """
        if timing_raw is None:
            timing_raw = {}
        if uids is None:
            uids = []
        with _timer("collect_trajectory", timing_raw):
            steps = []
            if self.config.agent.async_engine:
                gen_seq_generator = self.generate_agent_trajectories_async(timing_raw=timing_raw, meta_info=meta_info, mode="Step")
                for _, trajectory in enumerate(gen_seq_generator):
                    steps.append(trajectory)
            else:
                # generate_trajectories returns list of trajectories.
                steps = self.agent_execution_engine.generate_trajectories(timing_raw=timing_raw, mode="Step", meta_info=meta_info)
        # Sort trajectories by their idx, to ensure they are in order.
        steps.sort(key=lambda x: x["idx"])

        with _timer("transform_trajectory", timing_raw):
            # Transform the raw trajectories into DataProto format.
            final_gen_batch_output = self._transform_agent_steps(steps, uids=uids)
        return final_gen_batch_output

    def _transform_agent_trajectories(self, trajectories: list[dict]):
        """
        Helper function to transform a list of trajectories into tokenized DataProto format.

        Args:
            trajectories (list of dict): List of trajectories to process.

        Returns:
            DataProto: A structured dataset containing input tokens, masks, and rewards.
        """
        from verl.utils.torch_functional import pad_sequence_to_length

        all_initial_tokens_list = []
        all_response_tokens_list = []
        all_initial_str_list = []
        all_response_str_list = []
        all_masks_list = []
        traj_scores = []
        chat_completions = []
        all_response_token_lens = []
        all_prompt_token_lens = []
        traj_metrics = []
        metrics = {}

        for traj in trajectories:
            prompt_tokens = traj["prompt_tokens"]
            response_tokens = traj["response_tokens"]
            initial_str = self.tokenizer.decode(prompt_tokens, skip_special_tokens=True)
            response_str = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            # test if trajectory is empty
            assert prompt_tokens.numel() != 0 and response_tokens.numel() != 0, f"Both prompt {prompt_tokens.numel()} and response {response_tokens.numel()} of trajectory shouldn't be empty. Please check make sure environment is working and the config"
            all_initial_tokens_list.append(prompt_tokens)
            all_response_tokens_list.append(response_tokens)
            all_initial_str_list.append(initial_str)
            all_response_str_list.append(response_str)
            all_masks_list.append(traj["response_masks"])
            traj_scores.append(traj["trajectory_reward"])
            chat_completions.append(traj["chat_completions"])
            all_response_token_lens.append(traj["response_token_len"])
            all_prompt_token_lens.append(traj["prompt_token_len"])
            traj_metrics.append(traj["metrics"])

        # Flatten traj_metrics into a dict of lists
        traj_metrics = {k: [d[k] for d in traj_metrics] for k in traj_metrics[0]}
        # Aggregate metrics (mean, min, max)
        for k, v_list in traj_metrics.items():
            v_list = [v for v in v_list if v is not None and v >= 0]
            if not v_list:
                continue
            v_list = np.array(v_list)
            metrics.update(
                {
                    f"traj/{k}_mean": v_list.mean(),
                    f"traj/{k}_min": v_list.min(),
                    f"traj/{k}_max": v_list.max(),
                }
            )

        # Save chat completions to a file
        save_dir = os.path.join(self.config.trainer.default_local_dir, "chat_completions")
        os.makedirs(save_dir, exist_ok=True)
        # Save it into a jsonl files (self.global_steps)
        with open(os.path.join(save_dir, f"{self.global_steps}.jsonl"), "w") as f:
            for chat_completion in chat_completions:
                f.write(json.dumps(chat_completion) + "\n")

        # reverse the list and create tensors, pad, then flip to achieve left padding
        prompts_batch = torch.nn.utils.rnn.pad_sequence(
            [torch.flip(i, dims=[0]) for i in all_initial_tokens_list],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        ).flip(dims=[1])

        prompts_batch = pad_sequence_to_length(prompts_batch, self.config.data.max_prompt_length, self.tokenizer.pad_token_id, left_pad=True)

        response_batch = torch.nn.utils.rnn.pad_sequence(
            all_response_tokens_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )

        max_response_length = self.config.data.max_response_length
        response_batch = pad_sequence_to_length(response_batch, max_response_length, self.tokenizer.pad_token_id, left_pad=False)

        traj_mask = torch.nn.utils.rnn.pad_sequence(all_masks_list, batch_first=True, padding_value=0)
        traj_mask = pad_sequence_to_length(traj_mask, max_response_length, 0, left_pad=False)

        trajectory_batch = torch.concat([prompts_batch, response_batch], dim=1)

        attention_mask = torch.where(trajectory_batch != self.tokenizer.pad_token_id, 1, 0)

        # Compute position_ids
        position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

        # Place all rewards to last response token
        score_batch = torch.zeros_like(response_batch, dtype=torch.float32)

        prompt_length = prompts_batch.shape[1]
        valid_response_length_sequences = attention_mask[:, prompt_length:].sum(dim=-1)

        for i, traj_score in enumerate(traj_scores):
            last_valid_idx = valid_response_length_sequences[i] - 1
            if last_valid_idx >= 0 and last_valid_idx < score_batch.shape[1]:
                score_batch[i, last_valid_idx] = traj_score
        
        # pack response_str & initial_str as np.array
        prompt_str_batch = np.array(all_initial_str_list, dtype=object) # (batch_size,)
        response_str_batch = np.array(all_response_str_list, dtype=object) # (batch_size,)
        response_token_lens_batch = np.array(all_response_token_lens, dtype=np.int32) # (batch_size,)
        prompt_token_lens_batch = np.array(all_prompt_token_lens, dtype=np.int32) # (batch_size,)

        tensor_batch = {
            "input_ids": trajectory_batch,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": response_batch,
            "prompts": prompts_batch,
            "token_level_scores": score_batch,
            "traj_mask": traj_mask,
        }

        non_tensor_batch = {
            "response_strs": response_str_batch,
            "prompt_strs": prompt_str_batch,
            "response_token_lens": response_token_lens_batch,
            "prompt_token_lens": prompt_token_lens_batch,
        }

        self.visualize_trajectory(DataProto.from_dict(tensors=tensor_batch))

        return DataProto.from_dict(tensors=tensor_batch, non_tensors=non_tensor_batch), metrics

    def visualize_trajectory(self, tensor_batch, sample_idx=0, max_samples=1, mask_key="traj_mask"):
        """
        Visualize the trajectory from tensor_batch by detokenizing prompts and responses,
        and highlighting the masked parts with color.

        Args:
            tensor_batch: The tensor batch containing trajectory data
            sample_idx: Starting index of samples to visualize
            max_samples: Maximum number of samples to visualize
        """
        from rllm.misc import colorful_print

        # Get the relevant tensors
        prompts = tensor_batch.batch["prompts"]
        responses = tensor_batch.batch["responses"]
        traj_mask = tensor_batch.batch[mask_key]
        token_level_scores = tensor_batch.batch["token_level_scores"]

        batch_size = prompts.shape[0]
        end_idx = min(sample_idx + max_samples, batch_size)

        for i in range(sample_idx, end_idx):
            colorful_print(f"\n===== Sample {i} =====", fg="cyan", bold=True)

            # Detokenize prompt
            prompt_tokens = prompts[i]
            prompt_mask = prompt_tokens != self.tokenizer.pad_token_id
            valid_prompt_tokens = prompt_tokens[prompt_mask]
            prompt_text = self.tokenizer.decode(valid_prompt_tokens)

            colorful_print("Prompt:", fg="green", bold=True)
            colorful_print(f"{prompt_text}\n", fg="green")

            # Detokenize response with color highlighting for masked tokens
            response_tokens = responses[i]
            response_mask = traj_mask[i]

            # Get non-padding tokens
            valid_indices = response_tokens != self.tokenizer.pad_token_id
            valid_response_tokens = response_tokens[valid_indices]
            valid_response_mask = response_mask[valid_indices]

            # Then show token-by-token with masking
            colorful_print("Response with masking:", fg="yellow", bold=True)

            for j, (token, mask) in enumerate(zip(valid_response_tokens, valid_response_mask, strict=False)):
                token_text = self.tokenizer.decode(token)

                # Check if this token has a reward
                has_reward = token_level_scores[i, j] != 0

                # Apply different colors based on mask and rewards
                if mask == 0:
                    # Masked token (not used in training)
                    colorful_print(token_text, fg="red", end="")
                elif has_reward:
                    # Token with reward
                    colorful_print(token_text, bg="green", end="")

                    reward_info = ""
                    if has_reward:
                        reward_info += f" R:{token_level_scores[i, j].item():.2f}"

                    colorful_print(reward_info, fg="magenta", end="")
                else:
                    # Normal token used in training
                    colorful_print(token_text, fg="blue", end="")

            print()  # New line after all tokens

            # Print reward summary
            total_reward = token_level_scores[i].sum().item()
            colorful_print("Rewards:", fg="green", bold=True)
            print(f" Trajectory Reward={total_reward:.2f}")

    def generate_agent_trajectories_async(self, timing_raw=None, meta_info=None, mode="Token"):
        """
        Generates agent trajectories asynchronously using the agent execution engine.

        This method runs the asynchronous `trajectory_generator` in a
        separate thread and yields the results synchronously through a queue.
        This allows the main training loop (which might be synchronous) to consume
        asynchronously generated trajectories.

        Args:
            timing_raw (dict, optional): Dictionary to store timing information. Defaults to {}.
            meta_info (dict, optional): Additional metadata for the generation process. Defaults to None.

        Yields:
            Any: Items generated by the `trajectory_generator`, typically
                 representing parts or results of agent trajectories in token format.
        """
        if timing_raw is None:
            timing_raw = {}
        queue = Queue()

        def runner():
            async def consume():
                async for item in self.agent_execution_engine.trajectory_generator(timing_raw=timing_raw, mode=mode, meta_info=meta_info):
                    queue.put(item)
                queue.put(None)  # sentinel to signal done

            asyncio.run(consume())

        Thread(target=runner, daemon=True).start()
        while True:
            item = queue.get()
            if item is None:
                break
            yield item

    def _transform_agent_steps(self, steps: list[dict], uids: np.ndarray):
        from verl.utils.torch_functional import pad_sequence_to_length

        all_prompts_list = []
        all_responses_list = []

        step_numbers = []  # number of steps of each episode, 0 indexed
        all_steps_idx_list = []
        all_steps_is_last_step_list = []
        all_steps_step_num = []  # total number of steps the trajectory this step belongs to have
        all_steps_step_ids = []
        training_rewards = []
        all_mc_returns = []  # Monte Carlo returns for each episode
        # the last step will have reward assigned and be used for advantage calculation

        for episode in steps:
            episode_steps = episode["steps"]
            idx = episode["idx"]
            training_reward = episode["trajectory_reward"]
            mc_returns = episode["mc_returns"]

            all_prompts_list.extend([torch.tensor(self.tokenizer.encode(s["prompt"], add_special_tokens=False), dtype=torch.long) for s in episode_steps])
            all_responses_list.extend([torch.tensor(self.tokenizer.encode(s["response"], add_special_tokens=False), dtype=torch.long) for s in episode_steps])

            step_numbers.append(len(episode_steps) - 1)
            training_rewards.append(training_reward)
            all_mc_returns.extend(mc_returns)

            all_steps_idx_list.extend([idx for _ in range(len(episode_steps))])
            all_steps_is_last_step_list.extend([False for _ in range(len(episode_steps))])
            all_steps_is_last_step_list[-1] = True

            all_steps_step_num.extend([len(episode_steps) for _ in range(len(episode_steps))])
            all_steps_step_ids.extend([f"{uids[idx]}_step{i}" for i in range(len(episode_steps))])

        # Convert all steps into token tensors
        # reverse the list and create tensors, pad, then flip to achieve left padding
        prompts_batch = torch.nn.utils.rnn.pad_sequence(
            [torch.flip(i, dims=[0]) for i in all_prompts_list],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        ).flip(dims=[1])

        prompts_batch = pad_sequence_to_length(prompts_batch, self.config.data.max_prompt_length, self.tokenizer.pad_token_id, left_pad=True)

        response_batch = torch.nn.utils.rnn.pad_sequence(
            all_responses_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )

        max_response_length = self.config.data.max_response_length
        response_batch = pad_sequence_to_length(response_batch, max_response_length, self.tokenizer.pad_token_id, left_pad=False)

        complete_step_batch = torch.concat([prompts_batch, response_batch], dim=1)
        attention_mask = torch.where(complete_step_batch != self.tokenizer.pad_token_id, 1, 0)
        position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

        # same as regular repsonse_mask, padded tensors will have this zeroed out
        traj_mask = torch.where(response_batch != self.tokenizer.pad_token_id, 1, 0)

        # Place all rewards to last response token of the last_step response
        score_batch = torch.zeros_like(response_batch, dtype=torch.float32)
        mc_return_batch = torch.zeros_like(response_batch, dtype=torch.float32)

        prompt_length = prompts_batch.shape[1]
        valid_response_length_sequences = attention_mask[:, prompt_length:].sum(dim=-1)

        # reward is given for last token of every step for logging purposes, but only last steps will be used to calculate advantage
        step_index = 0
        for i, traj_score in enumerate(training_rewards):
            step_num = step_numbers[i] + 1  # since step_numbers is 0 indexed
            for _ in range(step_num):
                last_valid_idx = valid_response_length_sequences[step_index] - 1
                if last_valid_idx >= 0 and last_valid_idx < score_batch.shape[1]:
                    score_batch[step_index, last_valid_idx] = traj_score
                    mc_return_batch[step_index, last_valid_idx] = all_mc_returns[step_index]
                step_index += 1
        assert step_index == score_batch.shape[0], f"Number of total steps used should equal to batch size, but got {step_index} and {score_batch.shape[0]}"

        tensor_batch = {
            "input_ids": complete_step_batch,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": response_batch,
            "prompts": prompts_batch,
            "token_level_scores": score_batch,
            "mc_returns": mc_return_batch,
            "traj_mask": traj_mask,
        }

        batch_id = str(uuid.uuid4())
        non_tensor_batch = {
            "idxs": np.array(all_steps_idx_list),
            "step_nums": np.array(all_steps_step_num),
            "is_last_step": np.array(all_steps_is_last_step_list),
            "is_pad_step": np.array([False for _ in range(len(all_steps_idx_list))]),
            "batch_id": np.array([batch_id for _ in range(len(all_steps_idx_list))]),  # in case need to differentiate which iteration the step is coming from
            "step_ids": np.array(all_steps_step_ids),
        }

        meta_info = {"repeat_counts": [x + 1 for x in step_numbers]}

        result = DataProto.from_dict(tensors=tensor_batch, non_tensors=non_tensor_batch, meta_info=meta_info)

        # Find indices of last steps for visualization
        last_step_indices = [i for i, is_last in enumerate(non_tensor_batch["is_last_step"]) if is_last]
        if last_step_indices:
            sample_indices = np.random.choice(last_step_indices, size=min(2, len(last_step_indices)), replace=False)
            for idx in sample_indices:
                self.visualize_trajectory(result, sample_idx=idx, max_samples=1)
        return result

    def _stepwise_advantage_broadcast(self, last_step_batch, other_step_batch):
        """
        Broadcast the advantage from last_step_batch to all other steps.
        """

        # NOTE: Currently takes the average of advantages. For GRPO, advantage and returns is uniform for each token so this makes no difference.
        # NOTE: For simplicity, assumes advantage and return is the same, which also holds for GRPO variants
        if "response_mask" not in other_step_batch.batch.keys():
            other_step_batch.batch["response_mask"] = compute_response_mask(other_step_batch)
        if "response_mask" not in last_step_batch.batch.keys():
            last_step_batch.batch["response_mask"] = compute_response_mask(last_step_batch)
        src_indices = last_step_batch.non_tensor_batch["idxs"]
        src_total_steps = last_step_batch.non_tensor_batch["step_nums"]
        tgt_indices = other_step_batch.non_tensor_batch["idxs"]
        src_advantages = last_step_batch.batch["advantages"]
        src_mask = last_step_batch.batch["response_mask"]
        tgt_mask = other_step_batch.batch["response_mask"]

        # Build idx -> scalar advantage
        idx_to_scalar_adv = {}
        for i, idx in enumerate(src_indices):
            mask = src_mask[i].bool()
            scalar = src_advantages[i][mask].mean()

            if self.config.agent.normalize_step_advantage:
                # normalize the advantage against number of steps
                scalar = scalar / src_total_steps[i]
                # reassign the normalized advantage to last_step_batch as well
                last_step_batch.batch["advantages"][i][mask] = scalar

            idx_to_scalar_adv[int(idx)] = scalar

        # Create new tensor for other_step_batch with per-token assignment
        scalar_rows = torch.stack([torch.full_like(tgt_mask[i], fill_value=idx_to_scalar_adv[int(idx)], dtype=torch.float32) for i, idx in enumerate(tgt_indices)])  # shape: (N2, T)

        # Apply the response mask of the target batch
        final_advantage = scalar_rows * tgt_mask

        # Assignment
        other_step_batch.batch["advantages"] = final_advantage
        other_step_batch.batch["returns"] = final_advantage

    def _pad_dataproto_to_world_size(self, batch):
        world_sizes = []
        if self.use_critic and self.critic_wg.world_size != 0:
            world_sizes.append(self.critic_wg.world_size)
        if self.use_reference_policy and self.ref_policy_wg.world_size != 0:
            world_sizes.append(self.ref_policy_wg.world_size)
        if self.use_rm and self.rm_wg.world_size != 0:
            world_sizes.append(self.rm_wg.world_size)
        if self.hybrid_engine:
            if self.actor_rollout_wg.world_size != 0:
                world_sizes.append(self.actor_rollout_wg.world_size)
        else:
            if self.actor_wg.world_size != 0:
                world_sizes.append(self.actor_wg.world_size)
            if self.rollout_wg.world_size != 0:
                world_sizes.append(self.rollout_wg.world_size)
        if not world_sizes:
            return batch

        world_size = reduce(math.lcm, world_sizes)

        original_batch_size = batch.batch["prompts"].shape[0]
        batch, pad_size = pad_dataproto_to_divisor(batch, world_size)

        # for the padded dataproto, make the traj mask to 0. is_last_step also False
        for i in range(pad_size):
            idx = original_batch_size + i
            batch.non_tensor_batch["is_last_step"][idx] = False
            batch.non_tensor_batch["is_pad_step"][idx] = True

        return batch
    
    def compute_overlong_penalty(self, batch):
        overlong_buffer_len = self.config.reward_model.overlong_buffer.len
        max_response_length = self.config.data.max_response_length
        expected_len = max_response_length - overlong_buffer_len
        prompt_len = batch.batch["prompts"].shape[1]
        valid_response_len = np.array([
            batch[i].batch["attention_mask"][:prompt_len].sum()
            for i in range(len(batch))
        ])
        prompt_token_lens = batch.non_tensor_batch["prompt_token_lens"]
        exceed_len = (valid_response_len - prompt_token_lens) - max_response_length
        penalty_factor = self.config.reward_model.overlong_buffer.penalty_factor
        penalty = np.minimum(-exceed_len / expected_len * penalty_factor, 0)
        penalty = torch.tensor(penalty, dtype=torch.float32)
        return penalty
