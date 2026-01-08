# Copyright 2025 AxonRL Team. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Mostly lifted from gem/examples/train_oat_mt.py,
# then broken up and modified for clarity.

import functools
import json
import logging
import os
import re
import random
from copy import deepcopy
from dataclasses import dataclass, field
from time import time, sleep
from tqdm import tqdm
from typing import List, Literal, Optional, Tuple

import numpy as np
import torch.distributed as dist
import tree
import vllm
from oat.algorithms.ppo import PPOArgs
from oat.algorithms.ppo_multiturn import PPOMultiTurnActor, PPOMultiTurnLearner
from oat.args import default_args_validation, get_default_args
from oat.interface import get_program, lp
from oat.types import Transition
from oat.utils.ops import masked_sum
from torch.utils.data import Dataset

import gem
from gem.utils.parsing import extract_last_boxed_answer
from gem.wrappers.wrapper_factory import get_wrapper_fns
from gem.envs.game_env.wikigame.errors import BackendFailureException

""" +=========================================+ """
""" 1. Defining constants used in our training. """
""" +=========================================+ """

class NoSamplesLeftException(Exception):
    def __init__(self, message, err_code):
        super().__init__(message, err_code)
    def __str__(self):
        return "No valid samples to continue with!"

# Invalid action to be sent to the env to trigger format error penalty.
MALFORMED_ACTION = "<｜MALFORMED_ACTION｜>"

def apply_qwen3_game_template(observation: str) -> str:
    return (
        f"<|im_start|>user\nYou are playing language games. Make valid actions to win.\nObservation: {observation}"
        "\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def apply_no_template(observation: str) -> str:
    return observation

def apply_qwen3_general_template(question: str) -> str:
    return (
        f"<|im_start|>user\nQuestion: {question}"
        "\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def apply_code_template(question: str) -> str:
    return (
        "You are an expert Python programmer. "
        "You will be given a question (problem specification) and will generate a correct "
        "Python program that matches the specification and passes all tests."
        f"\nQuestion: {question}"
        "\nPlease reason step by step, and write your code in markdown format, e.g., ```python\n# YOUR CODE HERE\n```."
    )

TEMPLATE_FACTORY = {
    "qwen3_game": apply_qwen3_game_template,
    "no": apply_no_template,
    "qwen3_general": apply_qwen3_general_template,
    "code": apply_code_template,
}


""" +=================================================+ """
""" 2. Defining extra arguments/structure for training. """
""" +=================================================+ """


@dataclass
class Args(PPOArgs):
    # Environment settings
    env_id: str = "game:WikiGame-v0-hard"
    num_env: int = 1
    wrappers: str = ""
    async_env: bool = False

    # Algorithm settings
    length_norm_constant: Optional[int] = None

    # Template settings
    prompt_template: Literal["qwen3_game", "no", "qwen3_general", "code"] = "qwen3_game"

    # Reward settings
    gamma: float = 1.0  # Discount factor for Monte Carlo returns
    whiten_adv: bool = True  # Return batch normalization

    # Evaluation settings
    eval_steps: int = 32  # Evaluation interval in steps
    eval_games: int = 16  # Number of games for evaluation
    eval_dump_game_states: bool = True  # Whether to dump game states during evaluation
    eval_env_ids: List[str] = field(default_factory=lambda: ["game:WikiGame-v0-hard", "game:Sudoku-v0-easy", "qa:HotpotQA"])
    eval_only: bool = False  # If true, only run evaluation without training

    # Misc settings
    dump_experience_every: int = 1  # Dump experience data

    # Episode collection logic
    keep_generation_failed: bool = False  # Keep episodes with generation failures

    # Backend arguments
    wg_backend: Literal['kiwix', 'mw'] = "kiwix"
    wg_url: str = "http://localhost:8080"
    wg_query_delay_ms: int = 0
    wg_query_use_cache: bool = True
    wg_maxlen_value: int = 150
    wg_maxlen_unit: Literal['sentences', 'characters', 'words'] = 'characters'
    wg_variant: str = 'noregrets'

    # Kiwix-specific arguments
    kiwix_zimfile: str = "wikipedia_en_simple_all_nopic_2025-09"

""" +=======================================+ """
""" 3. Defining actor to collect experiences. """
""" +=======================================+ """


class Actor(PPOMultiTurnActor):
    def init(self, actor_id, save_path):
        super().init(actor_id, save_path)
        self.args.seed += 233 ** (actor_id + 1)
        self.game_state_save_path = os.path.join(self.save_path, "game_state")
        if actor_id == 0:
            os.makedirs(self.game_state_save_path, exist_ok=True)
        self.args: Args = self.args
        args = self.args
        self.oracle = None

        self.sampling_params = vllm.SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.generate_max_length,
            n=1,
            logprobs=True,
        )

        self.eval_sampling_params = vllm.SamplingParams(
            temperature=args.eval_temperature,
            top_p=args.eval_top_p,
            top_k=args.eval_top_k,
            max_tokens=args.eval_generate_max_length,
            n=1,
            logprobs=True,
        )

        self.step_count = 0

        # Get environment wrappers.
        wrappers = get_wrapper_fns(self.args.wrappers, tokenizer=self.tokenizer)

        # Instantiate vectorized environment.
        self.env = gem.make_vec(
            [self.args.env_id] * self.args.num_env,
            vec_kwargs=[{
                "seed": self.args.seed + j,
                "backend": self.args.wg_backend,
                "trawler_kwargs": {
                    "url": self.args.wg_url,
                    "zimfile": self.args.kiwix_zimfile,
                    "query_delay_ms": self.args.wg_query_delay_ms,
                    "query_use_cache": self.args.wg_query_use_cache,
                },
                "page_summary_length": (self.args.wg_maxlen_value, self.args.wg_maxlen_unit),
                "variant": self.args.wg_variant,
            } for j in range(self.args.num_env)],
            wrappers=wrappers,
            async_mode=self.args.async_env,
        )

        self.eval_envs = {}

        # Ensure different eval envs are created.
        for eval_env_id in tqdm(self.args.eval_env_ids, desc="Creating eval envs"):
            if 'wikigame' in eval_env_id.lower():
                self.eval_envs[eval_env_id] = gem.make_vec(
                    [eval_env_id] * self.args.num_env,
                    vec_kwargs=[{
                        "seed": self.args.seed + 1000 + j,
                        "backend": self.args.wg_backend,
                        "trawler_kwargs": {
                            "url": self.args.wg_url,
                            "zimfile": self.args.kiwix_zimfile,
                            "query_delay_ms": self.args.wg_query_delay_ms,
                            "query_use_cache": self.args.wg_query_use_cache,
                        },
                        "page_summary_length": (self.args.wg_maxlen_value, self.args.wg_maxlen_unit),
                        "variant": self.args.wg_variant,
                    } for j in range(self.args.num_env)],
                    wrappers=wrappers,
                    async_mode=self.args.async_env,
                )
            elif "qa" in eval_env_id.lower():
                self.eval_envs[eval_env_id] = gem.make_vec(
                    [eval_env_id] * self.args.num_env,
                    vec_kwargs=[{"seed": self.args.seed + 1000 + j, "extract_boxed": True} for j in range(self.args.num_env)],
                    wrappers=wrappers,
                    async_mode=self.args.async_env,
                )
            else:
                self.eval_envs[eval_env_id] = gem.make_vec(
                    [eval_env_id] * self.args.num_env,
                    vec_kwargs=[{"seed": self.args.seed + 1000 + j} for j in range(self.args.num_env)],
                    wrappers=wrappers,
                    async_mode=self.args.async_env,
                )
            sleep(1)  # Be nice to HF servers, if not you will get 502/443 error.

    def collect_experience(self):
        logging.info(
            f"Actor-{self.actor_id} starting to collect experiences at step {self.step_count}"
        )
        assert not self.args.eval_only, "Killswitch hit during experience collection!"
        env, min_steps = self.env, self.args.rollout_batch_size_per_device
        obs, _ = env.reset()
        done = False
        episodes = [[] for _ in range(env.num_envs)]
        finished_episodes = []
        finished_episodes_tool_uses = []
        finished_episodes_tool_success = []
        num_generation_failed = 0

        while True:

            try:
                action, extra = self.agent_act(obs)  # type: ignore
                next_obs, reward, terminated, truncated, info = env.step(action)
            except BackendFailureException as e:
                # Delay for 2 seconds to let the backend autonomously restart
                logging.error(
                    f"Actor-{self.actor_id} encountered BackendFailureException. "
                    "Delaying for 2 seconds to allow backend restart."
                )
                sleep(2)
                obs, _ = env.reset()
                continue
            except NoSamplesLeftException as e:
                logging.error(
                    f"Actor-{self.actor_id} ran out of valid samples to use. "
                    "Retrying... "
                )
                obs, _ = env.reset()
                continue

            done = terminated | truncated

            for i in range(env.num_envs):
                if extra[i]["generation_failed"]:
                    num_generation_failed += 1
                    if self.args.keep_generation_failed:
                        episodes[i][-1].reward += reward[i]
                        episodes[i][-1].done = True
                        finished_episodes.append(deepcopy(episodes[i]))
                        finished_episodes_tool_uses.append(
                            info[i].get("prev_ep_tool_use_counter", 0)
                            if done[i]
                            else info[i].get("tool_use_counter", 0)
                        )
                        finished_episodes_tool_success.append(
                            info[i].get("prev_ep_tool_success_counter", 0)
                            if done[i]
                            else info[i].get("tool_success_counter", 0)
                        )
                    episodes[i].clear()
                    if not done[i]:
                        next_obs[i] = env.envs[i].reset()[0]
                else:
                    transition = Transition(
                        obs=obs[i],
                        action=action[i],
                        rewards=reward[i],
                        done=done[i],
                        prompt=extra[i]["formatted_observation"],
                        prompt_ids=extra[i]["prompt_ids"],
                        response=extra[i]["response"],
                        response_ids=extra[i]["response_ids"],
                        response_logprobs=extra[i]["response_logprobs"],
                        response_is_truncated=extra[i]["response_is_truncated"],
                        action_is_formatted=extra[i]["action_is_formatted"],
                        info={},
                    )
                    episodes[i].append(transition)
                    if done[i]:
                        finished_episodes.append(deepcopy(episodes[i]))
                        finished_episodes_tool_uses.append(
                            info[i].get("prev_ep_tool_use_counter", 0)
                        )
                        finished_episodes_tool_success.append(
                            info[i].get("prev_ep_tool_success_counter", 0)
                        )
                        episodes[i].clear()

            obs = next_obs
            if len(tree.flatten(finished_episodes)) >= min_steps:
                break

        info = {
            "actor/num_generation_failed": num_generation_failed,
            "actor/prop_generation_failed": (
                num_generation_failed / len(finished_episodes)
                if self.args.keep_generation_failed
                else num_generation_failed
                / (len(finished_episodes) + num_generation_failed)
            ),
            "actor/num_tool_uses": np.mean(finished_episodes_tool_uses),
            "actor/num_tool_success": np.mean(finished_episodes_tool_success),
        }
        if self.step_count % self.args.dump_experience_every == 0:
            _to_dump = {}
            for i, ep in enumerate(finished_episodes):
                key = f"episode{i}"
                _to_dump[key] = []
                for transition in ep:
                    _to_dump[key].append(transition.format())
            with open(
                os.path.join(
                    self.game_state_save_path,
                    f"actor{self.actor_id}_step{self.step_count}.json",
                ),
                "w",
            ) as f:
                json.dump(
                    _to_dump,
                    f,
                    indent=4,
                )
        self.step_count += 1
        return finished_episodes, info

    def agent_act(self, vec_observation: List[str]) -> Tuple[str, dict]:
        """Use the current LLM as a policy to act.

        Args:
            vec_observation: Vectorized observation from TextArena environment.

        Returns:
            Tuple[str, dict]: Action and extra data.
        """
        formatted_observations = []
        for observation in vec_observation:
            observation = TEMPLATE_FACTORY[self.args.prompt_template](observation)
            if self.args.apply_chat_template:
                observation = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": observation}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            formatted_observations.append(observation)

        sampling_params = (
            self.eval_sampling_params if self.eval_mode else self.sampling_params
        )

        # Subsample to remove observations that exceed max model length
        idss = self.tokenizer(formatted_observations).input_ids
        exceeds_lengths = [len(ids) >= self.args.max_model_len for ids in idss]
        sub_formatted_observations = [
            o for o, e in zip(formatted_observations, exceeds_lengths) if not e
        ]
        if not len(sub_formatted_observations):
            raise NoSamplesLeftException()

        logging.info(f"\nNumber of observations: {len(sub_formatted_observations)}\n")
        sub_idss = self.tokenizer(sub_formatted_observations).input_ids
        logging.info(
            f"\nObservation lengths: {[len(ids) for ids in sub_idss]} out of {self.args.max_model_len}\n"
        )

        # Generate
        sub_outputs = self.generate(sub_formatted_observations, sampling_params)

        executable_actions = []
        extras = []
        sub_i = 0
        for i, exceeds_length in enumerate(exceeds_lengths):
            if exceeds_length:
                # if prompt exceeds max model length we skipped the generation
                executable_actions.append(MALFORMED_ACTION)
                extras.append({"generation_failed": True})
            else:
                raw_action = sub_outputs[sub_i].outputs[0].text
                prompt_token_ids = sub_outputs[sub_i].prompt_token_ids
                token_ids = sub_outputs[sub_i].outputs[0].token_ids
                response_logprobs = sub_outputs[sub_i].outputs[0].logprobs
                response_logprobs = [
                    item[token_ids[i]].logprob
                    for i, item in enumerate(response_logprobs)
                ]
                response_is_truncated = (
                    sub_outputs[sub_i].outputs[0].finish_reason == "length"
                )

                # Valid extraction = proper eos + proper format
                # Only used for metric logging
                extracted_action = (
                    MALFORMED_ACTION
                    if response_is_truncated
                    else self.extract_action(raw_action)
                )
                executable_actions.append(
                    MALFORMED_ACTION if response_is_truncated else raw_action
                )
                extras.append(
                    {
                        "formatted_observation": formatted_observations[i],
                        "prompt_ids": prompt_token_ids,
                        "response": raw_action,
                        "extracted_action": extracted_action,
                        "response_ids": token_ids,
                        "response_logprobs": response_logprobs,
                        "response_is_truncated": response_is_truncated,
                        "action_is_formatted": extracted_action != MALFORMED_ACTION,
                        "generation_failed": False,
                        "generation_max_length_reached": (
                            len(prompt_token_ids) + len(token_ids)
                            >= self.args.max_model_len
                        ),
                    }
                )
                sub_i += 1
        return executable_actions, extras  # type: ignore

    def extract_action(self, text: str) -> str:
        """
        Extract and format the actual action from the model's output.

        This method handles different template formats and ensures the action
        is properly formatted for the environment.

        Args:
            text: Raw text output from the model

        Returns:
            Cleaned and formatted action string ready for the environment
        """
        if not text:
            return ""  # Handle empty text case

        try:
            formatted_action = None
            if self.args.prompt_template in ["qwen3_game", "qwen3_general"] or (
                self.args.prompt_template == "no"
                and "qwen" in self.args.pretrain.lower()
            ):
                # Note: Regex is incapable of handling nested boxes.
                formatted_action = extract_last_boxed_answer(text)
                if formatted_action is None:
                    formatted_action = text.strip()
            elif self.args.prompt_template == "code":
                code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
                if not code_blocks:
                    formatted_action = None
                else:
                    formatted_action = code_blocks[-1].strip()
            else:
                raise NotImplementedError

            if formatted_action is None:
                formatted_action = MALFORMED_ACTION

            return formatted_action

        except Exception as e:
            logging.error(f"Error in extract_action: {e}")
            # Return invalid action if extraction fails.
            return MALFORMED_ACTION

    def run_eval_episode(self, env_id: Optional[str] = None) -> Tuple[dict, dict]:
        '''
        Screw it. Use multiple envs.
        '''
        # trial_seed = pow(int(time() * 1_000 + self.args.seed), 3, 10 ** 9 + 7)
        trajectories = []
        self.eval_mode = True

        env = self.eval_envs[env_id]
        term_rews = [0] * env.num_envs

        eval_result = {
            f"ep_{i}": {
                "reward_hist": [],
                "success": [],
                "response_lengths": [],
            } for i in range(env.num_envs)
        }
        trajectories = {
            f"ep_{i}": []
            for i in range(env.num_envs)
        }
        curr_rew = [0] * env.num_envs
        obs, _ = env.reset()
        dones = [False] * env.num_envs
        while not all(dones):
            action, extra = self.agent_act(obs) # type: ignore
            obs, reward, terminated, truncated, info = env.step(action)
            for i in range(env.num_envs):
                curr_rew[i] += reward[i]
                if not dones[i]:
                    # Only log for non-done envs
                    trajectories[f"ep_{i}"].append({key: extra[i].get(key, "DID NOT FIND") for key in ["formatted_observation", "response", "extracted_action"]})
                    eval_result[f"ep_{i}"]["response_lengths"].append(len(extra[i]["response_ids"]) if "response_ids" in extra[i] else -1)
                else:
                    term_rews[i] = reward[i]


            dones = terminated | truncated

        for i in range(env.num_envs):
            if (
                'Congratulations' in obs[i]
                or (
                    # I.e. reward is 1.0
                    'qa' in env_id.lower() and term_rews[i] >= 0.999999999
                )
            ):
                eval_result[f"ep_{i}"]["success"].append(1)
            else:
                eval_result[f"ep_{i}"]["success"].append(0)
            eval_result[f"ep_{i}"]["reward_hist"].append(curr_rew[i])

        return eval_result, trajectories


class DummyPromptDataset(Dataset):
    """Empty dataset to satisfy OAT's requirements without actually loading data."""

    def __init__(self, size=1):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        del idx
        return "", "", ""

""" +====================================+ """
""" 4. Defining learner update the policy. """
""" +====================================+ """


class Learner(PPOMultiTurnLearner):
    def _init(self, args: Args, actors: List[Actor]) -> None:
        """
        Initialize the learner.
        """
        # Call parent's _init but then override prepare_data
        super()._init(args, actors)
        self.args = args

        # Masked sum is the correct implementation!
        # Oat by default uses Dr.GRPO: https://arxiv.org/pdf/2503.20783
        self.masked_aggregator = functools.partial(
            masked_sum,
            constant_normalizer=args.length_norm_constant or args.generate_max_length,
        )

    def prepare_data(self, strategy, tokenizer):
        """
        Override the data preparation to avoid loading external datasets.
        Instead, create dummy datasets just to keep OAT's infrastructure happy.
        """
        # Create dummy dataset that satisfies OAT's requirements
        # but doesn't actually load any data
        # Used to control the training episode, set a large number.
        self.prompts_dataset = DummyPromptDataset(size=int(1e9))
        self.eval_prompts_dataset = DummyPromptDataset(size=self.args.eval_games)

        # Create the dataloaders
        self.prompts_dataloader = strategy.setup_dataloader(
            self.prompts_dataset,
            strategy.args.rollout_batch_size_per_device,
            shuffle=False,  # No need to shuffle dummy data
        )
        self.eval_prompts_dataloader = strategy.setup_dataloader(
            self.eval_prompts_dataset,
            strategy.args.eval_batch_size,
            shuffle=False,  # No need to shuffle dummy data
        )

    def evaluate(self, _unused_dataloader, steps):
        """
        Online evaluation with hierarchical metrics.

        We do three things here:
        1) Evaluation on games, either in-domain or out-domain, against various opponents (random, rule-based, LLMs);
        2) Evaluation on general reasoning tasks, including math, etc.
        """
        del _unused_dataloader
        assert not self.pi_beta_lags_behind, "pi beta lags behind for evaluation"
        self._pre_evaluate()

        self.strategy.print(f"Start evaluating on games at step {steps}")

        # 1) Game eval.
        t0 = time()
        # ------------------------------------------------------------------
        # Initialize metrics tracking
        # ------------------------------------------------------------------
        # For now we are only playing one game.
        # In future we can extend to multiple games by passing a list of env_ids.
        # eval_env_ids = [self.args.env_id, "game:Sudoku-v0-easy", "qa:HotpotQA"]

        # ------------------------------------------------------------------
        # Rank 0 distributes evaluation workloads to all ranks then collects and populates metrics
        # ------------------------------------------------------------------
        game_stats = {env_id: {"episodes": [], } for env_id in self.args.eval_env_ids}
        if self.strategy.is_rank_0():
            total_games = self.args.eval_games

            # Generate evaluation runs
            eval_runs_list = []
            for env_id in self.args.eval_env_ids:
                for game_nr in range(0, total_games, self.args.num_env):
                    eval_runs_list.append((env_id, game_nr))

            # Run evaluation
            futs = []
            progress_bar = tqdm(range(len(eval_runs_list)), desc="Evaluating")
            random.shuffle(eval_runs_list)

            for i, (env_id, game_nr) in enumerate(eval_runs_list):
                actor = self.actors[i % len(self.actors)]
                futs.append((env_id, actor.futures.run_eval_episode(env_id)))
                logging.info(f"Dispatched {env_id}, [{game_nr}... +{self.args.num_env}] to actor {i % len(self.actors)}")

                # Process results in batches
                if len(futs) == len(self.actors) or i == len(eval_runs_list) - 1:
                    for env_id_fut, fut in futs:
                        result, game_history = fut.result()
                        game_stats[env_id_fut]["episodes"].extend([{
                            'game_history': game_history[f"ep_{j}"],
                            'result': {
                                'reward_hist': result[f"ep_{j}"]["reward_hist"],
                                'success': result[f"ep_{j}"]["success"],
                                'response_lengths': result[f"ep_{j}"]["response_lengths"],
                            },
                        } for j in range(self.args.num_env)])
                            
                        progress_bar.update(1)
                    futs.clear()
                
            # Compute final metrics
            for env_id in game_stats.keys():
                episodes = game_stats[env_id]["episodes"]
                game_stats[env_id].update({
                    "eval/success_rate": np.mean(
                        [
                            ep["result"]["success"][-1]
                            for ep in game_stats[env_id]["episodes"]
                        ]
                    ) if episodes else 0.0,
                    "eval/avg_reward": np.mean(
                        [
                            sum(ep["result"]["reward_hist"])
                            for ep in game_stats[env_id]["episodes"]
                        ]
                    ) if episodes else 0.0,
                    "eval/avg_response_length": np.mean(
                        [
                            np.mean(ep["result"]["response_lengths"])
                            for ep in game_stats[env_id]["episodes"]
                        ]
                    ) if episodes else 0.0,
                })

            if self.args.eval_dump_game_states:

                eval_results_dir = os.path.join(
                    self.save_path, "eval_results",
                )

                os.makedirs(eval_results_dir, exist_ok = True)

                eval_results_path = os.path.join(
                    eval_results_dir,
                    f"{steps}_eval_game.json",
                )
                
                json.dump(
                    game_stats,
                    open(eval_results_path, "w"),
                    indent=4,
                )
            
            logging.info(
                f"Finished evaluating on games at step {steps} in {time() - t0:.2f} seconds"
            )

            # So apparently deepspeed is EXTREMELY picky about what you can log as a Tensor...
            # To comply we have no choice but to flatten the dictionary here.
            game_stats_flat = {
                f"eval/{env_id}/{key}": value
                for env_id, stats in game_stats.items()
                for key, value in stats.items()
                if key != "episodes"
            }
            game_stats_flat["eval/time_taken_seconds"] = time() - t0
            game_stats_flat["eval/step"] = steps
        else:
            game_stats_flat = None

        obj_list = [game_stats_flat]
        dist.broadcast_object_list(obj_list, src=0)
        game_stats_flat = obj_list[0]
        # CPU barrier to ensure all ranks sync here
        dist.barrier(group=self._same_actor_group)
        logging.info(f"rank {self.strategy.get_rank()} cpubarrier done")
        dist.barrier()
        self._post_evaluate()
        return game_stats_flat


def train(args: Args):
    """
    Reinforcement learning starts here.

    Args:
        args: Configuration arguments for the run
    """
    # Define a distributed program that composes Actors and Learners
    program, local_resources = get_program(args, learner_cls=Learner, actor_cls=Actor)

    print(args.max_model_len)

    # Launch the program
    lp.launch(
        program,
        launch_type=args.launch_type,
        local_resources=local_resources,
        terminal="current_terminal",
    )

if __name__ == "__main__":

    # Register custom envs
    from gem.envs.registration import register
    register(
        "qa:HotpotQAWithDocs",
        "gem.envs.qa_env:HotpotQaEnv",
        dataset_name="hotpotqa/hotpot_qa",
        split="train",
        question_key="question",
        documents_key="context",
        answer_key="answer",
    )

    register(
        "qa:2WikiQAWithDocs",
        "gem.envs.qa_env:HotpotQaEnv",
        dataset_name="framolfese/2WikiMultihopQA",
        split="train",
        question_key="question",
        documents_key="context",
        answer_key="answer",
        subset_key="default",
    )   

    # Get default arguments and customize them
    args: Args = get_default_args(Args)

    # Customization
    args.algo = "PPO"
    args.eval_batch_size = 32

    # CRITICAL: Disable oracle and dataset loading
    args.oracle = ""  # Empty string for no external oracle
    args.prompt_data = ""  # Don't load any dataset
    args.rollout_batch_size = args.rollout_batch_size_per_device * args.gpus
    if "concat_chat" in args.wrappers:
        assert (
            args.prompt_template == "no"
        ), "chat template is applied on env side already"
    args = default_args_validation(args)

    # Let's go
    train(args)
