# Mostly lifted from gem/examples/train_oat_mt.py,
# then broken up and modified for clarity.

import functools
import json
import logging
import os
import re
import random
from copy import deepcopy
from dataclasses import dataclass
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
INVALID_ACTION = "<｜INVALID_ACTION｜>"

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
    env_id: str = "rg:leg_counting"
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
    eval_dump_game_states:  bool = True  # Whether to dump game states during evaluation
    # eval_data: Optional[str] = "./data"
    # eval_input_key: str = "input"
    # eval_output_key: str = "answer"
    # eval_split: str = "all"

    # Misc settings
    dump_experience_every: int = 1  # Dump experience data

    # Episode collection logic
    keep_generation_failed: bool = False  # Keep episodes with generation failures

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
                "backend": "kiwix", 
                "trawler_kwargs": {
                    "url": "http://localhost:8080",
                    "zimfile": "wikipedia_en_simple_all_nopic_2025-09",
                    "query_delay_ms": 0,
                    "query_use_cache": True,
                },
                "page_summary_length": (150, 'characters')

            } for j in range(self.args.num_env)],
            wrappers=wrappers,
            async_mode=self.args.async_env,
        )

    def collect_experience(self):
        logging.info(
            f"Actor-{self.actor_id} starting to collect experiences at step {self.step_count}"
        )
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
                executable_actions.append(INVALID_ACTION)
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
                    INVALID_ACTION
                    if response_is_truncated
                    else self.extract_action(raw_action)
                )
                executable_actions.append(
                    INVALID_ACTION if response_is_truncated else raw_action
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
                        "action_is_formatted": extracted_action != INVALID_ACTION,
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
                formatted_action = INVALID_ACTION

            return formatted_action

        except Exception as e:
            logging.error(f"Error in extract_action: {e}")
            # Return invalid action if extraction fails.
            return INVALID_ACTION
    
    def run_evaluate_episode(self):
        '''
        As with the classical RL setup, evaluate on the same environment.

        Logs trajectories and statistics into a local archive and optionally
        to WandB.
        '''
        rewards = []
        trajectories = []
        self.eval_mode = True
        for _ in range(self.args.eval_games):
            curr_rew = 0.0
            curr_traj = []
            obs, _ = self.env.reset()
            done = False
            while not done:
                action, extra = self.agent_act(obs)  # type: ignore
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated | truncated
                curr_rew += reward
                curr_traj.append(extra.get(key, "") for key in ["formatted_observation", "response", "extracted_action"])
            rewards.append(curr_rew)
            trajectories.append(curr_traj)

        # Log the evaluation results
        self.log_evaluation_results(rewards)
        self.eval_mode = False

    def log_evaluation_results(self, rewards):
        """
        Log the evaluation results, including average reward and other statistics.
        """
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        logging.info(f"Evaluation results - Average Reward: {avg_reward}")

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

        # Create the dataloaders
        self.prompts_dataloader = strategy.setup_dataloader(
            self.prompts_dataset,
            strategy.args.rollout_batch_size_per_device,
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
        t0 = time.time()
        # ------------------------------------------------------------------
        # Initialize metrics tracking
        # ------------------------------------------------------------------
        # For now we are only playing one game.
        # In future we can extend to multiple games by passing a list of env_ids.
        eval_env_ids = [self.args.env_id]

        # ------------------------------------------------------------------
        # Rank 0 distributes evaluation workloads to all ranks then collects and populates metrics
        # ------------------------------------------------------------------
        if self.strategy.is_rank_0():
            total_games = self.args.eval_games

            # Generate evaluation runs
            eval_runs_list = []
            for env_id in eval_env_ids:
                for game_nr in range(total_games):
                    eval_runs_list.append((env_id, game_nr))

            # Run evaluation
            futs = []
            game_stats = {}
            progress_bar = tqdm(range(len(eval_runs_list)), desc="Evaluating")
            random.shuffle(eval_runs_list)

            for i, (env_id, game_nr) in enumerate(eval_runs_list):
                actor = self.actors[i % len(self.actors)]
                futs.append(actor.futures.run_eval_episode(env_id))

                # Process results in batches
                if len(futs) == len(self.actors) or i == len(eval_runs_list) - 1:
                    for fut in futs:
                        result, game_history = fut.result()
                        game_stats[game_nr] = {
                            'metadata': {
                                'env_id': env_id,
                                'game_nr': game_nr,
                            },
                            'metrics': result,
                            'history': game_history,
                        }
                        progress_bar.update(1)

                    futs.clear()

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

        dist.barrier()


def train(args: Args):
    """
    Reinforcement learning starts here.

    Args:
        args: Configuration arguments for the run
    """
    # Define a distributed program that composes Actors and Learners
    program, local_resources = get_program(args, learner_cls=Learner, actor_cls=Actor)

    # assert args.max_model_len == 12_000, f"wtf why {args.max_model_len} not 12K???"
    print(args.max_model_len)

    # Launch the program
    lp.launch(
        program,
        launch_type=args.launch_type,
        local_resources=local_resources,
        terminal="current_terminal",
    )


if __name__ == "__main__":
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