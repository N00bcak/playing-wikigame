# Copyright 2025 AxonRL Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Env for question answering datasets."""

import logging
import random
from functools import partial
from typing import Any, Optional, Tuple

from datasets import Dataset, DatasetDict, load_dataset

from gem.envs.qa_env import QaEnv, apply_prompt
from gem.utils.parsing import extract_last_boxed_answer, extract_last_tagged_answer


logger = logging.getLogger(__name__)

class HotpotQaEnv(QaEnv):
    """HotpotQA environment."""

    def __init__(
        self,
        dataset_name: Optional[str] = "",
        split: Optional[str] = None,
        dataset: Optional[Dataset] = None,
        question_key: str = "question",
        answer_key: str = "answer",
        seed: int = 0,
        extract_boxed: bool = False,
        load_from_cache_file: bool = True,
        subset_key: str = "distractor",
        documents_key: str = "context",
        **kwargs,
    ):
        super(QaEnv, self).__init__()
        self.seed = seed
        self.question_key = question_key
        self.answer_key = answer_key
        self.documents_key = documents_key
        if dataset is None:
            dataset = load_dataset(dataset_name, subset_key)
            logger.info(f"Loaded: {dataset=}")
        if isinstance(dataset, DatasetDict):
            if split is not None:
                dataset = dataset[split]
            elif len(list(dataset.keys())) == 1:
                dataset = dataset[list(dataset.keys())[0]]
            else:
                raise ValueError(
                    f"Dataset {dataset_name} has multiple splits. "
                    f"Please specify a split: {list(dataset.keys())}"
                )
        assert isinstance(dataset, Dataset), f"Expected a Dataset, got {type(dataset)}"
        apply_prompt_func = partial(apply_prompt, question_key=question_key)
        dataset = dataset.map(
            apply_prompt_func, load_from_cache_file=load_from_cache_file
        )
        self.dataset = dataset.shuffle(seed=self.seed)
        self.idx = 0
        self.epoch = 0

        if extract_boxed:
            self.extractor = extract_last_boxed_answer
        else:
            self.extractor = extract_last_tagged_answer

    def reset(self, seed: Optional[None] = None) -> Tuple[str, dict[str, Any]]:
        """Sample a question from the dataset."""

        super(QaEnv, self).reset(seed)
        if seed is not None:
            self.idx = random.randint(0, len(self.dataset) - 1)
        else:
            if self.idx == len(self.dataset):
                self.epoch += 1
                self.dataset = self.dataset.shuffle(seed=self.seed + self.epoch)
                self.idx = 0

        data = self.dataset[self.idx]
        self.first_obs = data[self.question_key]

        documents = data[self.documents_key]
        self.first_obs += "\n\nYou will receive some summaries to help you answer. Here are the summaries:"
        for doc_title, doc_sents in zip(documents["title"], documents["sentences"]):
            self.first_obs += f"\nTitle: {doc_title}\n"
            for sent in doc_sents:
                self.first_obs += f"{sent} "
            self.first_obs += "\n"

        self.answer = data[self.answer_key]
        self.idx += 1
        return self.first_obs, {}
