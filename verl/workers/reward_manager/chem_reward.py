# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import chem
from .registry import register


@register("chem")
class ChemRewardManager:
    """
    A reward manager for chemistry molecule generation tasks.
    Evaluates generated molecules based on format, validity, and drug-like properties.
    """

    def __init__(self, tokenizer, num_examine, reward_fn_key="data_source", **kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict: bool = False):
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": {}}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}

        print(f"🧪 [Chemistry Reward] reward_tensor.shape={reward_tensor.shape}, len(data)={len(data)}")

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # Decode prompt and response
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            # Get ground_truth (not used for chemistry, but kept for API compatibility)
            ground_truth = data_item.non_tensor_batch["reward_model"].get("ground_truth", {})
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            
            # Compute chemistry reward
            score = chem.compute_score(response_str, ground_truth)
            reward_extra_info["chem_score"].append(score)
            reward_tensor[i, valid_response_length - 1] = score

            print(f"🧪 [Chemistry Reward] Sample {i}: score={score:.2f}, position=({i}, {valid_response_length - 1})")

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str[:200] if len(prompt_str) > 200 else prompt_str)
                print("[response]", response_str[:200] if len(response_str) > 200 else response_str)
                print("[score]", score)
        
        print(f"🧪 [Chemistry Reward] Score tensor: min={reward_tensor.min():.2f}, "
              f"max={reward_tensor.max():.2f}, mean={reward_tensor[reward_tensor != 0].mean():.2f}")

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            return reward_tensor

