#!/usr/bin/env python3
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

"""
Convert SFT format chemistry dataset to RL training parquet format.
Reads JSON files with instruction/output pairs and converts to VERL RL dataset format.
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd

# ============================================================================
# CONFIGURABLE PARAMETERS - Modify these as needed
# ============================================================================
TRAIN_SIZE = 8000  # Number of samples for training set
VAL_SIZE = 1000    # Number of samples for validation set
TEST_SIZE = 1000   # Number of samples for test set

DATA_SOURCE = "egfr_molecule"
ABILITY = "chemistry"


def convert_sft_to_rl_format(sft_data: list[dict], split: str) -> list[dict]:
    """
    Convert SFT format data to RL training format.
    
    Args:
        sft_data: List of dicts with 'instruction', 'input', 'output' keys
        split: 'train', 'val', or 'test'
        
    Returns:
        List of dicts in RL format
    """
    rl_data = []
    
    for idx, item in enumerate(sft_data):
        instruction = item.get("instruction", "")
        # Note: We don't include the <SMILES> tags in the prompt
        # The model needs to learn to generate them
        
        rl_item = {
            "data_source": DATA_SOURCE,
            "prompt": [{"role": "user", "content": instruction}],
            "ability": ABILITY,
            "reward_model": {
                "style": "rule",
                "ground_truth": {}  # Empty dict - not used for molecule generation
            },
            "extra_info": {
                "split": split,
                "index": idx
            }
        }
        rl_data.append(rl_item)
    
    return rl_data


def main():
    parser = argparse.ArgumentParser(
        description="Convert SFT format chemistry dataset to RL training parquet format"
    )
    parser.add_argument(
        "sft_train_file",
        type=str,
        help="Path to SFT training JSON file (e.g., egfr_molecules_train.json)"
    )
    parser.add_argument(
        "sft_val_file",
        type=str,
        help="Path to SFT validation JSON file (e.g., egfr_molecules_val.json)"
    )
    parser.add_argument(
        "sft_test_file",
        type=str,
        help="Path to SFT test JSON file (e.g., egfr_molecules_test.json)"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="~/data/chem_rl",
        help="Output directory for parquet files (default: ~/data/chem_rl)"
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=TRAIN_SIZE,
        help=f"Number of training samples to use (default: {TRAIN_SIZE})"
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=VAL_SIZE,
        help=f"Number of validation samples to use (default: {VAL_SIZE})"
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=TEST_SIZE,
        help=f"Number of test samples to use (default: {TEST_SIZE})"
    )
    
    args = parser.parse_args()
    
    # Expand paths
    output_dir = Path(os.path.expanduser(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Chemistry RL Dataset Converter")
    print("=" * 80)
    
    # Process each split
    splits = {
        "train": (args.sft_train_file, args.train_size),
        "val": (args.sft_val_file, args.val_size),
        "test": (args.sft_test_file, args.test_size)
    }
    
    for split_name, (sft_file, max_size) in splits.items():
        print(f"\n[{split_name.upper()}] Processing {sft_file}...")
        
        # Load SFT data
        with open(sft_file, 'r', encoding='utf-8') as f:
            sft_data = json.load(f)
        
        print(f"  Loaded {len(sft_data)} samples")
        
        # Limit size if needed
        if len(sft_data) > max_size:
            print(f"  Limiting to {max_size} samples")
            sft_data = sft_data[:max_size]
        
        # Convert to RL format
        rl_data = convert_sft_to_rl_format(sft_data, split_name)
        
        # Convert to DataFrame and save as parquet
        df = pd.DataFrame(rl_data)
        output_file = output_dir / f"{split_name}.parquet"
        df.to_parquet(output_file, index=False)
        
        print(f"  Saved {len(rl_data)} samples to {output_file}")
        
        # Show first example
        if split_name == "train":
            print("\n" + "=" * 80)
            print("Example from training set:")
            print("=" * 80)
            example = rl_data[0]
            print(f"Data Source: {example['data_source']}")
            print(f"Ability: {example['ability']}")
            print(f"Prompt: {example['prompt']}")
            print(f"Reward Model: {example['reward_model']}")
            print(f"Extra Info: {example['extra_info']}")
    
    print("\n" + "=" * 80)
    print("Conversion complete!")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    print(f"Files created:")
    print(f"  - train.parquet ({args.train_size} samples)")
    print(f"  - val.parquet ({args.val_size} samples)")
    print(f"  - test.parquet ({args.test_size} samples)")
    print("\nYou can now use these parquet files in your RL training script.")


if __name__ == "__main__":
    main()

