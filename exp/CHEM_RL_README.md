# 化学分子生成 RL 训练系统

## 📋 概述

为 EGFR 抑制剂分子生成任务添加完整的 RL 训练支持，基于 GSPO 算法框架。

## 🎯 已实现功能

### 1. 数据集生成脚本
**文件**: `scripts/generate_chem_rl_dataset.py`

将 SFT 格式的化学数据集转换为 RL 训练所需的 parquet 格式。

**使用方法**:
```bash
python scripts/generate_chem_rl_dataset.py \
    /path/to/egfr_molecules_train.json \
    /path/to/egfr_molecules_val.json \
    /path/to/egfr_molecules_test.json \
    --output-dir ~/data/chem_rl \
    --train-size 8000 \
    --val-size 1000 \
    --test-size 1000
```

**可配置参数**:
- `--train-size`: 训练集样本数（默认 8000）
- `--val-size`: 验证集样本数（默认 1000）
- `--test-size`: 测试集样本数（默认 1000）
- `--output-dir`: 输出目录

**输出格式**:
```python
{
    "data_source": "egfr_molecule",
    "prompt": [{"role": "user", "content": "Design a potent EGFR inhibitor..."}],
    "ability": "chemistry",
    "reward_model": {"style": "rule", "ground_truth": {}},
    "extra_info": {"split": "train", "index": 0}
}
```

### 2. 化学奖励函数
**文件**: `verl/utils/reward_score/chem.py`

基于多维度药物属性评估生成的分子，总分范围：-5.0 到 ~11.5

**评分维度**:
- ✅ **格式检查** (+1.0 / -5.0): `<SMILES>...</SMILES>` 标签完整性
- ✅ **有效性** (+0.5 / -3.0): RDKit 分子验证
- ✅ **QED** (0-2.0): 药物相似性
- ✅ **SA Score** (0-2.0): 合成可及性
- ✅ **LogP** (0-1.5): 亲脂性（理想范围 1-4）
- ✅ **MW** (0-1.0): 分子量（理想范围 200-500）
- ✅ **TPSA** (0-1.0): 极性表面积（理想范围 40-120）
- ✅ **Lipinski** (0-1.5): 利宾斯基五规则
- ✅ **EGFR 特异性** (0-1.0): 芳香环和可旋转键

**依赖**:
- RDKit（必需）
- SA_Score（可选，如不可用则给予中性分）

### 3. 化学 Reward Manager
**文件**: `verl/workers/reward_manager/chem_reward.py`

- 使用 `@register("chem")` 装饰器注册
- 调用 `chem.compute_score` 计算奖励
- 与 `logic_rl` 保持完全一致的接口

### 4. 训练启动脚本
**文件**: `exp/chem.sh`

基于 `gspo.sh` 改造，仅修改数据和奖励相关配置。

**使用前需要修改的路径**:
```bash
# 数据路径
TRAIN_DATA="/root/autodl-tmp/myverl/data/chem/train.parquet"
VAL_DATA="/root/autodl-tmp/myverl/data/chem/val.parquet"

# 模型路径（化学 SFT 模型）
MODEL_PATH="/root/autodl-tmp/myverl/models/chem-sft-model"

# 检查点目录
CKPT_DIR="/root/autodl-tmp/verldev/Verl_RL/ckpts/chem/gspo_1"

# 项目名称
PROJECT_NAME="Chemistry-EGFR-RL"
EXPERIMENT_NAME="gspo_1"
```

**启动训练**:
```bash
bash exp/chem.sh
```

## 🚀 完整工作流程

### 步骤 1: 准备 SFT 数据
确保你有三个 SFT 格式的 JSON 文件：
- `egfr_molecules_train.json`
- `egfr_molecules_val.json`
- `egfr_molecules_test.json`

每个文件包含：
```json
{
    "instruction": "Design a potent EGFR inhibitor...",
    "input": "",
    "output": "<SMILES>CN1CCN(c2ccc(Nc3ncc4nc(Nc5ccccc5)n(C)c4n3)cc2)CC1</SMILES>"
}
```

### 步骤 2: 转换为 RL 数据集
```bash
python scripts/generate_chem_rl_dataset.py \
    /path/to/egfr_molecules_train.json \
    /path/to/egfr_molecules_val.json \
    /path/to/egfr_molecules_test.json \
    --output-dir /root/autodl-tmp/myverl/data/chem
```

### 步骤 3: 修改训练脚本路径
编辑 `exp/chem.sh`，更新：
- 数据路径
- 化学 SFT 模型路径
- 检查点保存路径

### 步骤 4: 启动训练
```bash
bash exp/chem.sh
```

## 🔧 关键设计决策

1. **标准 Prompt 格式**: 不需要手动添加特殊 token，`RLHFDataset` 会自动调用 `apply_chat_template`

2. **空 ground_truth**: 保持接口兼容但不使用，因为分子生成无标准答案

3. **评分返回 float**: 与 `logic_rl` 保持一致，内部计算多维度分数后返回总分

4. **完全复用框架**: 仅替换数据和奖励函数，其他代码（算法、优化器等）完全不变

## 📝 注意事项

1. **RDKit 依赖**: 确保已安装 RDKit：
   ```bash
   pip install rdkit
   ```

2. **SA_Score（可选）**: 如需 SA Score 评分，需要确保 RDKit 的 Contrib 目录包含 SA_Score

3. **响应长度**: 默认设置为 512，根据分子 SMILES 长度可以调整

4. **数据集大小**: 在 `generate_chem_rl_dataset.py` 顶部修改或通过命令行参数指定

5. **模型格式**: 模型需要已经通过 SFT 学会输出 `<SMILES>...</SMILES>` 格式

## ✅ 验证

所有文件已创建并通过 linting 检查：
- ✅ `scripts/generate_chem_rl_dataset.py`
- ✅ `verl/utils/reward_score/chem.py`
- ✅ `verl/workers/reward_manager/chem_reward.py`
- ✅ `verl/workers/reward_manager/__init__.py` (已更新)
- ✅ `exp/chem.sh`

## 🎉 完成！

现在你可以开始训练化学分子生成的 RL 模型了。祝训练顺利！

