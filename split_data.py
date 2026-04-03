"""
给 annotation.csv 的 outside 数据添加 Split 列：70% train, 30% val。
inside 数据的 Split 列设为 "test"。
只需要运行一次，生成新数据后重新运行。

python split_data.py
"""

import pandas as pd
import numpy as np

CSV_PATH = "data/low_mid_level_vision/un_crowding/annotation.csv"
RANDOM_SEED = 42
VAL_RATIO = 0.3

df = pd.read_csv(CSV_PATH)

# 默认全部设为 test（inside 数据）
df["Split"] = "test"

# 对 outside 数据做随机 70/30 分割
outside_mask = df["VernierInOut"] == "outside"
outside_indices = df[outside_mask].index.to_numpy()

rng = np.random.default_rng(RANDOM_SEED)
rng.shuffle(outside_indices)

val_size = int(len(outside_indices) * VAL_RATIO)
val_indices = outside_indices[:val_size]
train_indices = outside_indices[val_size:]

df.loc[train_indices, "Split"] = "train"
df.loc[val_indices, "Split"] = "val"

# 保存
df.to_csv(CSV_PATH, index=False)

# 统计
print(f"Outside train: {len(train_indices)}")
print(f"Outside val:   {len(val_indices)}")
print(f"Inside test:   {outside_mask.eq(False).sum()}")
print(f"\nSaved to: {CSV_PATH}")