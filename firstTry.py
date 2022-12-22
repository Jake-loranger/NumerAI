# Import necessary libaries
print("Importing libaries")

import pandas as pd 
import json 
import gc
from pathlib import Path
from lightgbm import LGBMRegressor
from numerapi import NumerAPI
from utils import (
    save_model,
    load_model,
    neutralize,
    get_biggest_change_features,
    validation_metrics,
    ERA_COL,
    DATA_TYPE_COL, 
    TARGET_COL,
    EXAMPLE_PREDS_COL
)

# Download data from numberai 
print("Downloading data from numerAI's API")

napi = NumerAPI()
current_round = napi.get_current_round()

# Download Datasets into v4 folder, Convert datasets to csv 
print("Downloading dataset files into v4 folder")

Path("./v4").mkdir(parents=False, exist_ok=True)
napi.download_dataset("v4/train.parquet")
napi.download_dataset("v4/validation.parquet")
napi.download_dataset("v4/live.parquet", f"v4/live_{current_round}.parquet")
napi.download_dataset("v4/validation_example_preds.parquet")
napi.download_dataset("v4/features.json")

# Acquire training data from pervious era, organize into feature columns
print("Gathering training data")

with open("v4/features.json", 'r') as f: 
    feature_metadata = json.load(f)
features = feature_metadata["feature_sets"]["small"]
read_columns = features + [ERA_COL, DATA_TYPE_COL, TARGET_COL]
training_data = pd.read_parquet('v4/train.parquet', columns=read_columns)
validation_data = pd.read_parquet('v4/validation.parquet', columns=read_columns)
live_data = pd.read_parquet(f'v4/live_{current_round}.parquet')

# Calculates correlation of each feature vs the target
print("Acquiring feature correlations to target")

all_features_corr = training_data.groupby(ERA_COL).apply(
    lambda era: era[features].corrwith(era[TARGET_COL])
)

print(all_features_corr)






# Set parameters, Create model

# Run model, aquire the data

# Create data sheets to analyze in NumerAI

# 

