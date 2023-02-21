# Import necessary libaries
print("Importing libaries")

import pandas as pd 
import json 
import gc
from pathlib import Path
from lightgbm import LGBMRegressor
from numerapi import NumerAPI
from sklearn.model_selection import (
    train_test_split, 
    GridSearchCV
)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import matplotlib.pylab as plt
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

napi = NumerAPI()
current_round = napi.get_current_round()

# Download Datasets into v4 folder, Convert datasets to csv 

Path("./v4").mkdir(parents=False, exist_ok=True)
napi.download_dataset("v4/train.parquet")
napi.download_dataset("v4/validation.parquet")
napi.download_dataset("v4/live.parquet", f"v4/live_{current_round}.parquet")
napi.download_dataset("v4/validation_example_preds.parquet")
napi.download_dataset("v4/features.json")

# Acquire training data from pervious era, organize into feature columns

with open("v4/features.json", 'r') as f: 
    feature_metadata = json.load(f)
features = feature_metadata["feature_sets"]["small"]
read_columns = features + [ERA_COL, DATA_TYPE_COL, TARGET_COL]
training_data = pd.read_parquet('v4/train.parquet', columns=read_columns)
validation_data = pd.read_parquet('v4/validation.parquet', columns=read_columns)
live_data = pd.read_parquet(f'v4/live_{current_round}.parquet')


# Spliting data into independent and dependent variables for training

x = training_data.loc[:,features]
y = training_data.loc[:,ERA_COL]
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size= 0.2, random_state=42)

print(x_train, x_valid, y_train, y_valid)

# Creating model and training data

model = GradientBoostingClassifier(n_estimators=5000, learning_rate=0.05, max_depth=3, subsample= 0.5, validation_fraction=0.1, n_iter_no_change=20, max_features='log2', verbose=1)
print(model.fit(x_train, y_train))





