# Taken from u/no_formal_agreement @https://forum.numer.ai/t/learning-two-uncorrelated-models/400/4
# Using to learn various ways of acquire correlation values with SciPy ML libaries
# Portion was added to convert dataset from .parquet -> .csv from https://wandb.ai/parmarsuraj99/massive_nmr/reports/A-Super-Easy-Guide-to-the-Super-Massive-Numerai-Dataset--VmlldzoxMTM4OTU2 
# Jake L added libary and datasetimportations 

import pandas as pd
import numpy as np
import json 
import gc
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy import stats
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

def get_spearman_by_era(p,target,eras):
    
    df = pd.DataFrame(columns = ('p','target','eras'))
    df.loc[:,'p'] = p
    df.loc[:,'target'] = target
    df.loc[:,'eras'] = eras
    era_names = df.eras.unique()
    output_dict = dict()
    for i in era_names:
        p0 = df.loc[df.eras == i,'p']
        target0 = df.loc[df.eras == i,'target']
        output_dict[i] = stats.spearmanr(p0,target0)
    return output_dict

def get_sharpe_ratio(spearmans):
    output = np.zeros((len(spearmans)))
    j = 0
    for i in spearmans:
        output[j] = spearmans[i][0]
        j = j + 1
    return np.mean(output)/np.std(output)


# Following portion was added by Jake Loranger to create CSV of data
print('loading files')

napi = NumerAPI()
current_round = napi.get_current_round() 

# Tournament data changes every week so we specify the round in their name. Training
# and validation data only change periodically, so no need to download them every time.

Path("./v4").mkdir(parents=False, exist_ok=True)
napi.download_dataset("v4/train.parquet")
napi.download_dataset("v4/validation.parquet")
napi.download_dataset("v4/live.parquet", f"v4/live_{current_round}.parquet")
napi.download_dataset("v4/validation_example_preds.parquet")
napi.download_dataset("v4/features.json")

with open("v4/features.json", 'r') as f: 
    feature_metadata = json.load(f)
features = feature_metadata["feature_sets"]["small"]
read_columns = features + [ERA_COL, DATA_TYPE_COL, TARGET_COL]

numerai_training_data = pd.read_parquet('v4/train.parquet', columns=read_columns)
numerai_tournament_data = pd.read_parquet(f'v4/live_{current_round}.parquet', columns= read_columns)
numerai_validation_data = pd.read_parquet('v4/validation.parquet', columns=read_columns)

print(numerai_training_data)

eras_train = numerai_training_data.loc[:,'era']
eras_valid = numerai_validation_data.loc[:,'era']

print('transforming data')
X_train = numerai_training_data.loc[:,'feature_intelligence1':'feature_wisdom46'].values
X_valid = numerai_validation_data.loc[:,'feature_intelligence1':'feature_wisdom46'].values


Y_train = numerai_training_data.loc[:,'target_kazutsugi'].values
Y_valid = numerai_validation_data.loc[:,'target_kazutsugi'].values


pca = PCA(svd_solver='full')
pca.fit(X_train)

X_train_pca = pca.transform(X_train)
X_valid_pca = pca.transform(X_valid)


models = list()
p_train = list()
p_valid = list()

num_models = 5

for i in range(num_models):

    X = np.c_[X_train_pca[:,i],X_train_pca[:,i+num_models],X_train_pca[:,i+num_models*2],X_train_pca[:,i+num_models*3],X_train_pca[:,i+num_models*4]]
    models.append(LinearRegression().fit(X,Y_train))
    p_train.append(models[i].predict(X))
    p_valid.append(models[i].predict(np.c_[X_valid_pca[:,i],X_valid_pca[:,i+num_models],X_valid_pca[:,i+num_models*2],X_valid_pca[:,i+num_models*3],X_valid_pca[:,i+num_models*4]]))
    
    spearman_by_era = get_spearman_by_era(p_train[i],Y_train,eras_train)
    spearman_by_era_valid = get_spearman_by_era(p_valid[i],Y_valid,eras_valid.reset_index(drop =True))
    print('Spearmans')
    print(stats.spearmanr(p_train[i],Y_train))
    print(stats.spearmanr(p_valid[i],Y_valid))
    print('Sharpe Ratios')
    print(get_sharpe_ratio(spearman_by_era))
    print(get_sharpe_ratio(spearman_by_era_valid))
    print('')
    
corr_train = np.corrcoef(p_train)
corr_valid = np.corrcoef(p_valid)

print('Correlation Coeficients')
print(np.corrcoef(p_train))
print(corr_train)
print(corr_valid)