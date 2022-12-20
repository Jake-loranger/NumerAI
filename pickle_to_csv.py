import pickle as pkl
import pandas as pd

with open("model_target.pkl", "rb") as f:
    object = pkl.load(f)

df = pd.DataFrame(object)
df.to_csv(r'model_target.csv')

