# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import matplotlib.pyplot as plt
import pandas as pd
import os

num_feats_path = os.path.join("../../", os.environ["COMPLETE_DATA_PATH"])

# # Load data

feat_30_secs_df = pd.read_csv(os.path.join(num_feats_path, "features_30_sec.csv"))

feat_30_secs_df.describe()

feat_3_secs_df = pd.read_csv(os.path.join(num_feats_path, "features_3_sec.csv"))

feat_3_secs_df.describe().loc[['mean', 'std']]

# Plot correlation between features

# +
df = feat_3_secs_df.copy()

f = plt.figure(figsize=(19, 15))
plt.matshow(df.corr(), fignum=f.number)
plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14, rotation=45)
plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.xticks(rotation=90)
plt.title('Correlation Matrix', fontsize=16);
plt.show()

# +
df = feat_30_secs_df.copy()

f = plt.figure(figsize=(19, 15))
plt.matshow(df.corr(), fignum=f.number)
plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14, rotation=45)
plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.xticks(rotation=90)
plt.title('Correlation Matrix', fontsize=16);
plt.show()
