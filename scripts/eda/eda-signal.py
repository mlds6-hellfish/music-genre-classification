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

import os
import IPython
import librosa
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import librosa
import librosa.display as lplot
import IPython
import sklearn.preprocessing as skp

num_feats_path = os.environ["COMPLETE_DATA_PATH"]
audio_data_path = os.environ["AUDIO_DATA_PATH"]
img_data_path = os.environ["IMG_DATA_PATH"]

# Each genre its own directory
genres_dirs = os.listdir(audio_data_path)
genres_dirs

# # Load data

feat_30_secs_df = pd.read_csv(os.path.join(num_feats_path, "features_30_sec.csv"))

feat_30_secs_df.describe()

feat_3_secs_df = pd.read_csv(os.path.join(num_feats_path, "features_3_sec.csv"))

feat_3_secs_df.describe()

# Explore if there are missing values

print(f"Columns with na values for 3secs dataset are:\
    { list(feat_3_secs_df.columns[feat_3_secs_df.isna().any()]) }")

print(f"Columns with na values for 3secs dataset are:\
    { list(feat_30_secs_df.columns[feat_30_secs_df.isna().any()]) }")

# ## Explore signals

genres_dirs[1]

jazz_dir = os.path.join(audio_data_path, 'jazz')
jazz_dir

jazz_files = os.listdir(jazz_dir)

audio1 = os.path.join(jazz_dir, jazz_files[0])

audio_data, sr = librosa.load(audio1)
audio_data, _ = librosa.effects.trim(audio_data)

# Listen the audio fragment

IPython.display.Audio(audio_data, rate=sr)

plt.figure(figsize=(12,6))
lplot.waveplot(audio_data)
plt.show()

# ## Explore numeric signal features

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
# -

# ## Correlation matrix for mean-related features

feat_3_secs_df.columns

feat_30_secs_df.columns

feat_30_secs_df.drop(['filename', 'length'], axis=1)

rm_cols = ['filename', 'length']

feat_30_secs_df.columns

rm_cols in list(feat_30_secs_df.columns)

feat_30_secs_df.columns

mean_cols = [col for col in feat_3_secs_df.columns if 'mean' in col]
mean_cols

ft_3s_corr = feat_3_secs_df[mean_cols].corr()

var_cols = [col for col in feat_3_secs_df.columns if 'var' in col]
var_cols


def plot_corr_heatmap(df, selected_cols):
    f, ax = plt.subplots(figsize=(15,10))

    df_corr = df[selected_cols].corr()
    # For hidding superior components of the matrix
    mask = np.triu(np.ones_like(df_corr, dtype=bool))
    
    cmap = sns.diverging_palette(0, 25, as_cmap=True, s=90, l=45, n=45)
    sns.heatmap(df_corr, vmax=.3, center=0, square=True, cmap=cmap, mask=mask)

    plt.title('Correlation Heatmap for mean-related features')
    plt.show()


plot_corr_heatmap(feat_3_secs_df, mean_cols)

plot_corr_heatmap(feat_3_secs_df, var_cols)

# ## Visualize BPM

tempo_3s_df = feat_3_secs_df[["label", "tempo"]]
tempo_30s_df = feat_30_secs_df[["label", "tempo"]]

# +
fig, ax = plt.subplots(figsize=(16, 8))

sns.boxplot(x='label', y = 'tempo', data=tempo_3s_df)

# +
fig, ax = plt.subplots(figsize=(16, 8))

sns.boxplot(x='label', y = 'tempo', data=tempo_30s_df)
# -

# ## Visualize (signal-based features) in 2D

data_3s = feat_3_secs_df.iloc[0:, 1:]
label_3s = data_3s['label']
X1 = data_3s.loc[:, data_3s.columns != 'label']

data_30s = feat_30_secs_df.iloc[0:, 1:]
label_30s = data_30s['label']
X2 = data_30s.loc[:, data_30s.columns != 'label']

# Standarize data

min_max_scaler = skp.MinMaxScaler()

scaled_3s = min_max_scaler.fit_transform(X1)
X1 = pd.DataFrame(scaled_3s, columns=X1.columns)

scaled_30s = min_max_scaler.fit_transform(X2)
X2 = pd.DataFrame(scaled_30s, columns=X2.columns)

X1.head()

X2.head()

from sklearn.decomposition import PCA

https://distill.pub/2019/activation-atlas/pca = PCA(n_components=2)

pca_components_x1 = pca.fit_transform(X1)
pca_x1_df = pd.DataFrame(data=pca_components_x1, columns=['pc1', 'pc2'])

pca_components_x2 = pca.fit_transform(X2)
pca_x2_df = pd.DataFrame(data=pca_components_x2, columns=['pc1', 'pc2'])

df_3s = pd.concat([pca_x1_df, label_3s], axis=1)
df_30s = pd.concat([pca_x2_df, label_30s], axis=1)

# +
# Plot 2d visualization for each duration

plt.figure(figsize=(12, 8))
sns.scatterplot(x="pc1", y="pc2", data=df_3s, hue='label')

plt.title('PCA visualization by genres (3 seconds fragments)', fontsize=18)


plt.xlabel('PC1', fontsize=15)
plt.ylabel('PC2', fontsize=15)

# +
plt.figure(figsize=(12,12))
sns.scatterplot(x="pc1", y="pc2", data=df_30s, hue='label')

plt.title('PCA visualization by genres (30 seconds fragments)', fontsize=18)

plt.xlabel('PC1', fontsize=15)
plt.ylabel('PC2', fontsize=15)

# +
# Load audio data
# -

blues_dir = os.path.join(audio_data_path, 'blues')
blues_dir

blues_files = os.listdir(blues_dir)

audio1_path = os.path.join(blues_dir, blues_files[0])

audio1_path

audio_data, sr = librosa.load(audio1_path)
# audio_data, _ = librosa.effects.trim(audio_data)

print(audio_data)
print(audio_data.shape)
print(f"Sample rate: {sr}")
print(f"Audio length: {audio_data.shape[0]/sr}")

audio_data, _ = librosa.effects.trim(audio_data)
print(audio_data)
print(audio_data.shape)

IPython.display.Audio(audio_data, rate=sr)

plt.figure(figsize=(12,6))
lplot.waveplot(audio_data)
plt.title("Sound wave blues track 76")
plt.show()

# Fourier transform

# +
n_fft = 2048 # Window size
hop_length = 512 # number audio of frames between STFT columns

D = np.abs(librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length))
D.shape
# -

plt.figure(figsize=(16,6))
plt.plot(D)

DB = librosa.amplitude_to_db(D, ref=np.max)

plt.figure(figsize=(16,6))
librosa.display.specshow(DB, sr=sr, hop_length=hop_length,
                         x_axis='time', y_axis='log')
plt.title("Spectrogram")
# plt.colorbar()

# Explore original spectrogram

blues_img_dir = os.path.join(img_data_path, "blues")

blues_imgs = os.listdir(blues_img_dir)

blues_track76_img = blues_imgs[2]
blues_track76_path = os.path.join(blues_img_dir, blues_track76_img)

# +
b76_spect = plt.imread(blues_track76_path)

plt.figure(figsize=(16,6))
plt.imshow(b76_spect)
# -

# Mel spectrogram

S = librosa.feature.melspectrogram(audio_data, sr=sr)
S_DB = librosa.amplitude_to_db(S, ref=np.max)

plt.figure(figsize=(16,6))
librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length,
                         x_axis='time', y_axis='log')
plt.title("Mel Spectrogram Blues")

S = librosa.feature.melspectrogram(audio_data, sr=sr)
S_DB = librosa.amplitude_to_db(S, ref=np.max)

# Audio features

zero_crossings = librosa.zero_crossings(audio_data, pad=False)
print(sum(zero_crossings))

plt.figure(figsize = (16, 6))
plt.plot(y_harm );
plt.plot(y_perc);

tempo, _ = librosa.beat.beat_track(audio_data, sr = sr)
tempo

# +
# Calculate the Spectral Centroids
spectral_centroids = librosa.feature.spectral_centroid(audio_data, sr=sr)[0]

# Shape is a vector
print('Centroids:', spectral_centroids, '\n')
print('Shape of Spectral Centroids:', spectral_centroids.shape, '\n')

# Computing the time variable for visualization
frames = range(len(spectral_centroids))

# Converts frame counts to time (seconds)
t = librosa.frames_to_time(frames)

print('frames:', frames, '\n')
print('t:', t)


# +
# Function that normalizes the Sound Data
def normalize(x, axis=0):
    return skp.minmax_scale(x, axis=axis)

#Plotting the Spectral Centroid along the waveform
plt.figure(figsize = (16, 6))
librosa.display.waveplot(audio_data, sr=sr, alpha=0.4);
plt.plot(t, normalize(spectral_centroids), color='orange');

# +
# Spectral RollOff Vector
spectral_rolloff = librosa.feature.spectral_rolloff(audio_data, sr=sr)[0]

# The plot
plt.figure(figsize = (16, 6))
librosa.display.waveplot(audio_data, sr=sr, alpha=0.4);
plt.plot(t, normalize(spectral_rolloff), color='orange');

# +
mfccs = librosa.feature.mfcc(audio_data, sr=sr)
print('mfccs shape:', mfccs.shape)

#Displaying  the MFCCs:
plt.figure(figsize = (16, 6))
librosa.display.specshow(mfccs, sr=sr, x_axis='time');
# -

mfccs.var()

# +
# Perform Feature Scaling
mfccs = skp.scale(mfccs, axis=1)
print('Mean:', mfccs.mean(), '\n')
print('Var:', mfccs.var())

plt.figure(figsize = (16, 6))
librosa.display.specshow(mfccs, sr=sr, x_axis='time', cmap = 'cool');

# +
# Increase or decrease hop_length to change how granular you want your data to be
hop_length = 5000

# Chromogram
chromagram = librosa.feature.chroma_stft(audio_data, sr=sr, hop_length=hop_length)
print('Chromogram shape:', chromagram.shape)

plt.figure(figsize=(16, 6))
librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length);
