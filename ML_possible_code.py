import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Setting up the directories
data_dir = "./"
background_noise_dir = os.path.join(data_dir, "_background_noise_")
speaker_folders = ["Benjamin Netanyahu", "Jens Stoltenberg", "Julia Gillard", "Margaret Tacher", "Nelson Mandela"]
speaker_paths = [os.path.join(data_dir, speaker) for speaker in speaker_folders]

def extract_features(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

features = []
labels = []

# Extracting features from the audio files
for speaker, spath in zip(speaker_folders, speaker_paths):
    for filename in os.listdir(spath):
        if filename.endswith(".wav"):
            data = extract_features(os.path.join(spath, filename))
            features.append(data)
            labels.append(speaker)

features_df = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(features[0].shape[0])])
features_df['label'] = labels

# Adding noise
noise_files = [os.path.join(background_noise_dir, file) for file in os.listdir(background_noise_dir) if file.endswith('.wav')]
for i in range(len(features_df)):
    noise = np.random.choice(noise_files)
    y_noise, sr_noise = librosa.load(noise, duration=1.0)
    y, sr = librosa.load(os.path.join(speaker_paths[features_df['label'][i]], str(i) + ".wav"), duration=1.0)
    y += y_noise
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    features_df.iloc[i, :-1] = np.mean(mfcc.T, axis=0)

# Label Encoding
label_encoder = LabelEncoder()
features_df['label'] = label_encoder.fit_transform(features_df['label'])

# Splitting the dataset
X = features_df.drop(columns=['label'])
y = features_df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)