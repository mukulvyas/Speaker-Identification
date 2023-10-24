import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Setting up the directories
root_dir = "../Speaker-Identification"
data_dir = os.path.join(root_dir, "16000_pcm_speeches")
background_noise_dir = "16000_pcm_speeches//_background_noise_"
speaker_folders = ["Benjamin_Netanyau", "Jens_Stoltenberg", "Julia_Gillard", "Margaret_Tarcher", "Nelson_Mandela"]
speaker_paths = ["16000_pcm_speeches//Benjamin_Netanyau", "16000_pcm_speeches//Jens_Stoltenberg", "16000_pcm_speeches//Julia_Gillard", "16000_pcm_speeches//Magaret_Tarcher", "16000_pcm_speeches//Nelson_Mandela"]

def extract_features(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

features = []
labels = []

# Extracting features from the audio files
for i in range(len(speaker_paths)):
    for file in os.listdir(speaker_paths[i]):
        if file.endswith(".wav"):
            file_name = os.path.join(speaker_paths[i], file)
            class_label = speaker_folders[i]
            data = extract_features(file_name)
            features.append(data)
            labels.append(class_label)

# Ensure there are extracted features before proceeding
if not features:
    raise ValueError("No features were extracted. Check the directories and data.")

features_df = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(features[0].shape[0])])
features_df['label'] = labels

# Corrected code
for label in labels:
    features_df['filename'] = [os.path.join(data_dir, label, file) for file in os.listdir(os.path.join(data_dir, label))]
# Adding noise
noise_files = [os.path.join(background_noise_dir, file) for file in os.listdir(background_noise_dir) if file.endswith('.wav')]
for i in range(len(features_df)):
    noise = np.random.choice(noise_files)
    y_noise, sr_noise = librosa.load(noise, duration=1.0)
    y, sr = librosa.load(os.path.join(data_dir, features_df['label'][i], str(i) + ".wav"), duration=1.0)
    y += y_noise
    file_name = features_df['filename'][i]
    y, sr = librosa.load(file_name, duration=1.0)
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