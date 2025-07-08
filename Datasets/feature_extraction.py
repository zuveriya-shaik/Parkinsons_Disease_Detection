import os
import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
from scipy.signal import welch
from sklearn.impute import SimpleImputer
from scipy.stats import entropy
import nolds

# Function definitions remain the same — spread, d2, rpde, dfa, ppe
def calculate_spread(snd):
    frequency, power = welch(snd.values, fs=snd.xmax, nperseg=1024)
    fo_mean = np.mean(frequency)
    spread1 = np.sum((frequency - fo_mean)**2 * power) / np.sum(power)
    spread2 = np.sum((frequency - fo_mean)**4 * power) / np.sum(power)
    return spread1, spread2

def calculate_d2(signal):
    if len(signal) < 2:
        return 2.382
    N = len(signal)
    distance_matrix = np.abs(signal[:, None] - signal)
    epsilon = np.mean(distance_matrix)
    count = np.sum(distance_matrix < epsilon, axis=1)
    return np.log(count / N) / np.log(N)

def calculate_rpde(signal):
    if len(signal) < 2:
        return 0.4985
    N = len(signal)
    distance_matrix = np.abs(signal[:, None] - signal)
    epsilon = np.mean(distance_matrix)
    recurrence = (distance_matrix < epsilon).astype(int)
    rpde = -np.sum(recurrence * np.log(recurrence + 1e-9)) / (np.sum(recurrence) + 1e-9)
    return rpde

def calculate_dfa(signal):
    if len(signal) < 2:
        return 0.7181
    return nolds.dfa(signal)

def calculate_ppe(pitch_values):
    pitch_periods = np.diff(pitch_values)
    pitch_periods = pitch_periods[pitch_periods > 0]
    return entropy(pitch_periods) if len(pitch_periods) > 0 else 0

# Extract features from a single file
def extract_praat_features(file_path):
    try:
        snd = parselmouth.Sound(file_path)
        pitch = snd.to_pitch()
        fo_mean = call(pitch, "Get mean", 0, 0, "Hertz")
        fo_max = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
        fo_min = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")

        point_process = call(snd, "To PointProcess (periodic, cc)", 75, 600)

        jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        jitter_absolute = call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
        rap = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        ppq = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
        ddp = 3 * rap

        shimmer_local = call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_db = call([snd, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_apq3 = call([snd, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_apq5 = call([snd, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_apq = call([snd, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_dda = 3 * shimmer_apq3

        harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75.0, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)
        nhr = 1 / hnr if hnr != 0 else None

        rpde = calculate_rpde(snd.values)
        dfa = calculate_dfa(snd.values)
        spread1, spread2 = calculate_spread(snd)
        d2 = calculate_d2(snd.values)
        ppe = calculate_ppe(pitch.selected_array['frequency'])

        return {
            "MDVP:Fo(Hz)": fo_mean, "MDVP:Fhi(Hz)": fo_max, "MDVP:Flo(Hz)": fo_min,
            "MDVP:Jitter(%)": jitter_local, "MDVP:Jitter(Abs)": jitter_absolute, "MDVP:RAP": rap,
            "MDVP:PPQ": ppq, "Jitter:DDP": ddp, "MDVP:Shimmer": shimmer_local,
            "MDVP:Shimmer(dB)": shimmer_db, "Shimmer:APQ3": shimmer_apq3,
            "Shimmer:APQ5": shimmer_apq5, "MDVP:APQ": shimmer_apq,
            "Shimmer:DDA": shimmer_dda, "NHR": nhr, "HNR": hnr,
            "RPDE": rpde, "DFA": dfa, "spread1": spread1,
            "spread2": spread2, "D2": d2, "PPE": ppe
        }

    except Exception as e:
        print(f"Error in file {file_path}: {e}")
        return None

# Define folders
folders = [
    ("C:\\Users\\pooji\\OneDrive\\Documents\\poojitha\\Major Project Btech\\Major Project Streamlit\\PD_AH\\PD_AH", 1),
    ("C:\\Users\\pooji\\OneDrive\\Documents\\poojitha\\Major Project Btech\\Major Project Streamlit\\HC_AH\\HC_AH", 0)
]



all_data = []

for folder_path, label in folders:
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".wav"):
            file_path = os.path.join(folder_path, file_name)
            print(f"Processing: {file_name}")
            features = extract_praat_features(file_path)

            if features is not None:
                features['File Name'] = file_name
                features['status'] = label
                all_data.append(features)

# Create DataFrame and handle missing values
df = pd.DataFrame(all_data)
numeric_cols = df.select_dtypes(include=[np.number]).columns
imputer = SimpleImputer(strategy='mean')
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

# Save the final CSV
df.to_csv("combined_audio_features.csv", index=False)
print("Saved all extracted features to 'combined_audio_features.csv'")
