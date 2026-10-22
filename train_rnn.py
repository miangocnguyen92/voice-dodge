"""
Train an RNN (LSTM) to recognize 'left' / 'right' / 'other'.
Run: python train_rnn.py
Output: kws_rnn.keras (model) + norm.npz (normalization stats); both are required by game.py.
"""
import glob
import os
import random

import librosa
import numpy as np

DATA_DIR = "speech_commands"        
# Audio settings:
#   SR = 16000      -> sample rate: 1 second of audio = 16,000 samples (16 kHz)
#   N_FFT = 1024    -> each frame analyzes a window of 1,024 samples (~64 ms)
#   HOP = 512       -> frames move forward by 512 samples, overlapping by 50%
#                      (1 second -> ~32 frames)
#   N_MFCC = 40     -> 40 MFCC features describe each frame (a tunable hyperparameter)
# Splitting audio into short frames captures how the sound changes over time,
# which gives the RNN a sequence to read step by step.
SR, N_MFCC, N_FFT, HOP = 16000, 40, 1024, 512
# 3 labels
LABELS = ["left", "right", "other"]
# Number of samples for the "other" class:
#   N_OTHER   -> recordings of non-command words (yes, no, up, down, ...)
#   N_SILENCE -> 1-second clips of background noise / silence
# "left" and "right" each have ~3,800 recordings, so 4,000 "other" words are
# randomly sampled to keep the classes roughly balanced.
#
# Trade-offs of random sampling:
#   + Prevents the "other" class from dominating training
#   - Discards most of the available non-command recordings
#
# Possible improvements:
#   1. Stratified sampling across words, so no single word is over-represented
#      in the "other" class
#   2. Predict every word individually instead of grouping them into "other"
N_OTHER, N_SILENCE = 4000, 1000        


def fix_length(y):
        #Force audio to exactly 1 second (16,000 samples): pad with zeros (silence) if shorter, trim if longer.
    return np.pad(y, (0, max(0, SR - len(y))))[:SR]


def mfcc(y):
    # Convert 1 second of audio into a sequence of shape (32 time frames, 40 MFCCs) as input for the RNN.
    m = librosa.feature.mfcc(y=fix_length(y), sr=SR, n_mfcc=N_MFCC,
                             n_fft=N_FFT, hop_length=HOP)
    # Transpose to (time, features) and convert to float32: lighter and the default type for TensorFlow
    return m.T.astype(np.float32)

# Load a .wav file and resample it to 16 kHz.
# librosa.load() returns (audio, sample_rate); [0] keeps only the audio array.
def load(path):
    return librosa.load(path, sr=SR)[0]

# Prepare the data
def build_dataset():
    X, y = [], []

    # Class 0, 1: "left", "right" 
    for i, word in enumerate(["left", "right"]):
        files = glob.glob(f"{DATA_DIR}/{word}/*.wav")
        print(f"{word}: {len(files)} file")
        if not files:
            raise SystemExit(f"Không tìm thấy file nào trong {DATA_DIR}/{word}/ — kiểm tra lại bước giải nén.")
        for f in files:
            X.append(mfcc(load(f)))
            y.append(i)

    # Class 2: Other words (yes, no, up, down...)
    others = [f for d in os.listdir(DATA_DIR)
              if os.path.isdir(f"{DATA_DIR}/{d}")
              and d not in ("left", "right", "_background_noise_")
              for f in glob.glob(f"{DATA_DIR}/{d}/*.wav")]
    for f in random.sample(others, N_OTHER):
        X.append(mfcc(load(f)))
        y.append(2)

    # Class 3: Noise, silence
    noises = [load(f) for f in glob.glob(f"{DATA_DIR}/_background_noise_/*.wav")]
    for _ in range(N_SILENCE):
        n = random.choice(noises)
        s = random.randint(0, len(n) - SR)
        X.append(mfcc(n[s:s + SR] * random.uniform(0.1, 1.0)))
        y.append(2)

    return np.stack(X), np.array(y)


if __name__ == "__main__":
    import tensorflow as tf
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import train_test_split

    # Extract features once and cache them to features.npz (reused on later runs)
    if os.path.exists("features.npz"):
        d = np.load("features.npz")
        X, y = d["X"], d["y"]
    else:
        X, y = build_dataset()
        np.savez("features.npz", X=X, y=y)
    print("Dataset:", X.shape, np.bincount(y, minlength=3))
    if (np.bincount(y, minlength=3) == 0).any():
        raise SystemExit("Có lớp không có mẫu nào — xoá features.npz và kiểm tra thư mục dữ liệu.")

    # Standardize each MFCC coefficient (z-score) and save mean/std for use in game.py
    mean, std = X.mean(axis=(0, 1)), X.std(axis=(0, 1)) + 1e-6
    np.savez("norm.npz", mean=mean, std=std)
    # standard scaler 
    X = (X - mean) / std

    # stratify=y   -> keeps the same class proportions in train and test sets
    # random_state -> fixes the split so results are reproducible across runs
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.15,
                                          stratify=y, random_state=42)

    # RNN: two stacked LSTM layers read the MFCC sequence frame by frame over time.
    # Training uses backpropagation: a forward pass computes predictions and the loss, then gradients flow backward through the network to update the weights.
    model = tf.keras.Sequential([
        tf.keras.layers.Input((X.shape[1], N_MFCC)), # 
        tf.keras.layers.LSTM(64, return_sequences=True), # gói 3 cổng mình học
        tf.keras.layers.Dropout(0.3), # trong hidden layer có nhiều nodes, có thể bị overfitting , drop 30% nodes để giảm overfitting
        tf.keras.layers.LSTM(64), # ra dc giá trị predict
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(len(LABELS), activation="softmax"),
    ])
    
    model.compile(optimizer="adam",# back progation, adam or radation descent
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"]) 
    model.summary()

    model.fit(Xtr, ytr, validation_split=0.15, epochs=30, batch_size=64,
              callbacks=[tf.keras.callbacks.EarlyStopping(
                  patience=4, restore_best_weights=True)])

    print("Test accuracy:", model.evaluate(Xte, yte, verbose=0)[1])
    print("Confusion matrix (left, right, other):")
    print(confusion_matrix(yte, model.predict(Xte).argmax(1), labels=[0, 1, 2]))

    # Save the trained model so game.py can load it
    model.save("kws_rnn.keras")
    # norm.npz stores the MFCC mean/std, not model weights; the weights are inside kws_rnn.keras
    print("Đã lưu kws_rnn.keras và norm.npz")
