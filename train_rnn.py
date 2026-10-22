"""
Train RNN (LSTM) nhận diện 'left' / 'right' / 'other'.
Chạy: python train_rnn.py
Kết quả: kws_rnn.keras (model) + norm.npz (thông số chuẩn hoá) — game.py cần 2 file này.
"""
import glob
import os
import random

import librosa
import numpy as np

DATA_DIR = "speech_commands"        
# SR = 16000, mỗi giây âm thanh chia thành 16000 số, gọi là Hz
# MFCC: cắt đoạn âm thanh thành từng khung nhỏ MFCC 
# ý nghĩa của chia ra là gì?
# mỗi khung thời gian có 40 number of features. User defined (dựa vào số frame để định nghĩa)
SR, N_MFCC, N_FFT, HOP = 16000, 40, 1024, 512
# 3 labels
LABELS = ["left", "right", "other"]
# số mẫu "từ khác" và "im lặng/tiếng ồn"
# sample left and right khoảng 3800 => chọn random = 4000 để training k bị imbalance
# noise (N_SILENCE)
# Cons của sampling: bỏ đi rất nhiều từ sound như left and right
# Pros: random sampling 4000 để tránh imbalance
# C1: nên làm stratified sampling để chia đểu việc chọn sample cho các mẫu thử , tránh TH chọn nhiều sample cho cùng 1 từ
# C2: predict từng từ luôn (thay vì chỉ predict left and right or other)
N_OTHER, N_SILENCE = 4000, 1000        


# tại sao cắt đúng 1 giây?
def fix_length(y):
    """Cắt đúng 1 giây âm thanh, nấu ngắn hơn thì thêm 0 là im lặng, dài hơn 1 giây thì cắt bớt."""
    """Đưa array về đúng 16000 số"""
    return np.pad(y, (0, max(0, SR - len(y))))[:SR]


def mfcc(y):
    """1 giây audio -> chuỗi (32 frame thời gian, 40 MFCC) làm input cho RNN."""
    m = librosa.feature.mfcc(y=fix_length(y), sr=SR, n_mfcc=N_MFCC,
                             n_fft=N_FFT, hop_length=HOP)
    # đỗi sang 32 bit, nhẹ hơn cho Tensorflow chạy
    return m.T.astype(np.float32)

# Đọc file wav và chuyển về 16k Hz 
# trả về [0]: chỉ lấy phần có audio (library librosa trả lại 2 biến audio + sr hz)
def load(path):
    return librosa.load(path, sr=SR)[0]

# chuẩn bị data
def build_dataset():
    X, y = [], []

    # Lớp 0, 1: "left", "right" 
    for i, word in enumerate(["left", "right"]):
        files = glob.glob(f"{DATA_DIR}/{word}/*.wav")
        print(f"{word}: {len(files)} file")
        if not files:
            raise SystemExit(f"Không tìm thấy file nào trong {DATA_DIR}/{word}/ — kiểm tra lại bước giải nén.")
        for f in files:
            X.append(mfcc(load(f)))
            y.append(i)

    # Lớp 2: các từ khác (yes, no, up, down...)
    others = [f for d in os.listdir(DATA_DIR)
              if os.path.isdir(f"{DATA_DIR}/{d}")
              and d not in ("left", "right", "_background_noise_")
              for f in glob.glob(f"{DATA_DIR}/{d}/*.wav")]
    for f in random.sample(others, N_OTHER):
        X.append(mfcc(load(f)))
        y.append(2)

    # Lớp 3: tiếng ồn nền / im lặng
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

    # Trích xuất feature 1 lần rồi cache lại
    if os.path.exists("features.npz"):
        d = np.load("features.npz")
        X, y = d["X"], d["y"]
    else:
        X, y = build_dataset()
        np.savez("features.npz", X=X, y=y)
    print("Dataset:", X.shape, np.bincount(y, minlength=3))
    if (np.bincount(y, minlength=3) == 0).any():
        raise SystemExit("Có lớp không có mẫu nào — xoá features.npz và kiểm tra thư mục dữ liệu.")

    # Chuẩn hoá theo từng hệ số MFCC
    mean, std = X.mean(axis=(0, 1)), X.std(axis=(0, 1)) + 1e-6
    np.savez("norm.npz", mean=mean, std=std)
    # standard scaler 
    X = (X - mean) / std

    # stratify y => theo class label để tránh bias từng label
    # random state: giữ nguyên performance cho mỗi lần train lại model (nên dùng trong đi làm)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.15,
                                          stratify=y, random_state=42)

    # RNN: 2 lớp LSTM đọc chuỗi MFCC theo thời gian
    # trời ơi ko hiểu gì hết???
    # backprogation là chạy hết model, rồi quay ngược lại
    model = tf.keras.Sequential([
        tf.keras.layers.Input((X.shape[1], N_MFCC)), # 
        tf.keras.layers.LSTM(64, return_sequences=True), # gói 3 cổng mình học
        tf.keras.layers.Dropout(0.3), # trong hidden layer có nhiều nodes, có thể bị overfitting , drop 30% nodes để giảm overfitting
        tf.keras.layers.LSTM(64), #ra dc giá trị predict
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(len(LABELS), activation="softmax"),
    ])
    # backprogation chạy và tính ngược lại bằng gì?
    model.compile(optimizer="adam",#back progation, adam or radation descent
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"]) # định nghĩa metric trả về
    model.summary()

    model.fit(Xtr, ytr, validation_split=0.15, epochs=30, batch_size=64,
              callbacks=[tf.keras.callbacks.EarlyStopping(
                  patience=4, restore_best_weights=True)])

    print("Test accuracy:", model.evaluate(Xte, yte, verbose=0)[1])
    print("Confusion matrix (left, right, other):")
    print(confusion_matrix(yte, model.predict(Xte).argmax(1), labels=[0, 1, 2]))

    # lưu model vào 1 file để load lên gamepy
    model.save("kws_rnn.keras")
    # norm = weight của từng node
    print("Đã lưu kws_rnn.keras và norm.npz")
