# 🎮 Voice Dodge — Voice-Controlled Game with an RNN Keyword Spotter

A small arcade game where the player dodges falling rocks by saying **"left"** or **"right"**.
The commands are recognized in real time by a Recurrent Neural Network (RNN) trained on the Google Speech Commands dataset.

![Demo](assets/demo.gif)
<!-- Record a short GIF of the game (e.g. with ScreenToGif) and save it as assets/demo.gif -->

---

## 📌 Project Overview

| | |
|---|---|
| **Goal** | Control a game hands-free using spoken keywords |
| **Task** | Keyword spotting (audio classification): `left`, `right` |
| **Model** | RNN (Keras / TensorFlow) |
| **Data** | Google Speech Commands |
| **Skills shown** | Audio feature extraction, sequence modeling, model deployment in an interactive app |

---

## 🗂️ Project Structure

```
voice-dodge/
├── README.md
├── requirements.txt
├── .gitignore
├── train_rnn.py      # Feature extraction + RNN training
├── game.py           # Game loop + real-time voice control
├── kws_rnn.keras     # Trained model
└── norm.npz          # Feature normalization stats (mean/std) used at inference
```

> The dataset (`speech_commands/`) and the extracted features (`features.npz`) are **not** included in this repo because of their size. See below for how to recreate them.

---

## 📥 Dataset

This project uses the **Google Speech Commands** dataset (Warden, 2018).

1. Download it here: http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
   (catalog page: https://www.tensorflow.org/datasets/catalog/speech_commands)
2. Extract it into a folder named `speech_commands/` in the project root.

---

## 🧠 Method

1. **Feature extraction** — Each 1-second audio clip is converted into a sequence of audio features (e.g. MFCCs).
2. **Normalization** — Features are standardized; the mean/std are saved to `norm.npz` so the game applies the exact same transformation at inference time.
3. **Model** — An RNN reads the feature sequence frame by frame and predicts the keyword.
4. **Deployment** — `game.py` records audio from the microphone, extracts features, runs the model, and moves the player.

<!-- Fill in your real architecture, e.g.:
Input (time_steps, n_mfcc) → LSTM(64) → Dropout(0.3) → Dense(32, relu) → Dense(n_classes, softmax) -->

---

## 📊 Results

| Metric | Value |
|---|---|
| Validation accuracy | `XX%` |
| Test accuracy | `XX%` |

<!-- Optional: add a confusion matrix image, e.g. assets/confusion_matrix.png -->

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/miangocnguyen92/voice-dodge.git
cd voice-dodge

# 2. Create a virtual environment and install dependencies
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt

# 3. Play the game (uses the pre-trained model)
python game.py

# 4. (Optional) Retrain the model — requires the dataset above
python train_rnn.py
```

A working microphone is required to play.

---

## ⚠️ Limitations & Future Work

- Only two keywords are supported; could be extended to `up`, `down`, `stop`, etc.
- Performance may drop with background noise or unfamiliar accents → add noise augmentation.
- Compare the RNN with a lightweight CNN or a pretrained audio model.

---

## 👤 Author

**Mia Nguyen** — [LinkedIn](https://www.linkedin.com/in/mianguyen92/) · [GitHub](https://github.com/miangocnguyen92)
