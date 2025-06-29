
# 🎙️ Speaker Recognition Using Classical Machine Learning Techniques

## 🧠 Overview

This project explores **speaker recognition** using **classical machine learning algorithms** on a dataset of speeches by globally recognized leaders. We focus on both **text-dependent** and **text-independent** speaker recognition using models like **GMM**, **Random Forest**, and **kNN**, achieving up to **98% accuracy**.

---

## 📁 Dataset

* **Speakers:** Benjamin Netanyahu, Jens Stoltenberg, Julia Gillard, Margaret Thatcher, Nelson Mandela
* **Format:** 1-second PCM encoded audio clips at 16,000 Hz
* **Noise:** Includes a "background\_noise" class to simulate real-world environments

---

## 🔍 Exploratory Data Analysis

* Audio normalization & framing
* Spectrogram and waveform visualization
* Frequency distribution and statistical summaries
* Background noise characterization

---

## 🛠️ Preprocessing Steps

* **Noise Injection** (using real-world background clips)
* **Spectral Filtering & Gating** to reduce ambient noise
* **Vocal Enhancement** using MFCC spread analysis
* **Trimming & Normalization**
* **Data Augmentation**: Pitch shift, time-stretching

---

## 🎯 Feature Engineering

* **MFCCs + Delta + Delta-Delta**
* **Mel Spectrogram**
* **Zero-Crossing Rate**
* **RMS Energy**
* **LPCC (Linear Prediction Cepstral Coefficients)**
* **Spectral Features** (centroid, contrast, bandwidth)

---

## 🤖 Models and Results

| Model               | Accuracy | R² Score | MSE       |
| ------------------- | -------- | -------- | --------- |
| GMM                 | 96%      | 0.95     | 0.092     |
| Random Forest       | **98%**  | **0.98** | **0.025** |
| k-Nearest Neighbors | 86%      | 0.76     | 0.471     |

> GMM was effective in probabilistic modeling, Random Forests excelled in pattern capture, and kNN provided a simpler baseline.

