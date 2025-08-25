# 🧠 Diabetic Eye Retinopathy - Quantum Learning Approach

This repository implements an improved InceptionV3-based deep learning model for Diabetic Eye Retinopathy classification, enhanced with advanced training techniques such as transfer learning, data augmentation, early stopping, learning rate scheduling, and fine-tuning.

The model saves metrics, confusion matrix, ROC curves, and reports for better evaluation.

## 📌 Key Features
Transfer Learning with InceptionV3 (ImageNet weights)
Two-stage training (frozen + fine-tuning)
Data Augmentation (rotation, shift, zoom, flips)
Early Stopping & Learning Rate Scheduler
Confusion Matrix, Classification Reports, and ROC Curve
Saves metrics in JSON, CSV, and visual plots
Works on CPU/GPU

## 📂 Project Structure
Diabetic-Eye-Retinopathy-Quantum-Learning/
│
- ├── im1_balanced/ # Dataset (train/val split)
- ├── results_inception/ # Training results (saved models, plots, reports)
- │ ├── best_inception_cpu.keras
│ ├── inception_trained_cpu.keras
│ ├── training_curves_accuracy.png
│ ├── training_curves_loss.png
│ ├── confusion_matrix.png
│ ├── classification_report.txt
│ ├── roc_curve.png
│ ├── training_metrics.json / .csv
│ └── roc_curve_data.json
│
├── main.py # Training script (InceptionV3)
└── README.md # Documentation

##🛠️ Tech Stack
Python
TensorFlow / Keras
NumPy, Pandas
Matplotlib, Seaborn
Scikit-learn

## 🚀 Getting Started
1. Clone the repository

git clone https://github.com/gagandeepsingh76/Diabetic-Eye-Retinopathy-Quantum-Learning.git

cd Diabetic-Eye-Retinopathy-Quantum-Learning

2. Install dependencies

pip install -r requirements.txt

3. Prepare dataset

Place your balanced dataset in im1_balanced/train and im1_balanced/val with subfolders:

im1_balanced/
│
├── train/
│ ├── Normal/
│ └── Abnormal/
│
└── val/
├── Normal/
└── Abnormal/

4. Train the model

python main.py

5. Results

Saved in results_inception/

Includes best model, training metrics, confusion matrix, ROC, and reports.

📊 Evaluation

Accuracy & Loss curves

Confusion Matrix (Normal vs Abnormal)

ROC Curve with AUC score

Classification Report (Precision, Recall, F1)

⭐ If you found this project useful, please consider giving it a star!

