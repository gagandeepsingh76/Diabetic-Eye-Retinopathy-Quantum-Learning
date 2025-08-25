"""
Improved InceptionV3 Training (CPU/GPU)
- Stronger data augmentation
- Transfer learning (2 phases: freeze + fine-tune)
- BatchNorm + Dropout in classifier head
- Early stopping + LR scheduler
- Saves metrics, confusion matrix, ROC, reports
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ===========================
# 1. Paths & Config
# ===========================
train_dir = "im1_balanced/train"
val_dir = "im1_balanced/val"
save_dir = "results_inception"
os.makedirs(save_dir, exist_ok=True)

img_size = (299, 299)   # InceptionV3 native size
batch_size = 32
epochs_stage1 = 10
epochs_stage2 = 20

# ===========================
# 2. Data Generators
# ===========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    channel_shift_range=20.0,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size,
    class_mode='binary', shuffle=True
)
val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=img_size, batch_size=batch_size,
    class_mode='binary', shuffle=False
)

# ===========================
# 3. Model (InceptionV3 + BatchNorm + Dropout)
# ===========================
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299,299,3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base model
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# ===========================
# 4. Callbacks
# ===========================
checkpoint_path = os.path.join(save_dir, "best_inception.keras")
callbacks = [
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=1),
    ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_accuracy', mode='max')
]

# ===========================
# 5. Training (Stage 1: Frozen Base)
# ===========================
history1 = model.fit(train_generator, validation_data=val_generator,
                     epochs=epochs_stage1, callbacks=callbacks)

# ===========================
# 6. Fine-tuning (Stage 2: Unfreeze deeper layers)
# ===========================
for layer in base_model.layers[-150:]:   # unfreeze last 150 layers
    if not isinstance(layer, BatchNormalization):  # keep BN stable
        layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

history2 = model.fit(train_generator, validation_data=val_generator,
                     epochs=epochs_stage2, callbacks=callbacks)

# Merge histories
history = {}
for k in history1.history.keys():
    history[k] = history1.history[k] + history2.history[k]

# ===========================
# 7. Save Training Curves + JSON + CSV
# ===========================
# Accuracy plot
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history['accuracy'], label='Train Acc')
plt.plot(history['val_accuracy'], label='Val Acc')
plt.legend(); plt.title("Accuracy")
plt.savefig(os.path.join(save_dir,"training_curves_accuracy.png"))

# Loss plot
plt.subplot(1,2,2)
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.legend(); plt.title("Loss")
plt.savefig(os.path.join(save_dir,"training_curves_loss.png"))
plt.close()

# Convert numpy/float32 to Python float for JSON
history_clean = {k: [float(x) for x in v] for k, v in history.items()}

# Save JSON + CSV
with open(os.path.join(save_dir,"training_metrics.json"), "w") as f:
    json.dump(history_clean, f, indent=4)
pd.DataFrame(history_clean).to_csv(os.path.join(save_dir,"training_metrics.csv"), index=False)

# ===========================
# 8. Confusion Matrix + Report
# ===========================
val_preds = model.predict(val_generator)
val_preds_class = (val_preds > 0.5).astype(int).ravel()

cm = confusion_matrix(val_generator.classes, val_preds_class)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Abnormal','Normal'], yticklabels=['Abnormal','Normal'])
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix")
plt.savefig(os.path.join(save_dir,"confusion_matrix.png"))
plt.close()

# Save numpy + json
np.save(os.path.join(save_dir,"confusion_matrix.npy"), cm.tolist())
with open(os.path.join(save_dir,"confusion_matrix.json"), "w") as f:
    json.dump(cm.tolist(), f)

# Classification report
report = classification_report(val_generator.classes, val_preds_class,
                               target_names=['Abnormal','Normal'], output_dict=True)
with open(os.path.join(save_dir,"classification_report.txt"), "w") as f:
    f.write(classification_report(val_generator.classes, val_preds_class, target_names=['Abnormal','Normal']))
with open(os.path.join(save_dir,"classification_report.json"), "w") as f:
    json.dump(report, f, indent=4)

# ===========================
# 9. ROC Curve
# ===========================
fpr, tpr, _ = roc_curve(val_generator.classes, val_preds)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0,1],[0,1],'--', color='gray')
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve"); plt.legend()
plt.savefig(os.path.join(save_dir,"roc_curve.png"))
plt.close()

roc_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(roc_auc)}
with open(os.path.join(save_dir,"roc_curve_data.json"), "w") as f:
    json.dump(roc_data, f, indent=4)

# ===========================
# 10. Save Final Model
# ===========================
model.save(os.path.join(save_dir,"inception_trained.keras"))
