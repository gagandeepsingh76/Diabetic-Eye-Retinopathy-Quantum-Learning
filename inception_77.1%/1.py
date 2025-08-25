#!/usr/bin/env python3
"""
CPU-only version of:
Quantum Transfer Learning using InceptionV3 (feature extractor) + optional PennyLane quantum layer.

This script forces TensorFlow to use CPU only (no GPUs), disables mixed precision, and preserves
the remainder of your pipeline (dataset loading, MixUp, model, optional hyperparameter search, eval).
"""

import os
import math
import numpy as np
import itertools
import json
import traceback
import sys

# -------------------- FORCE CPU --------------------
# Must set before importing tensorflow to ensure no GPU devices are used.
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # Force TF to ignore GPUs
# Optionally set TF_CPP_MIN_LOG_LEVEL to reduce TF logging (0 = all, 1 = filter INFO, 2 = WARN, 3 = ERROR)
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# --------------------------------------------------

import tensorflow as tf
from tensorflow.keras import layers, models

# Extra safety: if GPUs are still visible, hide them explicitly (no-op on CPU-only)
try:
    physical_gpus = tf.config.list_physical_devices('GPU')
    if physical_gpus:
        try:
            tf.config.set_visible_devices([], 'GPU')
        except Exception:
            # Some TF versions may raise; continue anyway
            pass
except Exception:
    pass

# Optional packages
try:
    import optuna
    OPTUNA_AVAILABLE = True
except Exception:
    OPTUNA_AVAILABLE = False

try:
    import keras_tuner as kt
    KERASTUNER_AVAILABLE = True
except Exception:
    KERASTUNER_AVAILABLE = False

# PennyLane for quantum layers (optional)
try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except Exception:
    PENNYLANE_AVAILABLE = False

# matplotlib is optional. If missing, we'll fall back to saving numeric output.
try:
    import matplotlib
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except Exception:
    PLOTTING_AVAILABLE = False
    plt = None

from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

# --------------------------- User settings ---------------------------
DATA_DIR = "im1"
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
IMG_SIZE = (299, 299)
BATCH_SIZE = 32  # CPU-friendly default
AUTOTUNE = tf.data.AUTOTUNE
SEED = 123
NUM_EPOCHS = 1
INITIAL_EPOCHS = 1            # frozen-backbone training only
DEFAULT_N_QUBITS = 4
DEFAULT_Q_LAYERS = 2
USE_QUANTUM = False               # keep False by default; enable only for experiments
CACHE_DATASETS = True
MIXUP_ALPHA = 0.15                # MixUp intensity (0 disables MixUp)
LABEL_SMOOTHING = 0.08            # label smoothing for categorical crossentropy
WEIGHT_DECAY = 1e-4               # AdamW weight decay
DENSE_AFTER_Q = 512               # larger head
DROPOUT_RATE = 0.5                # stronger dropout
EARLYSTOP_PATIENCE = 10
EXIT_AFTER_TRAIN = False         # run evaluation to generate confusion matrix
THRESHOLD_ACC = 0.85             # stop early when val_accuracy reaches this

# Hyperparameter search settings
RUN_HP_SEARCH = False
HP_TRIALS = 12
HP_EPOCHS = 6
HP_USE_OPTUNA = True

# --------------------------- CPU / Strategy info ---------------------------
# Use default strategy on CPU. We still call get_strategy() for compatibility.
strategy = tf.distribute.get_strategy()
print("Forcing CPU-only mode. TensorFlow devices visible:", tf.config.get_visible_devices())
print(f"Using batch size = {BATCH_SIZE}")
# Note: mixed precision is intentionally NOT enabled for CPU training.

# --------------------------- Utilities ---------------------------

def compute_class_counts(train_dir):
    classes = []
    counts = {}
    total = 0
    if not os.path.isdir(train_dir):
        return classes, counts, total
    for d in sorted(os.listdir(train_dir)):
        cls_path = os.path.join(train_dir, d)
        if os.path.isdir(cls_path):
            cnt = len([f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))])
            classes.append(d)
            counts[d] = cnt
            total += cnt
    return classes, counts, total


def compute_class_weight_from_counts(counts):
    classes = sorted(counts.keys())
    total = sum(counts.values())
    num_classes = len(classes) if len(classes) > 0 else 1
    class_weight = {}
    for i, cls in enumerate(classes):
        cls_count = counts[cls]
        if cls_count == 0:
            class_weight[i] = 1.0
        else:
            class_weight[i] = total / (num_classes * cls_count)
    return class_weight


# TF-native MixUp (batch-wise)
@tf.function
def mixup_batch(images, labels, alpha=MIXUP_ALPHA):
    if alpha <= 0.0:
        return images, labels
    batch_size = tf.shape(images)[0]
    lam = tf.random.uniform([], 0.0, 1.0)
    lam = tf.maximum(lam, 1.0 - lam)
    lam = tf.cast(lam, images.dtype)
    idx = tf.random.shuffle(tf.range(batch_size))
    mixed_images = lam * images + (1.0 - lam) * tf.gather(images, idx)
    mixed_labels = lam * labels + (1.0 - lam) * tf.gather(labels, idx)
    return mixed_images, mixed_labels


def categorical_focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        eps = 1e-7
        y_pred = tf.clip_by_value(y_pred, eps, 1. - eps)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_sum(loss, axis=1)
    return loss


# Learning rate schedule: linear warmup + cosine decay
class WarmUpCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr: float, total_steps: int, warmup_steps: int = 0, min_lr: float = 0.0, name: str = None):
        super().__init__()
        self.base_lr = float(base_lr)
        self.total_steps = int(max(1, total_steps))
        self.warmup_steps = int(max(0, warmup_steps))
        self.min_lr = float(min_lr)
        self.name = name or "WarmUpCosine"

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        total = tf.cast(self.total_steps, tf.float32)
        warm = tf.cast(self.warmup_steps, tf.float32)
        base = tf.constant(self.base_lr, tf.float32)
        min_lr = tf.constant(self.min_lr, tf.float32)

        def lr_warmup():
            # Linear warmup from min_lr to base over warmup steps
            slope = (base - min_lr) / tf.maximum(1.0, warm)
            return min_lr + slope * step

        def lr_cosine():
            # Cosine decay from base to min_lr over (total - warmup) steps
            progress = (step - warm) / tf.maximum(1.0, (total - warm))
            cosine_decay = 0.5 * (1.0 + tf.cos(tf.constant(math.pi) * tf.clip_by_value(progress, 0.0, 1.0)))
            return min_lr + (base - min_lr) * cosine_decay

        return tf.cond(step < warm, lr_warmup, lr_cosine)

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr": self.min_lr,
            "name": self.name,
        }


# Callback: stop when a metric crosses a threshold
class StopOnMetricThreshold(tf.keras.callbacks.Callback):
    def __init__(self, monitor: str = 'val_accuracy', threshold: float = 0.85):
        super().__init__()
        self.monitor = monitor
        self.threshold = float(threshold)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        value = logs.get(self.monitor)
        if value is not None and value >= self.threshold:
            print(f"Threshold reached: {self.monitor}={value:.4f} >= {self.threshold:.4f}. Stopping training.")
            self.model.stop_training = True

# --------------------------- Load datasets ---------------------------
print(f"Loading training data from: {TRAIN_DIR}")
print(f"Loading validation data from: {VAL_DIR}")

if not os.path.isdir(TRAIN_DIR) or not os.path.isdir(VAL_DIR):
    print('Error: Train or Val directory not found. Please ensure the `im1/train` and `im1/val` directories exist and contain class subfolders.')
    sys.exit(1)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    VAL_DIR,
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=False
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)
print("Detected classes:", class_names)

# compute counts & class weights
classes_from_fs, counts, total = compute_class_counts(TRAIN_DIR)
print('Training image counts per class (filesystem):', counts)
class_weight = compute_class_weight_from_counts(counts)
print('Computed class weights:', class_weight)

imbalance_ratio = 1.0
if counts:
    min_count = min([v for v in counts.values() if v > 0])
    if min_count > 0:
        imbalance_ratio = max(counts.values()) / min_count
use_focal = imbalance_ratio > 2.0
if use_focal:
    print(f"Detected strong class imbalance (ratio={imbalance_ratio:.2f}) — using focal loss")

# Cache + prefetch
if CACHE_DATASETS:
    try:
        train_ds = train_ds.cache()
        val_ds = val_ds.cache()
        print('Datasets cached in memory (if there is available RAM)')
    except Exception as e:
        print('Could not cache datasets:', e)

train_ds = train_ds.map(lambda x, y: (tf.image.resize(x, IMG_SIZE), y), num_parallel_calls=AUTOTUNE)
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (tf.image.resize(x, IMG_SIZE), y), num_parallel_calls=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

if MIXUP_ALPHA > 0.0:
    try:
        train_ds = train_ds.map(lambda x, y: mixup_batch(x, y, MIXUP_ALPHA), num_parallel_calls=AUTOTUNE)
        print('Applied MixUp augmentation to training dataset (TF-native)')
    except Exception as e:
        print('Could not apply MixUp via TF map — continuing without MixUp:', e)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.12),
    layers.RandomZoom(0.12),
    layers.RandomContrast(0.12),
])

# --------------------------- Quantum components (optional) ---------------------------
if USE_QUANTUM and not PENNYLANE_AVAILABLE:
    print("PennyLane not installed — quantum layer will be disabled automatically.")
    USE_QUANTUM = False

# --------------------------- Model builder ---------------------------

def build_feature_extractor():
    base_model = tf.keras.applications.InceptionV3(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        pooling='avg'
    )
    return base_model


def build_model(base_model, n_qubits=DEFAULT_N_QUBITS, q_layers=DEFAULT_Q_LAYERS, lr=1e-4, dense_after_q=DENSE_AFTER_Q, use_quantum=USE_QUANTUM, dropout_rate=DROPOUT_RATE, label_smoothing=LABEL_SMOOTHING, use_focal_loss=False):
    base_model.trainable = False
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.inception_v3.preprocess_input(x)
    x = base_model(x, training=False)

    if use_quantum and PENNYLANE_AVAILABLE:
        x = layers.Dense(n_qubits, activation='tanh', name='proj_to_qubits')(x)
        # quantum head omitted here for clarity
        x = layers.Dense(dense_after_q, activation='relu')(x)
    else:
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(dense_after_q, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32', name='predictions')(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    def make_optimizer(opt_name, lr, weight_decay=WEIGHT_DECAY):
        try:
            if opt_name == 'adamw':
                Opt = tf.keras.optimizers.experimental.AdamW
                return Opt(learning_rate=lr, weight_decay=weight_decay)
            elif opt_name == 'adam':
                return tf.keras.optimizers.Adam(learning_rate=lr)
            elif opt_name == 'sgd':
                return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
            else:
                return tf.keras.optimizers.Adam(learning_rate=lr)
        except Exception:
            return tf.keras.optimizers.Adam(learning_rate=lr)

    optimizer = make_optimizer('adamw', lr)

    if use_focal_loss:
        loss_fn = categorical_focal_loss()
    else:
        loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)

    model.compile(optimizer=optimizer,
                  loss=loss_fn,
                  metrics=['accuracy'])
    return model

# --------------------------- Hyperparameter search (optional) ---------------------------
# (unchanged; will run on CPU if enabled)

def run_optuna_search(train_ds, val_ds, base_model, class_weight, trials=12, epochs_per_trial=6):
    if not OPTUNA_AVAILABLE:
        print('Optuna not available — skipping Optuna search')
        return None

    def objective(trial):
        try:
            opt_name = trial.suggest_categorical('optimizer', ['adamw', 'adam', 'sgd'])
            lr = trial.suggest_loguniform('lr', 1e-6, 1e-3)
            weight_decay = trial.suggest_loguniform('weight_decay', 1e-7, 1e-3)
            dropout = trial.suggest_float('dropout', 0.2, 0.6)
            dense_head = trial.suggest_categorical('dense_head', [256, 512, 768])
            mixup_alpha = trial.suggest_float('mixup_alpha', 0.0, 0.25)
            label_smoothing = trial.suggest_float('label_smoothing', 0.0, 0.12)
            fine_tune_at = trial.suggest_int('fine_tune_at', 20, 80)
            use_focal = trial.suggest_categorical('use_focal', [False, True])

            with strategy.scope():
                model = build_model(base_model, lr=lr, dense_after_q=dense_head, dropout_rate=dropout, label_smoothing=label_smoothing, use_focal_loss=use_focal)

            cb = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]

            ds_train_for_trial = train_ds
            if mixup_alpha > 0:
                try:
                    ds_train_for_trial = ds_train_for_trial.map(lambda x, y: mixup_batch(x, y, mixup_alpha), num_parallel_calls=AUTOTUNE)
                except Exception:
                    pass

            history = model.fit(ds_train_for_trial, validation_data=val_ds, epochs=epochs_per_trial, callbacks=cb, class_weight=class_weight, verbose=0)
            val_acc = history.history.get('val_accuracy', [0])[-1]
            tf.keras.backend.clear_session()
            return val_acc
        except Exception as e:
            print('Trial failed with exception:', e)
            traceback.print_exc()
            tf.keras.backend.clear_session()
            return 0.0

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=trials)
    print('Optuna study best params:', study.best_params)
    return study.best_params


def run_keras_tuner_search(train_ds, val_ds, base_model, class_weight, max_trials=8, epochs_per_trial=6):
    if not KERASTUNER_AVAILABLE:
        print('Keras Tuner not available — skipping tuner search')
        return None

    def model_builder(hp):
        lr = hp.Float('lr', 1e-6, 1e-3, sampling='log', default=1e-4)
        dropout = hp.Float('dropout', 0.2, 0.6, step=0.1, default=0.4)
        dense_head = hp.Choice('dense_head', [256, 512, 768], default=512)
        use_focal = hp.Boolean('use_focal', default=False)
        model = build_model(base_model, lr=lr, dense_after_q=dense_head, dropout_rate=dropout, use_focal_loss=use_focal)
        return model

    tuner = kt.BayesianOptimization(model_builder, objective='val_accuracy', max_trials=max_trials, directory='kt_dir', project_name='quantum_inception_opt')
    tuner.search(train_ds, validation_data=val_ds, epochs=epochs_per_trial, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)], class_weight=class_weight)
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    print('Keras Tuner best hyperparameters:', best_hp.values)
    return best_hp.values

# --------------------------- Build and train (main) ---------------------------

with strategy.scope():
    base_model = build_feature_extractor()

best_hp = None
if RUN_HP_SEARCH:
    print('Starting hyperparameter search...')
    if OPTUNA_AVAILABLE and HP_USE_OPTUNA:
        try:
            best_hp = run_optuna_search(train_ds, val_ds, base_model, class_weight, trials=HP_TRIALS, epochs_per_trial=HP_EPOCHS)
        except Exception as e:
            print('Optuna search failed:', e)
            best_hp = None
    if best_hp is None and KERASTUNER_AVAILABLE:
        try:
            best_hp = run_keras_tuner_search(train_ds, val_ds, base_model, class_weight, max_trials=HP_TRIALS, epochs_per_trial=HP_EPOCHS)
        except Exception as e:
            print('Keras Tuner search failed:', e)
            best_hp = None
    print('Hyperparameter search complete. best_hp =', best_hp)

if isinstance(best_hp, dict):
    chosen_lr = best_hp.get('lr', 1e-4)
    chosen_dropout = best_hp.get('dropout', DROPOUT_RATE)
    chosen_dense = best_hp.get('dense_head', DENSE_AFTER_Q)
    chosen_use_focal = best_hp.get('use_focal', use_focal)
    chosen_weight_decay = best_hp.get('weight_decay', WEIGHT_DECAY)
    chosen_fine_tune_at = best_hp.get('fine_tune_at', FINE_TUNE_AT)
else:
    chosen_lr = 1e-4
    chosen_dropout = DROPOUT_RATE
    chosen_dense = DENSE_AFTER_Q
    chosen_use_focal = use_focal
    chosen_weight_decay = WEIGHT_DECAY
    chosen_fine_tune_at = FINE_TUNE_AT

print('Final hyperparameters to use (fine-tuning disabled):')
print(' lr=', chosen_lr, ' dropout=', chosen_dropout, ' dense=', chosen_dense, ' use_focal=', chosen_use_focal, ' weight_decay=', chosen_weight_decay)

with strategy.scope():
    model = build_model(base_model, lr=chosen_lr, dense_after_q=chosen_dense, dropout_rate=chosen_dropout, label_smoothing=LABEL_SMOOTHING, use_focal_loss=chosen_use_focal)

model.summary()

# Callbacks
checkpoint_filepath = 'best_quantum_inception_cpu.h5'
callbacks_initial = [
    tf.keras.callbacks.ModelCheckpoint(checkpoint_filepath, monitor='val_accuracy', save_best_only=True),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=EARLYSTOP_PATIENCE, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    StopOnMetricThreshold(monitor='val_accuracy', threshold=THRESHOLD_ACC),
]

steps_per_epoch = math.ceil(total / BATCH_SIZE) if total > 0 else None

# Re-compile model with warmup+cosine schedule for initial training when step info is available
with strategy.scope():
    if steps_per_epoch:
        init_total_steps = INITIAL_EPOCHS * steps_per_epoch
        init_warmup_steps = max(1, int(0.1 * init_total_steps))
        init_lr_schedule = WarmUpCosine(
            base_lr=chosen_lr,
            total_steps=init_total_steps,
            warmup_steps=init_warmup_steps,
            min_lr=max(1e-7, chosen_lr * 0.1),
        )
        try:
            Opt = tf.keras.optimizers.experimental.AdamW
            init_optimizer = Opt(learning_rate=init_lr_schedule, weight_decay=chosen_weight_decay)
        except Exception:
            init_optimizer = tf.keras.optimizers.Adam(learning_rate=init_lr_schedule)
    else:
        try:
            Opt = tf.keras.optimizers.experimental.AdamW
            init_optimizer = Opt(learning_rate=chosen_lr, weight_decay=chosen_weight_decay)
        except Exception:
            init_optimizer = tf.keras.optimizers.Adam(learning_rate=chosen_lr)
    model.compile(optimizer=init_optimizer, loss=model.loss, metrics=['accuracy'])

# Choose callbacks for initial training depending on whether a schedule is used
if steps_per_epoch:
    callbacks_init_actual = [
        tf.keras.callbacks.ModelCheckpoint(checkpoint_filepath, monitor='val_accuracy', save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=EARLYSTOP_PATIENCE, restore_best_weights=True),
        StopOnMetricThreshold(monitor='val_accuracy', threshold=THRESHOLD_ACC),
    ]
else:
    callbacks_init_actual = callbacks_initial

print("\n--- Starting training (frozen backbone only; fine-tuning disabled) ---\n")
history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS,
    callbacks=callbacks_init_actual,
    class_weight=class_weight
)

print("\n--- Skipping fine-tuning phase (explicitly disabled) ---\n")
history2 = None

# Save final model
model.save('quantum_inception_trained_cpu.h5')
print('Saved trained model to quantum_inception_trained_cpu.h5')

if EXIT_AFTER_TRAIN:
    sys.exit(0)

# --------------------------- Consolidate history ---------------------------

def merge_history(h1, h2):
    if h2 is None:
        return h1
    merged = {}
    for k in h1.history.keys():
        merged[k] = h1.history[k] + h2.history.get(k, [])
    return type('H', (), {'history': merged})

history = merge_history(history1, history2)

# --------------------------- Plot training curves ---------------------------

def save_metrics_json(history, out_prefix='train'):
    metrics = history.history
    out_file = f"{out_prefix}_metrics.json"
    try:
        with open(out_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved numeric training metrics to {out_file}")
    except Exception as e:
        print('Could not save JSON metrics:', e)


def plot_training(history, out_prefix='train'):
    if not PLOTTING_AVAILABLE:
        print('matplotlib not available — saving numeric metrics instead of plots')
        save_metrics_json(history, out_prefix=out_prefix)
        return

    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])

    epochs_range = range(1, len(acc) + 1)

    plt.figure()
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    acc_png = f"{out_prefix}_accuracy.png"
    plt.savefig(acc_png)
    print(f"Saved accuracy plot to {acc_png}")
    plt.close()

    plt.figure()
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    loss_png = f"{out_prefix}_loss.png"
    plt.savefig(loss_png)
    print(f"Saved loss plot to {loss_png}")
    plt.close()

plot_training(history)

# --------------------------- Evaluation ---------------------------

def create_normal_confusion_matrix(cm, class_names, save_path='confusion_matrix.png'):
    """
    Create a normal, clean confusion matrix with standard matplotlib styling
    """
    if not PLOTTING_AVAILABLE:
        return
    
    # Create figure with standard size
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Use standard colormap (Blues)
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues', aspect='auto')
    
    # Add simple title
    ax.set_title('Confusion Matrix - Diabetic Retinopathy Classification', 
                 fontsize=14, fontweight='normal', pad=20)
    
    # Add standard colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Number of Predictions', fontsize=10)
    
    # Set tick marks
    tick_marks = np.arange(NUM_CLASSES)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    
    # Set axis labels
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)
    
    # Add simple grid
    ax.grid(False)
    
    # Add text annotations with simple styling
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        value = cm[i, j]
        
        # Simple text color based on cell value
        if value > thresh:
            text_color = "white"
        else:
            text_color = "black"
        
        # Add the count with simple styling
        ax.text(j, i, f'{value:d}',
                horizontalalignment="center",
                verticalalignment="center",
                color=text_color,
                fontsize=10)
    
    # Standard axis labels
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    
    # Standard background
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Standard spines
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1)
    
    plt.tight_layout()
    
    # Save with standard quality
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved normal confusion matrix to {save_path}")
    plt.close()

# --------------------------- Evaluation ---------------------------

y_true_list = []
y_prob_list = []

for batch_images, batch_labels in val_ds:
    y_true_list.append(batch_labels.numpy())
    preds = model.predict(batch_images)
    y_prob_list.append(preds)

if len(y_true_list) == 0:
    print('Validation set appears empty — cannot evaluate. Exiting.')
    sys.exit(1)

y_true = np.vstack(y_true_list)
y_prob = np.vstack(y_prob_list)

y_true_indices = np.argmax(y_true, axis=1)
y_pred_indices = np.argmax(y_prob, axis=1)

cm = confusion_matrix(y_true_indices, y_pred_indices)

# Create the normal confusion matrix
create_normal_confusion_matrix(cm, class_names, 'confusion_matrix.png')

# Also save the raw data
np.save('confusion_matrix.npy', cm)
try:
    with open('confusion_matrix.json', 'w') as f:
        json.dump(cm.tolist(), f)
    print('Saved confusion matrix data to confusion_matrix.npy and confusion_matrix.json')
except Exception as e:
    print('Could not save confusion matrix JSON:', e)

report = classification_report(y_true_indices, y_pred_indices, target_names=class_names, digits=4)
with open('classification_report.txt', 'w') as f:
    f.write(report)
print('Saved classification report to classification_report.txt')
print('\n' + report)

# ROC curves
if NUM_CLASSES == 2:
    fpr, tpr, _ = roc_curve(y_true[:, 1], y_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    if PLOTTING_AVAILABLE:
        plt.figure()
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC - Binary')
        plt.legend(loc='lower right')
        plt.savefig('roc_curve.png')
        plt.close()
        print('Saved ROC curve to roc_curve.png')
    else:
        try:
            with open('roc_binary.json', 'w') as f:
                json.dump({'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': float(roc_auc)}, f)
            print('Saved ROC data to roc_binary.json')
        except Exception as e:
            print('Could not save ROC JSON:', e)
else:
    fpr = dict(); tpr = dict(); roc_auc = dict()
    for i in range(NUM_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr['micro'], tpr['micro'], _ = roc_curve(y_true.ravel(), y_prob.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

    if PLOTTING_AVAILABLE:
        plt.figure(figsize=(10, 8))
        plt.plot(fpr['micro'], tpr['micro'], label=f'micro-average ROC (area = {roc_auc["micro"]:.2f})', lw=2)
        for i in range(NUM_CLASSES):
            plt.plot(fpr[i], tpr[i], lw=1, label=f'ROC {class_names[i]} (AUC = {roc_auc[i]:.2f})')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multi-class ROC')
        plt.legend(loc='lower right')
        plt.savefig('roc_curve_multiclass.png')
        plt.close()
        print('Saved multi-class ROC curve to roc_curve_multiclass.png')
    else:
        try:
            roc_out = {}
            for i in range(NUM_CLASSES):
                roc_out[class_names[i]] = {'fpr': fpr[i].tolist(), 'tpr': tpr[i].tolist(), 'auc': float(roc_auc[i])}
            roc_out['micro'] = {'fpr': fpr['micro'].tolist(), 'tpr': tpr['micro'].tolist(), 'auc': float(roc_auc['micro'])}
            with open('roc_multiclass.json', 'w') as f:
                json.dump(roc_out, f)
            print('Saved ROC data to roc_multiclass.json')
        except Exception as e:
            print('Could not save ROC JSON:', e)

print('All evaluation artifacts saved: metrics (JSON or PNGs), confusion matrix, ROC curve(s), classification report.')
