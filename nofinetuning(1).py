"""
================================================================================
EXPERIMENT A: BASELINE MODEL (FROZEN BACKBONE ONLY)
================================================================================
"""

import os
import json
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# 1. SETUP & KONFIGURASI PATH OTOMATIS
# ==============================================================================
# Nama Folder Eksperimen (Otomatis dibuat)
ROOT_EXP_DIR = 'Experiment_Baseline_Frozen'
MODEL_DIR = os.path.join(ROOT_EXP_DIR, 'models')
VIS_DIR = os.path.join(ROOT_EXP_DIR, 'visualizations')

if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
if not os.path.exists(VIS_DIR): os.makedirs(VIS_DIR)

TRAIN_DIR = r'C:\paruparu\chest_xray\chest_xray\train'
VAL_DIR = r'C:\paruparu\chest_xray\chest_xray\val'
TEST_DIR = r'C:\paruparu\chest_xray\chest_xray\test'

BATCH_SIZE = 16
IMG_SIZE = 224
EPOCHS = 20  # Langsung 20 Epoch (Frozen)

# Warna Standar Jurnal
COLOR_TRAIN = '#0000FF' # Biru
COLOR_VAL   = '#FFA500' # Oranye

def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
set_seed(42)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError: pass

# ==============================================================================
# 2. DATA LOADING (SAMA PERSIS)
# ==============================================================================
print("\n[PROCESS] Loading Data for Baseline...")

train_datagen_augment = ImageDataGenerator(
    rescale=1./255, rotation_range=10, width_shift_range=0.1, height_shift_range=0.1,
    zoom_range=0.15, shear_range=0.1, brightness_range=[0.9, 1.1],
    horizontal_flip=False, fill_mode='constant', cval=0
)
val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator_base = train_datagen_augment.flow_from_directory(
    TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='binary', shuffle=True, seed=42
)

# Upsampling Logic
class_counts = np.bincount(train_generator_base.classes)
max_count = np.max(class_counts)
indices = []
for class_idx in range(len(class_counts)):
    class_indices = np.where(train_generator_base.classes == class_idx)[0]
    upsampled_indices = np.random.choice(class_indices, size=max_count, replace=True)
    indices.extend(upsampled_indices)
np.random.shuffle(indices)

def upsampled_generator_fn():
    while True:
        np.random.shuffle(indices)
        for i in range(0, len(indices), BATCH_SIZE):
            batch_indices = indices[i:i+BATCH_SIZE]
            if not batch_indices: continue
            batch_x, batch_y = [], []
            for idx in batch_indices:
                fname = train_generator_base.filepaths[idx]
                label = train_generator_base.classes[idx]
                img = tf.keras.preprocessing.image.load_img(fname, target_size=(IMG_SIZE, IMG_SIZE))
                x = tf.keras.preprocessing.image.img_to_array(img)
                x = train_datagen_augment.random_transform(x)
                x = train_datagen_augment.standardize(x)
                batch_x.append(x)
                batch_y.append(label)
            yield np.array(batch_x), np.array(batch_y)

train_generator = upsampled_generator_fn()
steps_per_epoch = len(indices) // BATCH_SIZE

val_generator = val_test_datagen.flow_from_directory(
    VAL_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='binary', shuffle=False)
test_generator = val_test_datagen.flow_from_directory(
    TEST_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='binary', shuffle=False)

# ==============================================================================
# 3. BUILD MODEL (BASELINE CONFIG)
# ==============================================================================
def build_model():
    base_model = DenseNet121(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # === PERBEDAAN UTAMA: BACKBONE FROZEN SELAMANYA ===
    base_model.trainable = False 
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='sigmoid')(x)
    return Model(inputs=base_model.input, outputs=output)

model = build_model()
metrics = ['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]

# Simpan Model di Folder Eksperimen
best_model_path = os.path.join(MODEL_DIR, 'best_baseline_model.h5')
checkpoint = ModelCheckpoint(best_model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)

# ==============================================================================
# 4. TRAINING STANDARD (SINGLE PHASE)
# ==============================================================================
print("\n" + "="*60)
print(">>> START TRAINING BASELINE (Frozen Only)")
print("="*60)

model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=metrics)

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[checkpoint],
    verbose=1
)

# ==============================================================================
# 5. LOGGING JSON & VISUALISASI
# ==============================================================================
print("\n[PROCESS] Saving Logs & Generating Visualizations...")

# A. Simpan Log JSON
history_dict = history.history
json_path = os.path.join(ROOT_EXP_DIR, 'history_log.json')
with open(json_path, 'w') as f:
    json.dump(history_dict, f)
print(f"✓ History JSON saved to: {json_path}")

# B. Visualisasi (Style Biru/Oranye)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(acc) + 1)

sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12})

def plot_metric_journal(train, val, name, filename):
    plt.figure(figsize=(10, 8))
    # Garis Train (Biru)
    plt.plot(epochs_range, train, label=f'Training {name}', color=COLOR_TRAIN, linewidth=2.5)
    # Garis Val (Oranye)
    plt.plot(epochs_range, val, label=f'Validation {name}', color=COLOR_VAL, linewidth=2.5)
    
    plt.title(f'Baseline Model: {name} History', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel(name, fontsize=14)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, filename), dpi=300)
    plt.close()

plot_metric_journal(acc, val_acc, 'Accuracy', '1_Accuracy.png')
plot_metric_journal(loss, val_loss, 'Loss', '2_Loss.png')

# C. Evaluasi Akhir
print("\n[PROCESS] Evaluating Best Baseline Model...")
model.load_weights(best_model_path)
test_generator.reset()
y_pred_probs = model.predict(test_generator)
y_pred_classes = (y_pred_probs > 0.5).astype(int).flatten()
y_true = test_generator.classes

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_true, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'],
            annot_kws={"size": 16, "weight": "bold"}, cbar=False)
plt.title('Baseline: Confusion Matrix', fontsize=16, fontweight='bold')
plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, '3_Confusion_Matrix.png'), dpi=300)
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color=COLOR_VAL, lw=3, label=f'AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.title('Baseline: ROC Curve', fontsize=16, fontweight='bold')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.legend(loc="lower right"); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, '4_ROC_Curve.png'), dpi=300)
plt.close()

print(f"\n[DONE] Hasil Baseline tersimpan di folder: {ROOT_EXP_DIR}")