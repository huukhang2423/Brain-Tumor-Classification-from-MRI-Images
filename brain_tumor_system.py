"""
BRAIN TUMOR DETECTION SYSTEM - SVM ONLY VERSION
Save as: brain_tumor_system.py
Run: python brain_tumor_system.py
"""

import os
import warnings
warnings.filterwarnings('ignore')

print("""
╔══════════════════════════════════════════════════════════════════╗
║                 BRAIN TUMOR DETECTION SYSTEM                     ║
║              SVM-Based MRI Classification System                 ║
║                      SVM ONLY VERSION                            ║
╚══════════════════════════════════════════════════════════════════╝
""")

# ============================================
# IMPORTS
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
import joblib
from tqdm import tqdm
import json

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, roc_curve, auc)

# Set seeds
np.random.seed(42)

# ============================================
# CONFIGURATION
# ============================================

BASE_PATH = r"C:\Users\Admin\Desktop\BRAIN\archive"
MODEL_SAVE_PATH = r"C:\Users\Admin\Desktop\BRAIN\models"
RESULTS_PATH = r"C:\Users\Admin\Desktop\BRAIN\results"

IMG_SIZE = (224, 224)
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
NUM_CLASSES = len(CLASSES)

# Create directories
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)

print(f"✓ Configuration loaded")
print(f"✓ Base path: {BASE_PATH}")
print(f"✓ Classes: {CLASSES}")

# ============================================
# DATA LOADING
# ============================================

print(f"\n{'='*70}")
print("STEP 1: DATA LOADING")
print(f"{'='*70}")

def load_data(base_path, split='Training'):
    """Load images and labels from directory"""
    images = []
    labels = []

    split_path = os.path.join(base_path, split)
    print(f"\nLoading {split} data from: {split_path}")

    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Path not found: {split_path}")

    for class_idx, class_name in enumerate(CLASSES):
        class_path = os.path.join(split_path, class_name)

        if not os.path.exists(class_path):
            print(f"⚠️  Warning: {class_path} not found!")
            continue

        image_files = [f for f in os.listdir(class_path)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        print(f"  Loading {class_name}: {len(image_files)} images")

        for img_file in tqdm(image_files, desc=f"  Processing {class_name}", leave=False):
            img_path = os.path.join(class_path, img_file)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, IMG_SIZE)
                images.append(img)
                labels.append(class_idx)
            except Exception as e:
                print(f"    ⚠️ Error loading {img_file}: {e}")

    images = np.array(images)
    labels = np.array(labels)

    print(f"\n✓ Loaded {len(images)} images")
    print(f"  Shape: {images.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Class distribution: {np.bincount(labels)}")

    return images, labels

# Load training and testing data
try:
    X_train_raw, y_train = load_data(BASE_PATH, 'Training')
    X_test_raw, y_test = load_data(BASE_PATH, 'Testing')
except FileNotFoundError as e:
    print(f"\n❌ ERROR: {e}")
    print("Please check BASE_PATH and folder structure!")
    exit(1)

print(f"\n✓ Total training samples: {len(X_train_raw)}")
print(f"✓ Total testing samples: {len(X_test_raw)}")

# ============================================
# FEATURE EXTRACTION
# ============================================

print(f"\n{'='*70}")
print("STEP 2: FEATURE EXTRACTION")
print(f"{'='*70}")

def extract_features(images):
    """Extract handcrafted features for SVM"""
    features = []

    print("\nExtracting features...")
    for idx, img in enumerate(tqdm(images, desc="Processing")):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-8)

        # Statistical features
        mean = np.mean(gray)
        std = np.std(gray)

        # Texture
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_var = np.var(laplacian)

        # Edge density
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

        # Combine features
        feature_vector = np.concatenate([
            hist[::16],  # Reduced histogram (16 bins)
            [mean, std, texture_var, edge_density]
        ])

        features.append(feature_vector)

    features = np.array(features)
    print(f"✓ Feature extraction complete. Shape: {features.shape}")

    return features

# Extract features
X_train_features = extract_features(X_train_raw)
X_test_features = extract_features(X_test_raw)

# Scale features
print("\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_features)
X_test_scaled = scaler.transform(X_test_features)

# Save scaler
scaler_path = os.path.join(MODEL_SAVE_PATH, 'scaler.pkl')
joblib.dump(scaler, scaler_path)
print(f"✓ Scaler saved: {scaler_path}")

# ============================================
# TRAIN SVM MODEL
# ============================================

print(f"\n{'='*70}")
print("STEP 3: TRAINING SVM MODEL")
print(f"{'='*70}")

print("\n→ Training SVM with RBF kernel...")
svm_model = SVC(kernel='rbf', C=10, gamma='scale', probability=True,
                random_state=42, class_weight='balanced', verbose=True)
svm_model.fit(X_train_scaled, y_train)

# Save model
model_path = os.path.join(MODEL_SAVE_PATH, 'svm_model.pkl')
joblib.dump(svm_model, model_path)
print(f"\n✓ SVM model saved: {model_path}")

# ============================================
# MODEL EVALUATION
# ============================================

print(f"\n{'='*70}")
print("STEP 4: MODEL EVALUATION")
print(f"{'='*70}")

# Predictions
print("\nMaking predictions on test set...")
y_pred = svm_model.predict(X_test_scaled)
y_probs = svm_model.predict_proba(X_test_scaled)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=CLASSES)

print(f"\n{'='*70}")
print("RESULTS")
print(f"{'='*70}")
print(f"\n🎯 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"\n📊 Classification Report:\n")
print(class_report)

# ============================================
# VISUALIZATIONS
# ============================================

print(f"\n{'='*70}")
print("STEP 5: GENERATING VISUALIZATIONS")
print(f"{'='*70}")

# 1. Confusion Matrix
print("\n→ Creating confusion matrix...")
plt.figure(figsize=(10, 8))
conf_matrix_norm = conf_matrix.astype('float') / (conf_matrix.sum(axis=1)[:, np.newaxis] + 1e-8)

sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=CLASSES, yticklabels=CLASSES,
            cbar_kws={'label': 'Normalized Count'})
plt.title(f'Confusion Matrix - SVM Model\nAccuracy: {accuracy:.4f}',
          fontweight='bold', fontsize=14)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
cm_path = os.path.join(RESULTS_PATH, 'confusion_matrix.png')
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {cm_path}")

# 2. ROC Curves
print("→ Creating ROC curves...")
fig, ax = plt.subplots(figsize=(10, 8))

for class_idx, class_name in enumerate(CLASSES):
    # One-vs-rest
    y_true_binary = (y_test == class_idx).astype(int)
    y_score = y_probs[:, class_idx]

    fpr, tpr, _ = roc_curve(y_true_binary, y_score)
    roc_auc = auc(fpr, tpr)

    ax.plot(fpr, tpr, linewidth=2,
            label=f'{class_name} (AUC={roc_auc:.3f})')

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - SVM Model', fontweight='bold', fontsize=14)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
roc_path = os.path.join(RESULTS_PATH, 'roc_curves.png')
plt.savefig(roc_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {roc_path}")

# 3. Sample Predictions
print("→ Creating sample predictions visualization...")
fig, axes = plt.subplots(4, 4, figsize=(16, 16))
axes = axes.ravel()

for i in range(16):
    idx = np.random.randint(0, len(X_test_raw))
    original_image = X_test_raw[idx]
    true_label = y_test[idx]

    # Get prediction
    features = X_test_scaled[idx].reshape(1, -1)
    pred_label = svm_model.predict(features)[0]
    probs = svm_model.predict_proba(features)[0]
    confidence = probs[pred_label]

    # Display
    axes[i].imshow(original_image)

    color = 'green' if pred_label == true_label else 'red'
    axes[i].set_title(f'True: {CLASSES[true_label]}\n'
                     f'Pred: {CLASSES[pred_label]} ({confidence:.2%})',
                     color=color, fontweight='bold', fontsize=10)
    axes[i].axis('off')

plt.suptitle('Sample Predictions - SVM Model', fontsize=16, fontweight='bold')
plt.tight_layout()
pred_path = os.path.join(RESULTS_PATH, 'sample_predictions.png')
plt.savefig(pred_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {pred_path}")

# 4. Class Distribution
print("→ Creating class distribution plot...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Training distribution
train_counts = np.bincount(y_train)
axes[0].bar(CLASSES, train_counts, color='skyblue', edgecolor='black')
axes[0].set_title('Training Set Distribution', fontweight='bold', fontsize=12)
axes[0].set_ylabel('Number of Samples', fontsize=11)
axes[0].grid(True, alpha=0.3, axis='y')
for i, count in enumerate(train_counts):
    axes[0].text(i, count + 20, str(count), ha='center', fontweight='bold')

# Test distribution
test_counts = np.bincount(y_test)
axes[1].bar(CLASSES, test_counts, color='lightcoral', edgecolor='black')
axes[1].set_title('Test Set Distribution', fontweight='bold', fontsize=12)
axes[1].set_ylabel('Number of Samples', fontsize=11)
axes[1].grid(True, alpha=0.3, axis='y')
for i, count in enumerate(test_counts):
    axes[1].text(i, count + 5, str(count), ha='center', fontweight='bold')

plt.suptitle('Dataset Class Distribution', fontsize=14, fontweight='bold')
plt.tight_layout()
dist_path = os.path.join(RESULTS_PATH, 'class_distribution.png')
plt.savefig(dist_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {dist_path}")

# ============================================
# SAVE RESULTS
# ============================================

print(f"\n{'='*70}")
print("STEP 6: SAVING RESULTS")
print(f"{'='*70}")

# Save detailed results
results = {
    'model': 'SVM',
    'accuracy': float(accuracy),
    'test_samples': int(len(y_test)),
    'train_samples': int(len(y_train)),
    'classes': CLASSES,
    'confusion_matrix': conf_matrix.tolist(),
    'classification_report': class_report,
    'config': {
        'kernel': 'rbf',
        'C': 10,
        'gamma': 'scale',
        'class_weight': 'balanced',
        'img_size': IMG_SIZE
    }
}

results_path = os.path.join(RESULTS_PATH, 'svm_results.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=4)
print(f"\n✓ Results saved: {results_path}")

# ============================================
# FINAL SUMMARY
# ============================================

print(f"\n{'='*70}")
print("FINAL SUMMARY")
print(f"{'='*70}")

print(f"\n📊 DATASET INFORMATION:")
print(f"  • Training samples: {len(X_train_raw)}")
print(f"  • Testing samples: {len(X_test_raw)}")
print(f"  • Classes: {CLASSES}")
print(f"  • Image size: {IMG_SIZE}")
print(f"  • Feature dimension: {X_train_scaled.shape[1]}")

print(f"\n🤖 MODEL INFORMATION:")
print(f"  • Algorithm: Support Vector Machine (SVM)")
print(f"  • Kernel: RBF (Radial Basis Function)")
print(f"  • C parameter: 10")
print(f"  • Gamma: scale")
print(f"  • Class weight: balanced")

print(f"\n📈 PERFORMANCE:")
print(f"  • Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print(f"\n📁 SAVED FILES:")
print(f"\n  Models:")
print(f"    • {model_path}")
print(f"    • {scaler_path}")
print(f"\n  Results:")
print(f"    • {cm_path}")
print(f"    • {roc_path}")
print(f"    • {pred_path}")
print(f"    • {dist_path}")
print(f"    • {results_path}")

print(f"\n{'='*70}")
print("✅ TRAINING COMPLETE!")
print(f"{'='*70}\n")

print("💡 NEXT STEPS:")
print("  1. Review confusion matrix to identify misclassification patterns")
print("  2. Check ROC curves for class-specific performance")
print("  3. Analyze sample predictions for model understanding")
print("  4. Consider trying different kernels (linear, polynomial)")
print("  5. Test on new unseen data for real-world validation")

print("\n" + "="*70)
print("📧 Ready for deployment or further tuning!")
print("="*70 + "\n")
