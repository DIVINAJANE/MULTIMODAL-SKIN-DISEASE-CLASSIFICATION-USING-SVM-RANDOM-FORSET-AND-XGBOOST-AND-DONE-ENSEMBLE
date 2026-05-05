import os
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import math

# ==============================
# DATASET PATH
# ==============================
DATASET_PATH = "/Users/idivinajane/Documents/PSORIASIS AND NORMAL SKIN"
IMG_SIZE = 128

features = []
labels = []
img_paths = []  # store paths for visualization

# ==============================
# FEATURE EXTRACTION FUNCTION
# ==============================
def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Texture features (GLCM)
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]

    # Color feature (red channel mean as erythema proxy)
    red_channel = image[:, :, 2]
    mean_red = np.mean(red_channel)

    # Edge feature
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges) / edges.size

    return [contrast, homogeneity, energy, mean_red, edge_density]

# ==============================
# LOAD IMAGES AND EXTRACT FEATURES
# ==============================
for folder in os.listdir(DATASET_PATH):
    class_path = os.path.join(DATASET_PATH, folder)
    if not os.path.isdir(class_path):
        continue

    label = 1 if "PSORIASIS" in folder.upper() else 0

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        feature_vector = extract_features(img)

        features.append(feature_vector)
        labels.append(label)
        img_paths.append(img_path)  # store path

# ==============================
# CONVERT TO ARRAYS
# ==============================
X = np.array(features)
y = np.array(labels)
img_paths = np.array(img_paths)

print("Total samples:", X.shape[0])
print("Feature vector size:", X.shape[1])

# ==============================
# SPLIT DATA
# ==============================
X_train, X_test, y_train, y_test, img_paths_train, img_paths_test = train_test_split(
    X, y, img_paths, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# SCALE FEATURES
# ==============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================
# TRAIN SVM
# ==============================
clf = SVC(kernel='linear', probability=True)
clf.fit(X_train_scaled, y_train)

# ==============================
# PREDICT
# ==============================
y_pred = clf.predict(X_test_scaled)

# ==============================
# EVALUATE
# ==============================
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ==============================
# FUNCTION TO DISPLAY IMAGES IN BATCHES
# ==============================
def show_images_batches(img_paths, predictions, true_labels, batch_size=50):
    """
    Display images in batches to avoid memory issues.
    """
    total = len(img_paths)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_paths = img_paths[start:end]
        batch_preds = predictions[start:end]
        batch_labels = true_labels[start:end]

        cols = 5  # 5 images per row
        rows = math.ceil(len(batch_paths) / cols)
        plt.figure(figsize=(20, rows*4))

        for i in range(len(batch_paths)):
            img = cv2.imread(batch_paths[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pred_label = 'PSORIASIS' if batch_preds[i] else 'NORMAL'
            true_label = 'PSORIASIS' if batch_labels[i] else 'NORMAL'
            color = 'green' if batch_preds[i] == batch_labels[i] else 'red'

            plt.subplot(rows, cols, i + 1)
            plt.imshow(img)
            plt.title(f"P: {pred_label}\nT: {true_label}", color=color, fontsize=10)
            plt.axis('off')

        plt.tight_layout()
        plt.show()

# ==============================
# DISPLAY ALL TEST IMAGES IN BATCHES
# ==============================
show_images_batches(img_paths_test, y_pred, y_test, batch_size=50)
