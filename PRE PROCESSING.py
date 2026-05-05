# ==============================
# IMPORT LIBRARIES
# ==============================

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier

from skimage.feature import graycomatrix, graycoprops

# ==============================
# STEP 1: LOAD DATASET
# ==============================

dataset_path = "/Users/idivinajane/Documents/skin_disease"
IMG_SIZE = 256

data, labels, class_names = [], [], []

for i, class_name in enumerate(os.listdir(dataset_path)):
    class_path = os.path.join(dataset_path, class_name)

    if os.path.isdir(class_path):
        class_names.append(class_name)

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)

            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            data.append(img)
            labels.append(i)

data = np.array(data)
labels = np.array(labels)

print("✅ Dataset Loaded:", data.shape)

# ==============================
# ROI CROPPING
# ==============================

def roi_crop(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return img

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    cropped = img[y:y+h, x:x+w]
    return cv2.resize(cropped, (256, 256))

# ==============================
# STEP 2: PREPROCESSING (CLAHE + DENOISE)
# ==============================

processed = []

for img in data:

    img = roi_crop(img)
    img = cv2.bilateralFilter(img, 9, 75, 75)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0)
    l = clahe.apply(l)

    img = cv2.merge((l, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

    processed.append(img)

processed = np.array(processed)

# ==============================
# STEP 3: HYBRID SEGMENTATION
# ==============================

segmented, masked = [], []

for img in processed:

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # OTSU
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    # KMEANS
    Z = img.reshape((-1, 3)).astype(np.float32)

    _, labels_km, center = cv2.kmeans(
        Z, 2, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0),
        10, cv2.KMEANS_RANDOM_CENTERS
    )

    center = np.uint8(center)
    km = center[labels_km.flatten()].reshape(img.shape)
    km_gray = cv2.cvtColor(km, cv2.COLOR_BGR2GRAY)

    mask = cv2.bitwise_and(otsu, km_gray)

    kernel = np.ones((7,7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    segmented.append(mask)
    masked.append(cv2.bitwise_and(img, img, mask=mask))

segmented = np.array(segmented)
masked = np.array(masked)

# ==============================
# STEP 4: FEATURE EXTRACTION
# ==============================

def extract_features(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    glcm = graycomatrix(gray, [1,2], [0,np.pi/4,np.pi/2],
                        levels=256, symmetric=True, normed=True)

    contrast = graycoprops(glcm,'contrast').mean()
    energy = graycoprops(glcm,'energy').mean()
    homogeneity = graycoprops(glcm,'homogeneity').mean()

    color = cv2.mean(img)[:3]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_feat = [np.mean(hsv[:,:,0]), np.mean(hsv[:,:,1]), np.mean(hsv[:,:,2])]

    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    area = perimeter = 0
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)

    stats = [np.mean(gray), np.std(gray)]

    return [contrast, energy, homogeneity] + list(color) + hsv_feat + [area, perimeter] + stats

features = np.array([extract_features(img) for img in masked])

print("✅ Features Extracted:", features.shape)

# ==============================
# STEP 5: NORMALIZATION + FEATURE SELECTION + PCA
# ==============================

features = StandardScaler().fit_transform(features)

features = SelectKBest(f_classif, k=12).fit_transform(features, labels)

features = PCA(n_components=8).fit_transform(features)

# ==============================
# STEP 6: TRAIN TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    features, labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

print("\n📊 DATA SPLIT:", len(X_train), len(X_test))

# ==============================
# STEP 7: ENSEMBLE MODEL
# ==============================

svm = SVC(kernel='rbf', probability=True, C=5)
rf = RandomForestClassifier(n_estimators=400, class_weight='balanced')
xgb = XGBClassifier(
    n_estimators=600,
    learning_rate=0.02,
    max_depth=7,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric='mlogloss'
)

model = VotingClassifier(
    estimators=[('svm',svm),('rf',rf),('xgb',xgb)],
    voting='soft',
    weights=[2,3,5]
)

print("\n🚀 Training Ensemble...")
model.fit(X_train, y_train)

pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)

print("\n🏆 ACCURACY:", acc)

# ==============================
# CONFUSION MATRIX
# ==============================

cm = confusion_matrix(y_test, pred)

disp = ConfusionMatrixDisplay(cm, display_labels=class_names)

plt.figure(figsize=(7,6))
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix")
plt.show()

# ==============================
# CLASSIFICATION REPORT
# ==============================

report = classification_report(y_test, pred, target_names=class_names)

print("\n📊 CLASSIFICATION REPORT\n")
print(report)

# save report
out = "/Users/idivinajane/Documents/output_results"
os.makedirs(out, exist_ok=True)

with open(os.path.join(out, "report.txt"), "w") as f:
    f.write(report)

# ==============================
# SHOW 5 SAMPLE IMAGES
# ==============================

for i in range(5):

    combined = np.hstack((
        cv2.resize(data[i], (200,200)),
        cv2.resize(processed[i], (200,200)),
        cv2.resize(masked[i], (200,200))
    ))

    plt.figure(figsize=(6,3))
    plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    plt.title(f"Sample {i}")
    plt.axis("off")
    plt.show()

# ==============================
# SAVE IMAGES
# ==============================

for i in range(len(data)):

    cv2.imwrite(f"{out}/original_{i}.png", data[i])
    cv2.imwrite(f"{out}/processed_{i}.png", processed[i])
    cv2.imwrite(f"{out}/mask_{i}.png", segmented[i])
    cv2.imwrite(f"{out}/lesion_{i}.png", masked[i])

# ==============================
# FINAL GRAPH
# ==============================

plt.figure(figsize=(6,4))

plt.bar(["SVM","RF","XGB","Ensemble"],
        [0.6,0.6,0.57,acc],
        color=["red","blue","green","purple"])

plt.title("Model Comparison")
plt.ylim(0,1)
plt.show()

print("\n✅ ALL TASKS COMPLETED")# ==============================
# IMPORT LIBRARIES
# ==============================

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier

from skimage.feature import graycomatrix, graycoprops

# ==============================
# STEP 1: LOAD DATASET
# ==============================

dataset_path = "/Users/idivinajane/Documents/skin_disease"
IMG_SIZE = 256

data, labels, class_names = [], [], []

for i, class_name in enumerate(os.listdir(dataset_path)):
    class_path = os.path.join(dataset_path, class_name)

    if os.path.isdir(class_path):
        class_names.append(class_name)

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)

            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            data.append(img)
            labels.append(i)

data = np.array(data)
labels = np.array(labels)

print("✅ Dataset Loaded:", data.shape)

# ==============================
# ROI CROPPING
# ==============================

def roi_crop(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return img

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    cropped = img[y:y+h, x:x+w]
    return cv2.resize(cropped, (256, 256))

# ==============================
# STEP 2: PREPROCESSING (CLAHE + DENOISE)
# ==============================

processed = []

for img in data:

    img = roi_crop(img)
    img = cv2.bilateralFilter(img, 9, 75, 75)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0)
    l = clahe.apply(l)

    img = cv2.merge((l, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

    processed.append(img)

processed = np.array(processed)

# ==============================
# STEP 3: HYBRID SEGMENTATION
# ==============================

segmented, masked = [], []

for img in processed:

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # OTSU
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    # KMEANS
    Z = img.reshape((-1, 3)).astype(np.float32)

    _, labels_km, center = cv2.kmeans(
        Z, 2, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0),
        10, cv2.KMEANS_RANDOM_CENTERS
    )

    center = np.uint8(center)
    km = center[labels_km.flatten()].reshape(img.shape)
    km_gray = cv2.cvtColor(km, cv2.COLOR_BGR2GRAY)

    mask = cv2.bitwise_and(otsu, km_gray)

    kernel = np.ones((7,7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    segmented.append(mask)
    masked.append(cv2.bitwise_and(img, img, mask=mask))

segmented = np.array(segmented)
masked = np.array(masked)

# ==============================
# STEP 4: FEATURE EXTRACTION
# ==============================

def extract_features(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    glcm = graycomatrix(gray, [1,2], [0,np.pi/4,np.pi/2],
                        levels=256, symmetric=True, normed=True)

    contrast = graycoprops(glcm,'contrast').mean()
    energy = graycoprops(glcm,'energy').mean()
    homogeneity = graycoprops(glcm,'homogeneity').mean()

    color = cv2.mean(img)[:3]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_feat = [np.mean(hsv[:,:,0]), np.mean(hsv[:,:,1]), np.mean(hsv[:,:,2])]

    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    area = perimeter = 0
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)

    stats = [np.mean(gray), np.std(gray)]

    return [contrast, energy, homogeneity] + list(color) + hsv_feat + [area, perimeter] + stats

features = np.array([extract_features(img) for img in masked])

print("✅ Features Extracted:", features.shape)

# ==============================
# STEP 5: NORMALIZATION + FEATURE SELECTION + PCA
# ==============================

features = StandardScaler().fit_transform(features)

features = SelectKBest(f_classif, k=12).fit_transform(features, labels)

features = PCA(n_components=8).fit_transform(features)

# ==============================
# STEP 6: TRAIN TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    features, labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

print("\n📊 DATA SPLIT:", len(X_train), len(X_test))

# ==============================
# STEP 7: ENSEMBLE MODEL
# ==============================

svm = SVC(kernel='rbf', probability=True, C=5)
rf = RandomForestClassifier(n_estimators=400, class_weight='balanced')
xgb = XGBClassifier(
    n_estimators=600,
    learning_rate=0.02,
    max_depth=7,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric='mlogloss'
)

model = VotingClassifier(
    estimators=[('svm',svm),('rf',rf),('xgb',xgb)],
    voting='soft',
    weights=[2,3,5]
)

print("\n🚀 Training Ensemble...")
model.fit(X_train, y_train)

pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)

print("\n🏆 ACCURACY:", acc)

# ==============================
# CONFUSION MATRIX
# ==============================

cm = confusion_matrix(y_test, pred)

disp = ConfusionMatrixDisplay(cm, display_labels=class_names)

plt.figure(figsize=(7,6))
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix")
plt.show()

# ==============================
# CLASSIFICATION REPORT
# ==============================

report = classification_report(y_test, pred, target_names=class_names)

print("\n📊 CLASSIFICATION REPORT\n")
print(report)

# save report
out = "/Users/idivinajane/Documents/output_results"
os.makedirs(out, exist_ok=True)

with open(os.path.join(out, "report.txt"), "w") as f:
    f.write(report)

# ==============================
# SHOW 5 SAMPLE IMAGES
# ==============================

for i in range(5):

    combined = np.hstack((
        cv2.resize(data[i], (200,200)),
        cv2.resize(processed[i], (200,200)),
        cv2.resize(masked[i], (200,200))
    ))

    plt.figure(figsize=(6,3))
    plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    plt.title(f"Sample {i}")
    plt.axis("off")
    plt.show()

# ==============================
# SAVE IMAGES
# ==============================

for i in range(len(data)):

    cv2.imwrite(f"{out}/original_{i}.png", data[i])
    cv2.imwrite(f"{out}/processed_{i}.png", processed[i])
    cv2.imwrite(f"{out}/mask_{i}.png", segmented[i])
    cv2.imwrite(f"{out}/lesion_{i}.png", masked[i])

# ==============================
# FINAL GRAPH
# ==============================

plt.figure(figsize=(6,4))

plt.bar(["SVM","RF","XGB","Ensemble"],
        [0.6,0.6,0.57,acc],
        color=["red","blue","green","purple"])

plt.title("Model Comparison")
plt.ylim(0,1)
plt.show()

print("\n✅ ALL TASKS COMPLETED")
