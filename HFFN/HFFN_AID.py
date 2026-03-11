"""
HFFN — Histogram Feature Fusion Network
TensorFlow/Keras implementation — AID Dataset (train / valid / test split)

Feature branch : SIFT descriptors (flattened to 25 600 features)
Image branch   : Pretrained CNN backbone (MobileNetV1/V2, InceptionV3, DenseNet121, Xception)
Fusion         : Concatenation -> Dense -> Softmax (2 classes)

Dataset : AID  (Autism Image Dataset)
Classes : autistic | non_autistic

Usage:
    python HFFN_AID.py
"""

from tensorflow.keras.applications import (
    MobileNet, MobileNetV2, Xception, InceptionV3, DenseNet121
)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, concatenate, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import cv2
import numpy as np
import os

# ── Configuration ─────────────────────────────────────────────────────────────
IMG_SIZE       = 224
DATADIR_TRAIN  = 'AID/train'
DATADIR_VALID  = 'AID/valid'
DATADIR_TEST   = 'AID/test'
CARTEGORIES    = ['autistic', 'non_autistic']
MODEL_SAVE_DIR = 'saved_models/HFFN'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
target_names   = ['autistic', 'non_autistic']

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_images(directory):
    X, Y = [], []
    for cat in CARTEGORIES:
        path      = os.path.join(directory, cat)
        class_num = CARTEGORIES.index(cat)
        for img_name in os.listdir(path):
            img = cv2.imread(os.path.join(path, img_name))
            if img is None:
                continue
            X.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
            Y.append(class_num)
    return np.array(X), np.array(Y)

def extract_sift(images):
    features = []
    for img in images:
        gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift    = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)
        fd      = des.flatten()
        fd      = cv2.resize(fd, (1, 25600)).flatten()
        features.append(fd)
    return np.array(features)

def load_backbone(name):
    cfg = dict(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
    return {
        'MobileNetV1': MobileNet,
        'MobileNetV2': MobileNetV2,
        'Xception'   : Xception,
        'InceptionV3': InceptionV3,
        'DenseNet121': DenseNet121,
    }[name](**cfg)

# ── Step 1: Load images ───────────────────────────────────────────────────────
print("Loading images …")
x_train, y_train = load_images(DATADIR_TRAIN)
x_valid, y_valid = load_images(DATADIR_VALID)
x_test,  y_test  = load_images(DATADIR_TEST)

# ── Step 2: Extract SIFT features ────────────────────────────────────────────
print("Extracting SIFT features …")
x_train_s = extract_sift(x_train)
x_valid_s = extract_sift(x_valid)
x_test_s  = extract_sift(x_test)

# ── Step 3: Normalise ─────────────────────────────────────────────────────────
x_train   = x_train   / 255.0;  x_valid   = x_valid   / 255.0;  x_test   = x_test   / 255.0
x_train_s = x_train_s / 255.0;  x_valid_s = x_valid_s / 255.0;  x_test_s = x_test_s / 255.0

print("Train:", x_train.shape, x_train_s.shape)
print("Valid:", x_valid.shape, x_valid_s.shape)
print("Test :", x_test.shape,  x_test_s.shape)

# ── Step 4: Build HFFN and run experiments ────────────────────────────────────
models_to_test = ['MobileNetV1', 'MobileNetV2', 'InceptionV3', 'DenseNet121', 'Xception']

for model_name in models_to_test:
    print(f"\n{'='*60}\nHFFN — {model_name}\n{'='*60}")

    backbone           = load_backbone(model_name)
    backbone.trainable = True

    img_input    = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='image_input')
    img_features = backbone(img_input)
    img_features = Flatten()(img_features)
    img_features = Dense(512, activation='relu')(img_features)
    img_features = Dropout(0.4)(img_features)
    img_out      = Dense(256, activation='relu')(img_features)

    sift_input = Input(shape=(25600,), name='sift_input')
    sift_feat  = Dense(256, activation='relu')(sift_input)
    sift_feat  = Dropout(0.4)(sift_feat)
    sift_out   = Dense(128, activation='relu')(sift_feat)

    fused  = concatenate([img_out, sift_out])
    fused  = Dense(256, activation='relu')(fused)
    fused  = Dropout(0.5)(fused)
    output = Dense(2, activation='softmax')(fused)

    model = Model(inputs=[img_input, sift_input], outputs=output,
                  name=f'HFFN_{model_name}')
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(learning_rate=0.0001),
                  metrics=['accuracy'])
    model.summary()

    model.fit(
        [x_train, x_train_s], y_train,
        batch_size=32,
        epochs=70,
        validation_data=([x_valid, x_valid_s], y_valid),
        shuffle=True
    )

    save_path = os.path.join(MODEL_SAVE_DIR, f'HFFN_{model_name}.h5')
    model.save(save_path)
    print(f"Saved: {save_path}")

    y_pred = np.argmax(model.predict([x_test, x_test_s]), axis=1)

    print(f"\nResults — HFFN {model_name}:")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=target_names))
    cm           = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = cm.ravel()
    print("Sensitivity:", TP / (TP + FN) if (TP + FN) > 0 else 0)
    print("Specificity:", TN / (TN + FP) if (TN + FP) > 0 else 0)
