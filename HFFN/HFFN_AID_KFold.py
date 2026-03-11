"""
HFFN — Histogram Feature Fusion Network
TensorFlow/Keras implementation — AID Dataset with 5-Fold Cross-Validation

Feature branch : SIFT descriptors (flattened to 25 600 features)
Image branch   : Pretrained CNN backbone (MobileNetV1/V2, InceptionV3, DenseNet121, Xception)
Training       : 5-Fold CV  |  Augmentation: rotation-90, zoom-20%, horizontal-flip

Dataset : AID  (Autism Image Dataset)
Classes : autistic | non_autistic

Usage:
    python HFFN_AID_KFold.py
"""

from tensorflow.keras.applications import (
    MobileNet, MobileNetV2, Xception, InceptionV3, DenseNet121
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, concatenate, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import KFold
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

datagen = ImageDataGenerator(rotation_range=90, zoom_range=0.2, horizontal_flip=True)

def data_generator(X, X_sift, y, batch_size=32):
    while True:
        idx = np.random.permutation(len(X))
        for i in range(0, len(idx), batch_size):
            b = idx[i:i + batch_size]
            if len(b) < batch_size:
                continue
            aug = np.array([
                next(datagen.flow(X[j].reshape(1, *X[j].shape), batch_size=1))[0]
                for j in b
            ])
            yield (aug, X_sift[b]), y[b]

print("Loading images …")
x_train, y_train = load_images(DATADIR_TRAIN)
x_valid, y_valid = load_images(DATADIR_VALID)
x_test,  y_test  = load_images(DATADIR_TEST)

print("Extracting SIFT features …")
x_train_s = extract_sift(x_train);  x_valid_s = extract_sift(x_valid);  x_test_s = extract_sift(x_test)

x_train   = x_train   / 255.0;  x_valid   = x_valid   / 255.0;  x_test   = x_test   / 255.0
x_train_s = x_train_s / 255.0;  x_valid_s = x_valid_s / 255.0;  x_test_s = x_test_s / 255.0
y_train = np.array(y_train);    y_valid = np.array(y_valid);    y_test  = np.array(y_test)

x_all   = np.concatenate((x_train,   x_valid),   axis=0)
x_all_s = np.concatenate((x_train_s, x_valid_s), axis=0)
y_all   = np.concatenate((y_train,   y_valid),   axis=0)

models_to_test = ['MobileNetV1', 'MobileNetV2', 'InceptionV3', 'DenseNet121', 'Xception']

for model_name in models_to_test:
    print(f"\n{'='*60}\nHFFN KFold — {model_name}\n{'='*60}")
    kf      = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_no = 1

    for train_idx, val_idx in kf.split(x_all):
        print(f"  Fold {fold_no} …")
        xf_tr, xf_val = x_all[train_idx],   x_all[val_idx]
        xs_tr, xs_val = x_all_s[train_idx], x_all_s[val_idx]
        yf_tr, yf_val = y_all[train_idx],   y_all[val_idx]

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

        model = Model(inputs=[img_input, sift_input], outputs=output, name=f'HFFN_{model_name}')
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

        steps = len(xf_tr) // 32
        model.fit(
            data_generator(xf_tr, xs_tr, yf_tr, batch_size=32),
            steps_per_epoch=steps, epochs=70,
            validation_data=([xf_val, xs_val], yf_val)
        )

        save_path = os.path.join(MODEL_SAVE_DIR, f'HFFN_{model_name}_fold{fold_no}.h5')
        model.save(save_path)

        y_pred = np.argmax(model.predict([x_test, x_test_s]), axis=1)
        print(f"\n  Results — HFFN {model_name} Fold {fold_no}:")
        print("  Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("  Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred, target_names=target_names))
        cm = confusion_matrix(y_test, y_pred); TN, FP, FN, TP = cm.ravel()
        print("  Sensitivity:", TP/(TP+FN) if (TP+FN)>0 else 0)
        print("  Specificity:", TN/(TN+FP) if (TN+FP)>0 else 0)
        fold_no += 1
