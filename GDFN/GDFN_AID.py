"""
GDFN — Geometric Distance Feature Network
TensorFlow/Keras implementation — AID Dataset (train / valid / test split)

Feature branch : Dlib 68-point facial landmark distances (31 pairs)
Image branch   : Pretrained CNN backbone (MobileNetV1/V2, InceptionV3, DenseNet121, Xception)
Fusion         : Concatenation -> Dense -> Softmax (2 classes)

Dataset : AID  (Autism Image Dataset)
Classes : autistic | non_autistic

Usage:
    python GDFN_AID.py
"""

from tensorflow.keras.applications import (
    MobileNet, MobileNetV2,
    Xception, InceptionV3, DenseNet121
)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, concatenate, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import cv2
import numpy as np
import os
import pickle
import dlib

# ── Configuration ─────────────────────────────────────────────────────────────
IMG_SIZE       = 224
DATADIR_TRAIN  = 'AID/train'
DATADIR_VALID  = 'AID/valid'
DATADIR_TEST   = 'AID/test'
CARTEGORIES    = ['autistic', 'non_autistic']
MODEL_SAVE_DIR = 'saved_models/GDFN'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
target_names   = ['autistic', 'non_autistic']

detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# ── Landmark configuration (same 31 pairs used in all GDFN variants) ──────────
KEY_POINT_LABELS = {
    18: 'C', 37: 'D', 41: 'E', 40: 'F', 28: 'B', 43: 'G', 47: 'H', 46: 'I',
    27: 'J', 32: 'K', 34: 'L', 36: 'M', 51: 'O', 52: 'P', 53: 'Q',
    55: 'R',  9: 'S', 49: 'N'
}
POINT_PAIRS = [
    ('B','S'), ('E','J'), ('G','J'), ('L','S'), ('C','H'), ('E','N'),
    ('G','O'), ('M','N'), ('C','K'), ('E','P'), ('G','S'), ('M','S'),
    ('C','L'), ('E','Q'), ('H','S'), ('N','O'), ('C','O'), ('E','S'),
    ('J','K'), ('N','Q'), ('D','H'), ('F','O'), ('J','L'), ('Q','R'),
    ('D','I'), ('F','P'), ('J','N'), ('Q','S'), ('D','J'), ('F','Q'),
    ('J','Q')
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def calculate_distance(p1, p2, coordinates):
    if p1 in coordinates and p2 in coordinates:
        x1, y1 = coordinates[p1]
        x2, y2 = coordinates[p2]
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return None

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

def extract_distances(images):
    result = []
    for img in images:
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        coords = {}
        for face in faces:
            lm = predictor(gray, face)
            for n in range(68):
                if (n + 1) in KEY_POINT_LABELS:
                    coords[KEY_POINT_LABELS[n + 1]] = (lm.part(n).x, lm.part(n).y)
        D = []
        for p1, p2 in POINT_PAIRS:
            d = calculate_distance(p1, p2, coords)
            D.append(d if d is not None else 0.0)
        result.append(D)
    return np.array(result)

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

# ── Step 2: Extract distances ─────────────────────────────────────────────────
print("Extracting landmark distances …")
x_train_d = extract_distances(x_train)
x_valid_d = extract_distances(x_valid)
x_test_d  = extract_distances(x_test)

# ── Step 3: Normalise ─────────────────────────────────────────────────────────
x_train   = x_train   / 255.0
x_valid   = x_valid   / 255.0
x_test    = x_test    / 255.0
x_train_d = x_train_d / 255.0
x_valid_d = x_valid_d / 255.0
x_test_d  = x_test_d  / 255.0

print("Train:", x_train.shape, x_train_d.shape)
print("Valid:", x_valid.shape, x_valid_d.shape)
print("Test :", x_test.shape,  x_test_d.shape)

# ── Step 4: Build GDFN and run experiments ────────────────────────────────────
models_to_test = ['MobileNetV1', 'MobileNetV2', 'InceptionV3', 'DenseNet121', 'Xception']

for model_name in models_to_test:
    print(f"\n{'='*60}\nGDFN — {model_name}\n{'='*60}")

    backbone           = load_backbone(model_name)
    backbone.trainable = True

    # Image branch
    img_input    = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='image_input')
    img_features = backbone(img_input)
    img_features = Flatten()(img_features)
    img_features = Dense(512, activation='relu')(img_features)
    img_features = Dropout(0.4)(img_features)
    img_out      = Dense(256, activation='relu')(img_features)

    # Distance branch
    dist_input = Input(shape=(31,), name='distance_input')
    dist_feat  = Dense(256, activation='relu')(dist_input)
    dist_feat  = Dropout(0.4)(dist_feat)
    dist_out   = Dense(128, activation='relu')(dist_feat)

    # Fusion
    fused  = concatenate([img_out, dist_out])
    fused  = Dense(256, activation='relu')(fused)
    fused  = Dropout(0.5)(fused)
    output = Dense(2, activation='softmax')(fused)

    model = Model(inputs=[img_input, dist_input], outputs=output,
                  name=f'GDFN_{model_name}')
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(learning_rate=0.0001),
                  metrics=['accuracy'])
    model.summary()

    model.fit(
        [x_train, x_train_d], y_train,
        batch_size=32,
        epochs=70,
        validation_data=([x_valid, x_valid_d], y_valid),
        shuffle=True
    )

    save_path = os.path.join(MODEL_SAVE_DIR, f'GDFN_{model_name}.h5')
    model.save(save_path)
    print(f"Saved: {save_path}")

    y_pred = np.argmax(model.predict([x_test, x_test_d]), axis=1)

    print(f"\nResults — GDFN {model_name}:")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=target_names))
    cm           = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = cm.ravel()
    print("Sensitivity:", TP / (TP + FN) if (TP + FN) > 0 else 0)
    print("Specificity:", TN / (TN + FP) if (TN + FP) > 0 else 0)
