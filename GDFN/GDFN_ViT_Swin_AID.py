"""
GDFN — Geometric Distance Feature Network
PyTorch implementation — AID Dataset (train / valid / test split)
Backbones: ViT-B/16  |  Swin-B/4  (pretrained via timm)

Feature branch : Dlib 68-point facial landmark distances (31 pairs)
Image branch   : ViT-B/16 (768-d) or Swin-B (1024-d) pretrained backbone
Fusion         : Concatenation -> Dense -> Softmax (2 classes)

Dataset : AID  (Autism Image Dataset)
Classes : autistic | non_autistic

Install: pip install timm torch torchvision dlib opencv-python scikit-learn
Usage  : python GDFN_ViT_Swin_AID.py
"""

import cv2
import numpy as np
import os
import dlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ── Configuration ─────────────────────────────────────────────────────────────
IMG_SIZE       = 224
DATADIR_TRAIN  = 'AID/train'
DATADIR_VALID  = 'AID/valid'
DATADIR_TEST   = 'AID/test'
CARTEGORIES    = ['autistic', 'non_autistic']
MODEL_SAVE_DIR = 'saved_models/GDFN'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
target_names   = ['autistic', 'non_autistic']
device         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

KEY_POINT_LABELS = {
    18:'C', 37:'D', 41:'E', 40:'F', 28:'B', 43:'G', 47:'H', 46:'I',
    27:'J', 32:'K', 34:'L', 36:'M', 51:'O', 52:'P', 53:'Q', 55:'R', 9:'S', 49:'N'
}
POINT_PAIRS = [
    ('B','S'),('E','J'),('G','J'),('L','S'),('C','H'),('E','N'),
    ('G','O'),('M','N'),('C','K'),('E','P'),('G','S'),('M','S'),
    ('C','L'),('E','Q'),('H','S'),('N','O'),('C','O'),('E','S'),
    ('J','K'),('N','Q'),('D','H'),('F','O'),('J','L'),('Q','R'),
    ('D','I'),('F','P'),('J','N'),('Q','S'),('D','J'),('F','Q'),('J','Q')
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def calculate_distance(p1, p2, coords):
    if p1 in coords and p2 in coords:
        x1,y1=coords[p1]; x2,y2=coords[p2]
        return np.sqrt((x2-x1)**2+(y2-y1)**2)
    return None

def load_images(directory):
    X, Y = [], []
    for cat in CARTEGORIES:
        path = os.path.join(directory, cat); class_num = CARTEGORIES.index(cat)
        for img_name in os.listdir(path):
            img = cv2.imread(os.path.join(path, img_name))
            if img is None: continue
            X.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE))); Y.append(class_num)
    return np.array(X), np.array(Y)

def extract_distances(images):
    result = []
    for img in images:
        gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); faces=detector(gray); coords={}
        for face in faces:
            lm=predictor(gray, face)
            for n in range(68):
                if (n+1) in KEY_POINT_LABELS:
                    coords[KEY_POINT_LABELS[n+1]]=(lm.part(n).x, lm.part(n).y)
        D=[calculate_distance(p1,p2,coords) or 0.0 for p1,p2 in POINT_PAIRS]
        result.append(D)
    return np.array(result)

# ── Load & prepare data ───────────────────────────────────────────────────────
print("Loading images …")
x_train, y_train = load_images(DATADIR_TRAIN)
x_valid, y_valid = load_images(DATADIR_VALID)
x_test,  y_test  = load_images(DATADIR_TEST)

print("Extracting landmark distances …")
x_train_d = extract_distances(x_train)
x_valid_d = extract_distances(x_valid)
x_test_d  = extract_distances(x_test)

x_train=x_train/255.0; x_valid=x_valid/255.0; x_test=x_test/255.0
x_train_d=x_train_d/255.0; x_valid_d=x_valid_d/255.0; x_test_d=x_test_d/255.0
y_train=np.array(y_train); y_valid=np.array(y_valid); y_test=np.array(y_test)

# ── Dataset ───────────────────────────────────────────────────────────────────
class DualInputDataset(Dataset):
    def __init__(self, images, features, labels):
        self.images=images.astype(np.float32); self.features=features.astype(np.float32)
        self.labels=labels.astype(np.int64)
        self.normalize=transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        img=torch.from_numpy(self.images[idx,:,:,::-1].copy()).permute(2,0,1)
        img=self.normalize(img)
        return img, torch.from_numpy(self.features[idx]), torch.tensor(self.labels[idx], dtype=torch.long)

# ── GDFN Model ────────────────────────────────────────────────────────────────
class GDFN(nn.Module):
    """Geometric Distance Feature Network with ViT-B/16 or Swin-B backbone."""
    def __init__(self, backbone_name, dist_dim=31, num_classes=2):
        super().__init__()
        self.backbone  = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        feat_dim       = self.backbone.num_features
        self.img_fc1   = nn.Linear(feat_dim, 512); self.img_drop1 = nn.Dropout(0.4)
        self.img_fc2   = nn.Linear(512, 256)
        self.dist_fc1  = nn.Linear(dist_dim, 256); self.dist_drop1 = nn.Dropout(0.4)
        self.dist_fc2  = nn.Linear(256, 128)
        self.fusion_fc = nn.Linear(384, 256); self.fusion_drop = nn.Dropout(0.5)
        self.output    = nn.Linear(256, num_classes)
    def forward(self, img, dist):
        x=self.backbone(img)
        x=F.relu(self.img_fc1(x)); x=self.img_drop1(x); x=F.relu(self.img_fc2(x))
        y=F.relu(self.dist_fc1(dist)); y=self.dist_drop1(y); y=F.relu(self.dist_fc2(y))
        z=torch.cat([x,y],dim=1)
        z=F.relu(self.fusion_fc(z)); z=self.fusion_drop(z)
        return self.output(z)

# ── Training helpers ──────────────────────────────────────────────────────────
def train_epoch(model, loader, opt, crit):
    model.train(); total_loss=correct=total=0
    for imgs,feats,labels in loader:
        imgs,feats,labels=imgs.to(device),feats.to(device),labels.to(device)
        opt.zero_grad(); out=model(imgs,feats); loss=crit(out,labels)
        loss.backward(); opt.step()
        total_loss+=loss.item()*imgs.size(0); correct+=(out.argmax(1)==labels).sum().item(); total+=labels.size(0)
    return total_loss/total, correct/total

def evaluate(model, loader):
    model.eval(); preds=[]; truths=[]
    with torch.no_grad():
        for imgs,feats,labels in loader:
            out=model(imgs.to(device),feats.to(device)).argmax(1).cpu().numpy()
            preds.extend(out); truths.extend(labels.numpy())
    return np.array(preds), np.array(truths)

# ── Experiment loop ───────────────────────────────────────────────────────────
MODELS = {'ViT-B16':'vit_base_patch16_224', 'Swin-B':'swin_base_patch4_window7_224'}

train_loader=DataLoader(DualInputDataset(x_train,x_train_d,y_train), batch_size=32, shuffle=True)
valid_loader=DataLoader(DualInputDataset(x_valid,x_valid_d,y_valid), batch_size=32)
test_loader =DataLoader(DualInputDataset(x_test, x_test_d, y_test),  batch_size=32)

for name, timm_name in MODELS.items():
    print(f"\n{'='*60}\nGDFN — {name}\n{'='*60}")
    model=GDFN(timm_name).to(device)
    opt=torch.optim.Adam(model.parameters(), lr=0.0001); crit=nn.CrossEntropyLoss()
    for epoch in range(70):
        tr_loss,tr_acc=train_epoch(model,train_loader,opt,crit)
        vp,vt=evaluate(model,valid_loader); val_acc=accuracy_score(vt,vp)
        print(f"  Epoch {epoch+1:2d}/70  loss={tr_loss:.4f}  train={tr_acc:.4f}  val={val_acc:.4f}")
    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, f'GDFN_{name}.pt'))
    y_pred,y_true=evaluate(model,test_loader)
    print(f"\nResults — GDFN {name}:")
    print("Confusion Matrix:\n", confusion_matrix(y_true,y_pred))
    print("Accuracy:", accuracy_score(y_true,y_pred))
    print(classification_report(y_true,y_pred,target_names=target_names))
    cm=confusion_matrix(y_true,y_pred); TN,FP,FN,TP=cm.ravel()
    print("Sensitivity:", TP/(TP+FN) if (TP+FN)>0 else 0)
    print("Specificity:", TN/(TN+FP) if (TN+FP)>0 else 0)
