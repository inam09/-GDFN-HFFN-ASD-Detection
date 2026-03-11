"""
GDFN — Geometric Distance Feature Network
PyTorch implementation — Attention Dataset with 5-Fold Cross-Validation
Backbones: ViT-B/16  |  Swin-B/4  (pretrained via timm)

Feature branch : Dlib 68-point facial landmark distances (31 pairs)
Training       : 80/20 split -> 5-Fold CV on 80%  |  Augmentation: rotation-90, zoom-20%, h-flip

Dataset : Attention (Eye-Tracking / Gaze Attention Dataset)
Classes : ASD | TD

Install: pip install timm torch torchvision dlib opencv-python scikit-learn
Usage  : python GDFN_ViT_Swin_Attention.py
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
from sklearn.model_selection import KFold, train_test_split

IMG_SIZE       = 224
DATADIR        = 'test-face-vs1'
CARTEGORIES    = ['ASD', 'TD']
MODEL_SAVE_DIR = 'saved_models/GDFN'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
target_names   = ['ASD', 'TD']
device         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

KEY_POINT_LABELS = {
    18:'C',37:'D',41:'E',40:'F',28:'B',43:'G',47:'H',46:'I',
    27:'J',32:'K',34:'L',36:'M',51:'O',52:'P',53:'Q',55:'R',9:'S',49:'N'
}
POINT_PAIRS = [
    ('B','S'),('E','J'),('G','J'),('L','S'),('C','H'),('E','N'),
    ('G','O'),('M','N'),('C','K'),('E','P'),('G','S'),('M','S'),
    ('C','L'),('E','Q'),('H','S'),('N','O'),('C','O'),('E','S'),
    ('J','K'),('N','Q'),('D','H'),('F','O'),('J','L'),('Q','R'),
    ('D','I'),('F','P'),('J','N'),('Q','S'),('D','J'),('F','Q'),('J','Q')
]

def extract_distances(images):
    result=[]
    for img in images:
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY); faces=detector(gray); coords={}
        for face in faces:
            lm=predictor(gray,face)
            for n in range(68):
                if (n+1) in KEY_POINT_LABELS: coords[KEY_POINT_LABELS[n+1]]=(lm.part(n).x,lm.part(n).y)
        D=[]
        for p1,p2 in POINT_PAIRS:
            if p1 in coords and p2 in coords:
                x1,y1=coords[p1]; x2,y2=coords[p2]; D.append(np.sqrt((x2-x1)**2+(y2-y1)**2))
            else: D.append(0.0)
        result.append(D)
    return np.array(result)

class DualInputDataset(Dataset):
    def __init__(self,images,features,labels,augment=False):
        self.images=images.astype(np.float32); self.features=features.astype(np.float32)
        self.labels=labels.astype(np.int64); self.augment=augment
        self.norm=transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        self.aug=transforms.Compose([transforms.RandomRotation(90),transforms.RandomHorizontalFlip(),
                                     transforms.RandomResizedCrop(IMG_SIZE,scale=(0.8,1.0))])
    def __len__(self): return len(self.labels)
    def __getitem__(self,idx):
        img=torch.from_numpy(self.images[idx,:,:,::-1].copy()).permute(2,0,1)
        if self.augment: img=self.aug(img)
        return self.norm(img),torch.from_numpy(self.features[idx]),torch.tensor(self.labels[idx],dtype=torch.long)

class GDFN(nn.Module):
    def __init__(self,backbone_name,dist_dim=31,num_classes=2):
        super().__init__()
        self.backbone=timm.create_model(backbone_name,pretrained=True,num_classes=0)
        fd=self.backbone.num_features
        self.img_fc1=nn.Linear(fd,512); self.img_drop1=nn.Dropout(0.4); self.img_fc2=nn.Linear(512,256)
        self.dist_fc1=nn.Linear(dist_dim,256); self.dist_drop1=nn.Dropout(0.4); self.dist_fc2=nn.Linear(256,128)
        self.fusion_fc=nn.Linear(384,256); self.fusion_drop=nn.Dropout(0.5); self.output=nn.Linear(256,num_classes)
    def forward(self,img,dist):
        x=self.backbone(img); x=F.relu(self.img_fc1(x)); x=self.img_drop1(x); x=F.relu(self.img_fc2(x))
        y=F.relu(self.dist_fc1(dist)); y=self.dist_drop1(y); y=F.relu(self.dist_fc2(y))
        z=torch.cat([x,y],dim=1); z=F.relu(self.fusion_fc(z)); z=self.fusion_drop(z); return self.output(z)

def train_epoch(model,loader,opt,crit):
    model.train(); tl=correct=total=0
    for imgs,feats,labels in loader:
        imgs,feats,labels=imgs.to(device),feats.to(device),labels.to(device)
        opt.zero_grad(); out=model(imgs,feats); loss=crit(out,labels); loss.backward(); opt.step()
        tl+=loss.item()*imgs.size(0); correct+=(out.argmax(1)==labels).sum().item(); total+=labels.size(0)
    return tl/total, correct/total

def evaluate(model,loader):
    model.eval(); preds=[]; truths=[]
    with torch.no_grad():
        for imgs,feats,labels in loader:
            preds.extend(model(imgs.to(device),feats.to(device)).argmax(1).cpu().numpy()); truths.extend(labels.numpy())
    return np.array(preds),np.array(truths)

print("Loading images …")
x_all_raw,y_all_raw=[],[]
for cat in CARTEGORIES:
    path=os.path.join(DATADIR,cat); cn=CARTEGORIES.index(cat)
    for f in os.listdir(path):
        img=cv2.imread(os.path.join(path,f))
        if img is None: continue
        x_all_raw.append(cv2.resize(img,(IMG_SIZE,IMG_SIZE))); y_all_raw.append(cn)
x_all_raw=np.array(x_all_raw); y_all_raw=np.array(y_all_raw)
x_train,x_test,y_train,y_test=train_test_split(x_all_raw,y_all_raw,test_size=0.2,random_state=42,stratify=y_all_raw)

print("Extracting distances …")
x_train_d=extract_distances(x_train); x_test_d=extract_distances(x_test)
x_train/=255.0; x_test/=255.0; x_train_d/=255.0; x_test_d/=255.0
y_train=np.array(y_train); y_test=np.array(y_test)
x_all=x_train; x_all_d=x_train_d; y_all=y_train

test_loader=DataLoader(DualInputDataset(x_test,x_test_d,y_test),batch_size=32)
MODELS={'ViT-B16':'vit_base_patch16_224','Swin-B':'swin_base_patch4_window7_224'}

for name,timm_name in MODELS.items():
    print(f"\n{'='*60}\nGDFN Attention — {name}\n{'='*60}")
    kf=KFold(n_splits=5,shuffle=True,random_state=42); fold_no=1; fold_accs=[]
    for tr_idx,val_idx in kf.split(x_all):
        xf_tr,xf_val=x_all[tr_idx],x_all[val_idx]
        xd_tr,xd_val=x_all_d[tr_idx],x_all_d[val_idx]
        yf_tr,yf_val=y_all[tr_idx],y_all[val_idx]
        tr_loader=DataLoader(DualInputDataset(xf_tr,xd_tr,yf_tr,augment=True),batch_size=32,shuffle=True)
        val_loader=DataLoader(DualInputDataset(xf_val,xd_val,yf_val),batch_size=32)
        model=GDFN(timm_name).to(device); opt=torch.optim.Adam(model.parameters(),lr=0.0001); crit=nn.CrossEntropyLoss()
        for epoch in range(70):
            tl,ta=train_epoch(model,tr_loader,opt,crit); vp,vt=evaluate(model,val_loader)
            print(f"  Fold{fold_no} Ep{epoch+1:2d}/70 loss={tl:.4f} train={ta:.4f} val={accuracy_score(vt,vp):.4f}")
        torch.save(model.state_dict(),os.path.join(MODEL_SAVE_DIR,f'GDFN_Attn_{name}_fold{fold_no}.pt'))
        y_pred,y_true=evaluate(model,test_loader); acc=accuracy_score(y_true,y_pred); fold_accs.append(acc)
        print(f"\n  Results — GDFN Attention {name} Fold {fold_no}:")
        print("  CM:\n",confusion_matrix(y_true,y_pred)); print("  Acc:",acc)
        print(classification_report(y_true,y_pred,target_names=target_names))
        cm=confusion_matrix(y_true,y_pred); TN,FP,FN,TP=cm.ravel()
        print("  Sensitivity:",TP/(TP+FN) if (TP+FN)>0 else 0); print("  Specificity:",TN/(TN+FP) if (TN+FP)>0 else 0)
        fold_no+=1
    print(f"\nAvg Acc {name}: {np.mean(fold_accs):.4f}  Std: {np.std(fold_accs):.4f}")
