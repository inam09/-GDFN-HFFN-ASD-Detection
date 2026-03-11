"""
HFFN — Histogram Feature Fusion Network
PyTorch implementation — Attention Dataset with 5-Fold Cross-Validation
Backbones: ViT-B/16  |  Swin-B/4  (pretrained via timm)

Feature branch : SIFT descriptors (flattened to 25 600 features)
Training       : 80/20 split -> 5-Fold CV on 80%  |  Augmentation: rotation-90, zoom-20%, h-flip

Dataset : Attention (Eye-Tracking / Gaze Attention Dataset)
Classes : ASD | TD

Install: pip install timm torch torchvision opencv-contrib-python scikit-learn
Usage  : python HFFN_ViT_Swin_Attention.py
"""

import cv2, numpy as np, os, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import KFold, train_test_split

IMG_SIZE=224; DATADIR='test-face-vs1'; CARTEGORIES=['ASD','TD']
MODEL_SAVE_DIR='saved_models/HFFN'; os.makedirs(MODEL_SAVE_DIR,exist_ok=True)
target_names=['ASD','TD']; device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def extract_sift(images):
    feats=[]
    for img in images:
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY); sift=cv2.SIFT_create()
        kp,des=sift.detectAndCompute(gray,None); fd=des.flatten()
        feats.append(cv2.resize(fd,(1,25600)).flatten())
    return np.array(feats)

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

class HFFN(nn.Module):
    def __init__(self,backbone_name,sift_dim=25600,num_classes=2):
        super().__init__()
        self.backbone=timm.create_model(backbone_name,pretrained=True,num_classes=0); fd=self.backbone.num_features
        self.img_fc1=nn.Linear(fd,512); self.img_drop1=nn.Dropout(0.4); self.img_fc2=nn.Linear(512,256)
        self.sift_fc1=nn.Linear(sift_dim,256); self.sift_drop1=nn.Dropout(0.4); self.sift_fc2=nn.Linear(256,128)
        self.fusion_fc=nn.Linear(384,256); self.fusion_drop=nn.Dropout(0.5); self.output=nn.Linear(256,num_classes)
    def forward(self,img,sift):
        x=self.backbone(img); x=F.relu(self.img_fc1(x)); x=self.img_drop1(x); x=F.relu(self.img_fc2(x))
        y=F.relu(self.sift_fc1(sift)); y=self.sift_drop1(y); y=F.relu(self.sift_fc2(y))
        z=torch.cat([x,y],dim=1); z=F.relu(self.fusion_fc(z)); z=self.fusion_drop(z); return self.output(z)

def train_epoch(model,loader,opt,crit):
    model.train(); tl=correct=total=0
    for imgs,feats,labels in loader:
        imgs,feats,labels=imgs.to(device),feats.to(device),labels.to(device)
        opt.zero_grad(); out=model(imgs,feats); loss=crit(out,labels); loss.backward(); opt.step()
        tl+=loss.item()*imgs.size(0); correct+=(out.argmax(1)==labels).sum().item(); total+=labels.size(0)
    return tl/total,correct/total

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
print("Extracting SIFT …")
x_train_s=extract_sift(x_train); x_test_s=extract_sift(x_test)
x_train/=255.0; x_test/=255.0; x_train_s/=255.0; x_test_s/=255.0
y_train=np.array(y_train); y_test=np.array(y_test)
x_all=x_train; x_all_s=x_train_s; y_all=y_train
test_loader=DataLoader(DualInputDataset(x_test,x_test_s,y_test),batch_size=32)
MODELS={'ViT-B16':'vit_base_patch16_224','Swin-B':'swin_base_patch4_window7_224'}

for name,timm_name in MODELS.items():
    print(f"\n{'='*60}\nHFFN Attention — {name}\n{'='*60}")
    kf=KFold(n_splits=5,shuffle=True,random_state=42); fold_no=1; fold_accs=[]
    for tr_idx,val_idx in kf.split(x_all):
        xf_tr,xf_val=x_all[tr_idx],x_all[val_idx]; xs_tr,xs_val=x_all_s[tr_idx],x_all_s[val_idx]
        yf_tr,yf_val=y_all[tr_idx],y_all[val_idx]
        tr_ld=DataLoader(DualInputDataset(xf_tr,xs_tr,yf_tr,augment=True),batch_size=32,shuffle=True)
        val_ld=DataLoader(DualInputDataset(xf_val,xs_val,yf_val),batch_size=32)
        model=HFFN(timm_name).to(device); opt=torch.optim.Adam(model.parameters(),lr=0.0001); crit=nn.CrossEntropyLoss()
        for epoch in range(70):
            tl,ta=train_epoch(model,tr_ld,opt,crit); vp,vt=evaluate(model,val_ld)
            print(f"  Fold{fold_no} Ep{epoch+1:2d}/70 loss={tl:.4f} train={ta:.4f} val={accuracy_score(vt,vp):.4f}")
        torch.save(model.state_dict(),os.path.join(MODEL_SAVE_DIR,f'HFFN_Attn_{name}_fold{fold_no}.pt'))
        y_pred,y_true=evaluate(model,test_loader); acc=accuracy_score(y_true,y_pred); fold_accs.append(acc)
        print(f"\n  Results — HFFN Attention {name} Fold {fold_no}:")
        print("  CM:\n",confusion_matrix(y_true,y_pred)); print("  Acc:",acc)
        print(classification_report(y_true,y_pred,target_names=target_names))
        cm=confusion_matrix(y_true,y_pred); TN,FP,FN,TP=cm.ravel()
        print("  Sensitivity:",TP/(TP+FN) if (TP+FN)>0 else 0); print("  Specificity:",TN/(TN+FP) if (TN+FP)>0 else 0)
        fold_no+=1
    print(f"\nAvg Acc {name}: {np.mean(fold_accs):.4f}  Std: {np.std(fold_accs):.4f}")
