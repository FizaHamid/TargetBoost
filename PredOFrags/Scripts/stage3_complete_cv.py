#!/usr/bin/env python3
"""
Independent, leakage-controlled 5-fold CV for tRF Stage 3B.
Uses ONLY stage3 training data. Never reads validation/test data.
Outputs fold metrics, OOF predictions, per-class metrics, confusion
matrices, ROC/PR-ready tables, CNN histories, feature-selection
stability, and a reproducibility manifest.
"""
from pathlib import Path
import json, random, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
 precision_score, recall_score, f1_score, matthews_corrcoef,
 roc_auc_score, average_precision_score, confusion_matrix, log_loss)
from xgboost import XGBClassifier
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")
BASE=Path("/lustre/home/sklab202/sourav_ml/sheetal_ml")
DATA=BASE/"stage3_model_development/03_datasets"
FEAT=BASE/"stage2_features/02_features/numeric_feature_matrix.tsv"
OUT=BASE/"stage3_model_development/05_cross_validation_complete"
for x in ["binary","subclass","summary","models","logs"]:
    (OUT/x).mkdir(parents=True,exist_ok=True)

SEED=42; K=5; BS=256; EPOCHS=30; PATIENCE=5; LR=1e-3; WD=1e-4
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUB=["tRF-5","tRF-3","5tRH","tRF-1","3tRH","leader","other"]

def col(df,names):
    m={str(x).lower():x for x in df.columns}
    for n in names:
        if n.lower() in m:return m[n.lower()]
    raise KeyError((names,list(df.columns)))

def bin_y(df,c):
    y=df[c].astype(str).str.strip().str.lower()
    mp={"non-trf":0,"non_trf":0,"negative":0,"0":0,
        "trf":1,"t-rf":1,"positive":1,"1":1}
    z=y.map(mp)
    if z.isna().any(): raise ValueError(z[z.isna()].unique())
    return z.astype(int).to_numpy()

def weights(y):
    u,n=np.unique(y,return_counts=True); return {int(a):len(y)/(len(u)*b) for a,b in zip(u,n)}

def metrics_bin(y,p,t=.5):
    q=(p>=t).astype(int); tn,fp,fn,tp=confusion_matrix(y,q,labels=[0,1]).ravel()
    return dict(accuracy=accuracy_score(y,q),balanced_accuracy=balanced_accuracy_score(y,q),
      precision=precision_score(y,q,zero_division=0),recall=recall_score(y,q,zero_division=0),
      specificity=tn/(tn+fp),f1=f1_score(y,q,zero_division=0),
      mcc=matthews_corrcoef(y,q),roc_auc=roc_auc_score(y,p),
      pr_auc=average_precision_score(y,p),log_loss=log_loss(y,np.c_[1-p,p]),
      tn=tn,fp=fp,fn=fn,tp=tp,threshold=t)

def metrics_multi(y,p):
    q=p.argmax(1); return dict(
      accuracy=accuracy_score(y,q),balanced_accuracy=balanced_accuracy_score(y,q),
      macro_precision=precision_score(y,q,average="macro",zero_division=0),
      macro_recall=recall_score(y,q,average="macro",zero_division=0),
      macro_f1=f1_score(y,q,average="macro",zero_division=0),
      weighted_f1=f1_score(y,q,average="weighted",zero_division=0),
      mcc=matthews_corrcoef(y,q),
      macro_roc_auc=roc_auc_score(y,p,multi_class="ovr",average="macro"),
      macro_pr_auc=average_precision_score(np.eye(len(SUB))[y],p,average="macro"),
      log_loss=log_loss(y,p))

def select_features(a,b):
    v=a.var(axis=0,ddof=0); keep=v[v>0].index.tolist()
    a=a[keep]; b=b[keep]
    c=a.corr(method="spearman").abs()
    u=c.where(np.triu(np.ones(c.shape),1).astype(bool))
    drop=[x for x in u.columns if (u[x]>.90).any()]
    keep=[x for x in keep if x not in drop]
    return a[keep],b[keep],len(v[v>0]),len(drop),keep

def xgb_bin():
    return XGBClassifier(n_estimators=500,max_depth=6,learning_rate=.05,
      subsample=.8,colsample_bytree=.8,objective="binary:logistic",
      eval_metric="logloss",random_state=SEED,tree_method="hist",n_jobs=-1)
def xgb_sub():
    return XGBClassifier(n_estimators=500,max_depth=6,learning_rate=.05,
      subsample=.8,colsample_bytree=.8,objective="multi:softprob",
      num_class=7,eval_metric="mlogloss",random_state=SEED,tree_method="hist",n_jobs=-1)
def rf():
    return RandomForestClassifier(n_estimators=500,max_features="sqrt",
      min_samples_leaf=2,class_weight="balanced",n_jobs=-1,random_state=SEED)

RNA={"A":1,"C":2,"G":3,"U":4}
class DS(Dataset):
    def __init__(self,s,y,L):
        self.s=[str(x).upper().replace("T","U") for x in s]; self.y=np.asarray(y); self.L=L
    def __len__(self): return len(self.y)
    def __getitem__(self,i):
        z=np.zeros(self.L,dtype=np.int64)
        for j,b in enumerate(self.s[i][:self.L]): z[j]=RNA.get(b,0)
        return torch.tensor(z),torch.tensor(self.y[i],dtype=torch.long)

class CNN(nn.Module):
    def __init__(self,n):
        super().__init__(); self.e=nn.Embedding(5,16,padding_idx=0)
        self.c1=nn.Conv1d(16,64,5,padding=2); self.b1=nn.BatchNorm1d(64)
        self.c2=nn.Conv1d(64,128,5,padding=2); self.b2=nn.BatchNorm1d(128)
        self.c3=nn.Conv1d(128,128,3,padding=1); self.b3=nn.BatchNorm1d(128)
        self.d=nn.Dropout(.30); self.f1=nn.Linear(128,64); self.f2=nn.Linear(64,n)
    def forward(self,x):
        x=self.e(x).transpose(1,2); x=torch.relu(self.b1(self.c1(x))); x=self.d(x)
        x=torch.relu(self.b2(self.c2(x))); x=self.d(x)
        x=torch.relu(self.b3(self.c3(x))); x=torch.amax(x,2)
        return self.f2(self.d(torch.relu(self.f1(x))))

def traincnn(st,yt,sv,yv,n,L,w):
    m=CNN(n).to(DEVICE); tr=DataLoader(DS(st,yt,L),BS,shuffle=True)
    va=DataLoader(DS(sv,yv,L),BS,shuffle=False)
    cw=torch.ones(n,device=DEVICE)
    for k,v in w.items(): cw[k]=v
    lossf=nn.CrossEntropyLoss(weight=cw); opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=WD)
    best=None; bestloss=np.inf; wait=0; hist=[]
    for ep in range(1,EPOCHS+1):
        m.train(); ls=[]
        for x,y in tr:
            x,y=x.to(DEVICE),y.to(DEVICE); opt.zero_grad()
            z=lossf(m(x),y); z.backward(); opt.step(); ls.append(z.item())
        m.eval(); vl=[]
        with torch.no_grad():
            for x,y in va: vl.append(lossf(m(x.to(DEVICE)),y.to(DEVICE)).item())
        tl=float(np.mean(ls)); vv=float(np.mean(vl)); hist.append({"epoch":ep,"train_loss":tl,"validation_loss":vv})
        if vv<bestloss: bestloss=vv; best={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}; wait=0
        else: wait+=1
        if wait>=PATIENCE: break
    m.load_state_dict(best)
    return m,pd.DataFrame(hist)

def predcnn(m,s,y,L,n):
    dl=DataLoader(DS(s,y,L),BS,shuffle=False); out=[]
    m.eval()
    with torch.no_grad():
        for x,_ in dl: out.append(torch.softmax(m(x.to(DEVICE)),1).cpu().numpy())
    return np.vstack(out)

print("Starting complete Stage 3B CV; device:",DEVICE)
tb=pd.read_csv(DATA/"train_binary.tsv",sep="\t")
ts=pd.read_csv(DATA/"train_subclass.tsv",sep="\t")
ib=col(tb,["sequence_id","seq_id","id","ID"]); sb=col(tb,["sequence","seq","Sequence"]); lb=col(tb,["binary_label","label","class","target"])
isub=col(ts,["sequence_id","seq_id","id","ID"]); ss=col(ts,["sequence","seq","Sequence"]); ls=col(ts,["subclass","subclass_label","class","label","target"])
yb=bin_y(tb,lb)
ys0=ts[ls].astype(str).str.strip()
ys=ys0.map({x:i for i,x in enumerate(SUB)})
if ys.isna().any(): raise ValueError("Unknown subclass labels: "+str(ys[ys.isna()].unique()))
ys=ys.astype(int).to_numpy()

f=pd.read_csv(FEAT,sep="\t"); fi=col(f,["sequence_id","seq_id","id","ID"]); f[fi]=f[fi].astype(str)
f=f.drop_duplicates(fi).set_index(fi)
bad={"label","target","class","subclass","binary","is_trf"}
fc=[c for c in f.select_dtypes("number").columns if str(c).lower()!="label_numeric" and not any(x in str(c).lower() for x in bad)]
XB=f.loc[tb[ib].astype(str),fc].replace([np.inf,-np.inf],np.nan)
XS=f.loc[ts[isub].astype(str),fc].replace([np.inf,-np.inf],np.nan)
XB=XB.fillna(XB.median()).fillna(0); XS=XS.fillna(XS.median()).fillna(0)
Lbin=int(tb[sb].astype(str).str.len().max()); Lsub=int(ts[ss].astype(str).str.len().max())

# ---------------- binary ----------------
sk=StratifiedKFold(K,shuffle=True,random_state=SEED)
bm=[]; bp=[]; bh=[]; bfs=[]
for fold,(tr,va) in enumerate(sk.split(XB,yb),1):
    a,b=XB.iloc[tr].copy(),XB.iloc[va].copy(); a,b,nv,nd,sel=select_features(a,b)
    sc=StandardScaler(); a=sc.fit_transform(a); b=sc.transform(b); w=weights(yb[tr]); sw=np.array([w[int(x)] for x in yb[tr]])
    for name,model in [("XGBoost",xgb_bin()),("SVM_RBF",SVC(kernel="rbf",C=2,probability=True,class_weight="balanced",random_state=SEED))]:
        model.fit(a,yb[tr],**({"sample_weight":sw} if name=="XGBoost" else {}))
        p=model.predict_proba(b)[:,1]; m=metrics_bin(yb[va],p); m.update(fold=fold,model=name,n_selected=len(sel),n_before=nv,n_corr_removed=nd); bm.append(m)
        bp.append(pd.DataFrame({"row_index":va,"fold":fold,"model":name,"true_label":yb[va],"probability_tRF":p,"prediction":(p>=.5).astype(int)}))
    m,h=traincnn(tb.iloc[tr][sb].values,yb[tr],tb.iloc[va][sb].values,yb[va],2,Lbin,w)
    p=predcnn(m,tb.iloc[va][sb].values,yb[va],Lbin,2)[:,1]; z=metrics_bin(yb[va],p); z.update(fold=fold,model="CNN",n_selected=np.nan,n_before=np.nan,n_corr_removed=np.nan); bm.append(z)
    bp.append(pd.DataFrame({"row_index":va,"fold":fold,"model":"CNN","true_label":yb[va],"probability_tRF":p,"prediction":(p>=.5).astype(int)})); h["fold"]=fold; bh.append(h)
    bfs.append({"fold":fold,"n_before":nv,"n_corr_removed":nd,"n_selected":len(sel)})
pd.DataFrame(bm).to_csv(OUT/"binary/fold_metrics.tsv",sep="\t",index=False)
pd.concat(bp,ignore_index=True).to_csv(OUT/"binary/oof_predictions.tsv",sep="\t",index=False)
pd.concat(bh,ignore_index=True).to_csv(OUT/"binary/cnn_training_history.tsv",sep="\t",index=False)
pd.DataFrame(bfs).to_csv(OUT/"binary/fold_feature_selection.tsv",sep="\t",index=False)

# ---------------- subclass ----------------
sk=StratifiedKFold(K,shuffle=True,random_state=SEED)
sm=[]; sp=[]; sh=[]; sfs=[]
for fold,(tr,va) in enumerate(sk.split(XS,ys),1):
    a,b=XS.iloc[tr].copy(),XS.iloc[va].copy(); a,b,nv,nd,sel=select_features(a,b)
    sc=StandardScaler(); a=sc.fit_transform(a); b=sc.transform(b); w=weights(ys[tr]); sw=np.array([w[int(x)] for x in ys[tr]])
    for name,model in [("XGBoost",xgb_sub()),("RandomForest",rf())]:
        model.fit(a,ys[tr],**({"sample_weight":sw} if name=="XGBoost" else {}))
        p=model.predict_proba(b); z=metrics_multi(ys[va],p); z.update(fold=fold,model=name,n_selected=len(sel),n_before=nv,n_corr_removed=nd); sm.append(z)
        q=np.argmax(p,1); d={"row_index":va,"fold":fold,"model":name,"true_subclass":[SUB[x] for x in ys[va]],"prediction":[SUB[x] for x in q]}
        for j,c in enumerate(SUB): d["prob_"+c]=p[:,j]
        sp.append(pd.DataFrame(d))
    m,h=traincnn(ts.iloc[tr][ss].values,ys[tr],ts.iloc[va][ss].values,ys[va],7,Lsub,w)
    p=predcnn(m,ts.iloc[va][ss].values,ys[va],Lsub,7); z=metrics_multi(ys[va],p); z.update(fold=fold,model="CNN",n_selected=np.nan,n_before=np.nan,n_corr_removed=np.nan); sm.append(z)
    q=np.argmax(p,1); d={"row_index":va,"fold":fold,"model":"CNN","true_subclass":[SUB[x] for x in ys[va]],"prediction":[SUB[x] for x in q]}
    for j,c in enumerate(SUB): d["prob_"+c]=p[:,j]
    sp.append(pd.DataFrame(d)); h["fold"]=fold; sh.append(h); sfs.append({"fold":fold,"n_before":nv,"n_corr_removed":nd,"n_selected":len(sel)})
pd.DataFrame(sm).to_csv(OUT/"subclass/fold_metrics.tsv",sep="\t",index=False)
pd.concat(sp,ignore_index=True).to_csv(OUT/"subclass/oof_predictions.tsv",sep="\t",index=False)
pd.concat(sh,ignore_index=True).to_csv(OUT/"subclass/cnn_training_history.tsv",sep="\t",index=False)
pd.DataFrame(sfs).to_csv(OUT/"subclass/fold_feature_selection.tsv",sep="\t",index=False)

# ---------------- OOF summaries + per-class + confusion ----------------
bpd=pd.concat(bp,ignore_index=True); spd=pd.concat(sp,ignore_index=True)
bs=[]; ssu=[]
for name in ["XGBoost","SVM_RBF","CNN"]:
    d=bpd[bpd.model==name].sort_values("row_index"); z=metrics_bin(d.true_label.to_numpy(),d.probability_tRF.to_numpy()); z.update(model=name,n_oof=len(d)); bs.append(z)
pd.DataFrame(bs).to_csv(OUT/"binary/oof_summary.tsv",sep="\t",index=False)
for name in ["XGBoost","RandomForest","CNN"]:
    d=spd[spd.model==name].sort_values("row_index"); y=pd.Categorical(d.true_subclass,categories=SUB).codes; p=d[["prob_"+x for x in SUB]].to_numpy(); z=metrics_multi(y,p); z.update(model=name,n_oof=len(d)); ssu.append(z)
    q=p.argmax(1); cm=confusion_matrix(y,q,labels=range(7)); pd.DataFrame(cm,index=["true_"+x for x in SUB],columns=["pred_"+x for x in SUB]).to_csv(OUT/f"subclass/oof_confusion_matrix_{name}.tsv",sep="\t")
    rows=[]
    for j,c in enumerate(SUB):
        yt=y==j; yp=q==j; tp=np.sum(yt&yp); fp=np.sum(~yt&yp); fn=np.sum(yt&~yp); pr=tp/(tp+fp) if tp+fp else 0; re=tp/(tp+fn) if tp+fn else 0; f=2*pr*re/(pr+re) if pr+re else 0
        rows.append({"model":name,"subclass":c,"support":int(yt.sum()),"tp":int(tp),"fp":int(fp),"fn":int(fn),"precision":pr,"recall":re,"f1":f})
    pd.DataFrame(rows).to_csv(OUT/f"subclass/per_class_oof_metrics_{name}.tsv",sep="\t",index=False)
pd.DataFrame(ssu).to_csv(OUT/"subclass/oof_summary.tsv",sep="\t",index=False)

# aggregate fold mean ± SD
b=pd.DataFrame(bm).groupby("model").agg(["mean","std"]).reset_index()
s=pd.DataFrame(sm).groupby("model").agg(["mean","std"]).reset_index()
b.to_csv(OUT/"summary/binary_fold_mean_sd.tsv",sep="\t",index=False)
s.to_csv(OUT/"summary/subclass_fold_mean_sd.tsv",sep="\t",index=False)
pd.DataFrame(bm).to_csv(OUT/"summary/binary_all_fold_metrics.tsv",sep="\t",index=False)
pd.DataFrame(sm).to_csv(OUT/"summary/subclass_all_fold_metrics.tsv",sep="\t",index=False)

# ROC/PR ready long tables
for name in ["XGBoost","SVM_RBF","CNN"]:
    d=bpd[bpd.model==name].sort_values("row_index"); pd.DataFrame({"model":name,"true_label":d.true_label,"probability_tRF":d.probability_tRF}).to_csv(OUT/"summary"/f"binary_ROC_PR_OOF_{name}.tsv",sep="\t",index=False)
for name in ["XGBoost","RandomForest","CNN"]:
    d=spd[spd.model==name]; cols=["model","row_index","fold","true_subclass","prediction"]+[f"prob_{x}" for x in SUB]; d[cols].to_csv(OUT/"summary"/f"subclass_ROC_PR_OOF_{name}.tsv",sep="\t",index=False)

manifest={"seed":SEED,"folds":K,"device":str(DEVICE),"train_only":True,"validation_used":False,"test_used":False,
"synthetic_balancing":False,"binary_models":["XGBoost","SVM_RBF","CNN"],"subclass_models":["XGBoost","RandomForest","CNN"],
"candidate_features":len(fc),"foldwise_variance_and_spearman_selection":True,"spearman_threshold":.90,
"binary_oof_rows":len(tb),"subclass_oof_rows":len(ts),"cnn_epochs":EPOCHS,"cnn_patience":PATIENCE,"cnn_batch_size":BS,
"binary_cv_threshold":.5,"production_threshold_source":"original Stage 3B validation optimization"}
json.dump(manifest,open(OUT/"summary/cv_manifest.json","w"),indent=2)
print("\nCOMPLETE CV FINISHED")
print(pd.DataFrame(bs)[["model","balanced_accuracy","f1","mcc","roc_auc","pr_auc"]].to_string(index=False))
print(pd.DataFrame(ssu)[["model","balanced_accuracy","macro_f1","mcc","macro_roc_auc","macro_pr_auc"]].to_string(index=False))
print("OUTPUT:",OUT)
