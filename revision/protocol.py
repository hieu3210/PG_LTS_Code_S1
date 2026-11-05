"""Subject-stratified splits and fixed-label clustered uncertainty."""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix,recall_score,precision_recall_fscore_support

def split_subjects(frames,indices,n_splits,seed):
    indices=np.array(indices,int);subjects=sorted({frames[i]['subject'] for i in indices})
    labels=[]
    for s in subjects:
        labs={frames[i]['y'] for i in indices if frames[i]['subject']==s}
        if len(labs)!=1:raise ValueError('One subject has multiple labels.')
        labels.append(next(iter(labs)))
    labels=np.array(labels)
    if min(sum(labels==c) for c in np.unique(labels))<n_splits:
        raise ValueError(f'Not enough subjects per species for {n_splits} folds.')
    result=[]
    for tr,te in StratifiedKFold(n_splits,shuffle=True,random_state=seed).split(subjects,labels):
        train=set(np.array(subjects)[tr]);test=set(np.array(subjects)[te]);assert not train&test
        a=np.array([i for i in indices if frames[i]['subject'] in train]);b=np.array([i for i in indices if frames[i]['subject'] in test])
        assert set(frames[i]['y'] for i in a)==set(labels)
        result.append((a,b))
    return result

def score(frames,pred,labels):return float(f1_score([x['y'] for x in frames],pred,labels=labels,average='macro',zero_division=0))

def subject_rows(rows):
    result=[]
    for subject in sorted({r['subject'] for r in rows}):
        sub=[r for r in rows if r['subject']==subject];labs=sorted({r['pred'] for r in sub})
        # Deterministic lexicographic tie-break, reported in guide.
        pred=max(labs,key=lambda c:sum(r['pred']==c for r in sub))
        result.append({'subject':subject,'y':sub[0]['y'],'pred':pred,'segments':len(sub)})
    return result

class SubjectBootstrap:
    def __init__(self,rows):
        self.indices={};self.by_class={}
        for i,r in enumerate(rows):
            self.indices.setdefault(r['subject'],[]).append(i)
            self.by_class.setdefault(r['y'],set()).add(r['subject'])
        self.classes=[sorted(self.by_class[c]) for c in sorted(self.by_class)]
    def draw(self,rng):
        return np.concatenate([self.indices[s] for subjects in self.classes for s in rng.choice(subjects,len(subjects),replace=True)]).astype(int)

def clustered_draw(rows,rng):
    return SubjectBootstrap(rows).draw(rng)


def summary(rows,n_boot,seed):
    labels=sorted({r['y'] for r in rows});rng=np.random.default_rng(seed);out=[]
    for level,data in [('segment',rows),('subject',subject_rows(rows))]:
        y=np.array([r['y'] for r in data]);p=np.array([r['pred'] for r in data]);draws=[];per_class=[];sampler=SubjectBootstrap(data)
        for _ in range(n_boot):
            ix=sampler.draw(rng);draws.append(f1_score(y[ix],p[ix],labels=labels,average='macro',zero_division=0));per_class.append(np.array(precision_recall_fscore_support(y[ix],p[ix],labels=labels,zero_division=0)[:3]))
        lo,hi=np.quantile(draws,[.025,.975]);out.append({'level':level,'metric':'macro_f1','point':float(f1_score(y,p,labels=labels,average='macro',zero_division=0)),'lo':lo,'hi':hi})
        point_class=precision_recall_fscore_support(y,p,labels=labels,zero_division=0)[:3]
        for j,c in enumerate(labels):
            for m,metric in enumerate(['precision','recall','f1']):
                q=np.quantile(np.array(per_class)[:,m,j],[.025,.975]);mask=y==c
                out.append({'level':level,'metric':metric,'species':c,'point':float(point_class[m][j]),'correct':int(np.sum(p[mask]==c)),'total':int(mask.sum()),'predicted_total':int(np.sum(p==c)),'lo':q[0],'hi':q[1]})
    return out

def paired(a,b,n_boot,seed,level="segment"):
    if level=="subject":
        a=[dict(r,id=r["subject"]) for r in subject_rows(a)]
        b=[dict(r,id=r["subject"]) for r in subject_rows(b)]
    def key(r):return r['id']
    aa={key(r):r for r in a};bb={key(r):r for r in b}
    if set(aa)!=set(bb):raise ValueError('Paired comparison requires identical segment IDs.')
    ids=sorted(aa);a=[aa[k] for k in ids];b=[bb[k] for k in ids]
    assert all(x['y']==y['y'] and x['subject']==y['subject'] for x,y in zip(a,b))
    labels=sorted({r['y'] for r in a});y=np.array([r['y'] for r in a]);pa=np.array([r['pred'] for r in a]);pb=np.array([r['pred'] for r in b]);rng=np.random.default_rng(seed)
    def delta(ix):return f1_score(y[ix],pa[ix],labels=labels,average='macro',zero_division=0)-f1_score(y[ix],pb[ix],labels=labels,average='macro',zero_division=0)
    sampler=SubjectBootstrap(a)
    d=[delta(sampler.draw(rng)) for _ in range(n_boot)];lo,hi=np.quantile(d,[.025,.975])
    return {'level':level,'delta_macro_f1':float(delta(np.arange(len(a)))),'lo':lo,'hi':hi,'subjects':len({r['subject'] for r in a}),'interval_scope':'fixed OOF models; species-stratified subject-cluster bootstrap'}
