"""Conditional within-cycle randomization: class labels are NOT shuffled."""
import numpy as np
import hashlib
from scipy.stats import binomtest
from .models import Model,permute_block,get_cycles
from .storage import csv_write
from .protocol import paired

def residual(mu):
    # mu: cycles x bins x channels x states. All calculations in float64.
    mu=np.asarray(mu,float);L=mu.shape[1];O=mu.mean(1)
    M=np.einsum('nlra,nlrb->nrab',mu[:,:-1],mu[:,1:])/(L-1)
    E=(L*L*np.einsum('nra,nrb->nrab',O,O)-np.einsum('nlra,nlrb->nrab',mu,mu))/(L*(L-1))
    return M-E

def statistic(mu):return float(np.abs(residual(mu).mean(0)).sum((-2,-1)).mean())

def randomize(mu,rng):
    ix=np.argsort(rng.random(mu.shape[:2]),axis=1)
    return np.take_along_axis(mu,ix[:,:,None,None],axis=1)

def bh(p):
    p=np.asarray(p,float);order=np.argsort(p);q=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1];out=np.empty_like(q);out[order]=np.minimum(q,1);return out

def synthetic_calibration(cfg,store,folder):
    rows=[]
    for L in cfg['calibration_lengths']:
        for kind in ['random','structured']:
            for rep in range(cfg['calibration_repetitions']):
                def build(L=L,kind=kind,rep=rep):
                    rng=np.random.default_rng(cfg['seed']+L*100000+rep);states=np.arange(L)%4
                    states=rng.permutation(states) if kind=='random' else np.sort(states)
                    mu=np.eye(4)[states][None,:,None,:];obs=statistic(mu)
                    null=np.array([statistic(randomize(mu,rng)) for _ in range(cfg['null_permutations'])])
                    p=(1+np.sum(null>=obs-1e-12))/(len(null)+1)
                    return {'data_mode':'synthetic_control','length':L,'control':kind,'rep':rep,'statistic':obs,'p_value':float(p),'reject':int(p<=cfg['alpha'])}
                rows.append(store.task(f'calibration/{L}/{kind}/{rep}',build))
    csv_write(folder/'synthetic_null_power_raw.csv',rows);agg=[]
    for L in cfg['calibration_lengths']:
        for kind in ['random','structured']:
            subset=[r for r in rows if r['length']==L and r['control']==kind];k=sum(r['reject'] for r in subset);ci=binomtest(k,len(subset)).proportion_ci()
            agg.append({'data_mode':'synthetic_control','length':L,'control':kind,'rejection_rate':k/len(subset),'lo':ci.low,'hi':ci.high,'repetitions':len(subset),'meaning':'false positive rate' if kind=='random' else 'power for this structured alternative'})
    csv_write(folder/'synthetic_calibration.csv',agg)

def measured_order(frames,results,cfg,store,folder):
    rows=[];null_rows=[]
    for (fold,arm),entry in results.items():
        if arm!='phase_ordered':continue
        model=entry['model'];fs=[frames[i] for i in entry['test']]
        for subject in sorted({f['subject'] for f in fs}):
            subset=[f for f in fs if f['subject']==subject]
            def build(subset=subset,model=model,subject=subject):
                mu=[model.cyc_sem.transform(get_cycles(f,model.c['B'],cfg['max_cycles'])) for f in subset]
                obs=np.mean([statistic(m) for m in mu]);rng=np.random.default_rng(cfg['seed']+int.from_bytes(hashlib.sha256(subject.encode()).digest()[:4],'little'))
                null=[float(np.mean([statistic(randomize(m,rng)) for m in mu])) for _ in range(cfg['null_permutations'])]
                p=(1+sum(v>=obs-1e-12 for v in null))/(len(null)+1)
                return {'subject':subject,'fold':fold,'species':subset[0]['y'],'statistic':float(obs),'p_value':p,'segments':len(subset),'null':null}
            r=store.task(f'order/subject/{fold}/{subject}',build);rows.append({k:v for k,v in r.items() if k!='null'})
            null_rows.extend({'subject':subject,'fold':fold,'draw':j,'statistic':v} for j,v in enumerate(r['null']))
    if rows:
        for r,q in zip(rows,bh([r['p_value'] for r in rows])):r['q_value_bh']=float(q)
    csv_write(folder/'order_subject_tests.csv',rows);csv_write(folder/'order_null_statistics.csv',null_rows)

def permutation_classification(frames,results,cfg,store,folder):
    allrows=[];deltas=[]
    for rep in range(cfg['permutation_repetitions']):
        permrows=[];ordered=[]
        for (fold,arm),entry in results.items():
            if arm!='phase_ordered':continue
            def build(entry=entry,fold=fold):
                model=Model(entry['candidate'],cfg,perm_seed=cfg['seed']+rep+1).fit([frames[i] for i in entry['train']]);fs=[frames[i] for i in entry['test']];pred=model.predict(fs)
                return [{'id':f['id'],'subject':f['subject'],'y':f['y'],'pred':str(p),'fold':fold,'permutation_rep':rep} for f,p in zip(fs,pred)]
            rows=store.task(f'perm_classifier/{rep}/{fold}',build);permrows.extend(rows);ordered.extend(entry['rows'])
        if ordered:
            deltas.append({'rep':rep,'scope':'fixed inner-selected ordered configuration; refit permuted train','a':'phase_ordered','b':'permuted',**paired(ordered,permrows,cfg['bootstrap'],cfg['seed'])})
        allrows.extend(permrows)
    csv_write(folder/'permutation_predictions.csv',allrows);csv_write(folder/'permutation_paired_deltas.csv',deltas)
