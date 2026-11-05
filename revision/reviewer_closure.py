"""Three reviewer-closure experiments. Requires authentic prepared v1 radar checkpoints.

Baseline inputs are identical phase-resolved cycles used by the PG-LTS arm (same B,
max_cycles, selection, label, and held-out unit). Aeon receives fixed-length padded
series with an explicit validity channel. Padded values are never imputed from test labels.
"""
import os,argparse,hashlib,json,time,platform
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from . import ROOT
from .raw_baselines import load_prepared
from .models import Model,get_cycles
from .phase import cycles
from .protocol import split_subjects,summary,paired
from .storage import Store,run_lock,sha,digest,json_write,csv_write
from .followup import read_csv
from .robustness import corrupt_frame


def seed_of(*args):return int.from_bytes(hashlib.sha256(':'.join(map(str,args)).encode()).digest()[:4],'little')

def tensor(frames,B,max_cycles):
    # Do not collapse within-cycle trajectories to occupancy; preserve temporal order.
    result=[]
    for f in frames:
        b=get_cycles(f,B,max_cycles)
        if len(b)<2:raise ValueError('Selected segment has fewer than two usable cycles: '+f['id'])
        n=min(len(b),max_cycles);x=np.zeros((4,max_cycles*B),np.float64)
        x[:3,:n*B]=b[:n].transpose(2,0,1).reshape(3,n*B)
        x[3,:n*B]=1.0
        result.append(x)
    return np.stack(result)

def aeon_model(name,params,seed):
    if name=='DrCIF':
        from aeon.classification.interval_based import DrCIFClassifier
        return DrCIFClassifier(n_estimators=params['n_estimators'],random_state=seed,n_jobs=int(os.environ.get('PGLTS_AEON_JOBS','1')))
    if name=='RDST':
        from aeon.classification.shapelet_based import RDSTClassifier
        return RDSTClassifier(max_shapelets=params['max_shapelets'],random_state=seed)
    raise ValueError(name)

def fingerprint_models(source):
    x=[]
    for p in sorted((source/'checkpoints').glob('outer__*__*.joblib')):
        r=p.with_suffix('.done.json')
        if r.exists() and json.loads(r.read_text())['sha256']==sha(p):x.append({'name':p.name,'sha256':sha(p)})
    return x

def run_baseline(frames,source,cfg,store,out):
    original=json.loads((source/'run_identity.json').read_text())['config']
    folds=split_subjects(frames,np.arange(len(frames)),original['outer_folds'],original['seed'])
    prior=read_csv(source/'splits.csv');selected=[];search=[];preds=[];fits=[];fold_rows=[]
    labels=sorted({f['y'] for f in frames});B=cfg['B'];N=cfg['max_cycles']
    for fold,(tr,te) in enumerate(folds):
        expected={r['subject'] for r in prior if r['level']=='outer' and int(r['outer_fold'])==fold and r['role']=='test'}
        if expected!={frames[i]['subject'] for i in te}:raise ValueError('Outer fold mismatch: '+str(fold))
        fold_rows += [{'fold':fold,'role':role,'subject':s} for role,ix in [('train',tr),('test',te)] for s in sorted({frames[i]['subject'] for i in ix})]
        inner=split_subjects(frames,tr,original['inner_folds'],original['seed']+fold+1)
        for method,options in cfg['baseline_grid'].items():
            ranking=[]
            for param in options:
                cand=digest({'method':method,'params':param})[:12];ys=[];ps=[]
                for j,(it,iv) in enumerate(inner):
                    def build(it=it,iv=iv,j=j):
                        m=aeon_model(method,param,seed_of(cfg['seed'],fold,method,cand,j))
                        m.fit(tensor([frames[i] for i in it],B,N),np.array([frames[i]['y'] for i in it]))
                        return {'y':[frames[i]['y'] for i in iv],'p':list(m.predict(tensor([frames[i] for i in iv],B,N)))}
                    r=store.task(f'matched/inner/{fold}/{method}/{cand}/{j}',build);ys+=r['y'];ps+=r['p']
                score=float(f1_score(ys,ps,labels=labels,average='macro',zero_division=0));ranking.append((score,cand,param));search.append({'fold':fold,'method':method,'candidate':cand,'parameters':json.dumps(param,sort_keys=True),'inner_macro_f1':score})
            score,_,best=sorted(ranking,key=lambda x:(-x[0],x[1]))[0]
            def fit():
                t=time.perf_counter();m=aeon_model(method,best,seed_of(cfg['seed'],fold,method,'outer'));m.fit(tensor([frames[i] for i in tr],B,N),np.array([frames[i]['y'] for i in tr]));return m,time.perf_counter()-t
            m,seconds=store.task(f'matched/outer/{fold}/{method}',fit)
            t=time.perf_counter();p=m.predict(tensor([frames[i] for i in te],B,N));elapsed=time.perf_counter()-t
            selected.append({'fold':fold,'method':method,'parameters':json.dumps(best,sort_keys=True),'inner_macro_f1':score,'B':B,'max_cycles':N})
            fits.append({'fold':fold,'method':method,'fit_seconds':seconds,'inference_seconds':elapsed,'n_inference_segments':len(te),'scope':'excludes preprocessing; CPU host-dependent'})
            preds += [dict(id=frames[i]['id'],subject=frames[i]['subject'],y=frames[i]['y'],pred=str(q),fold=fold,arm=method) for i,q in zip(te,p)]
            for name,rs in [('matched_oof_predictions',preds),('matched_inner_search',search),('matched_selected_configs',selected),('matched_runtime',fits),('matched_split_audit',fold_rows)]:csv_write(out/(name+'.csv'),rs)
    # Matched phase input reaches all models before semantic mapping; PG does not see validity padding.
    for method in cfg['baseline_grid']:
        rows=[r for r in preds if r['arm']==method]
        csv_write(out/('metrics_'+method+'.csv'),summary(rows,cfg['bootstrap'],cfg['seed']))
    return preds,folds

def run_matched_pg(frames,source,cfg,store,out,folds):
    """Independent inner-fold tuning and outer-fold PG-LTS refit on EXACT baseline cycles."""
    original=json.loads((source/'run_identity.json').read_text())['config']
    labels=sorted({f['y'] for f in frames});rows=[];search=[];selected=[]
    opts=dict(original,max_cycles=cfg['max_cycles'],threads=cfg['threads'])
    for fold,(tr,te) in enumerate(folds):
        inner=split_subjects(frames,tr,original['inner_folds'],original['seed']+fold+1)
        for arm in cfg['pg_arms']:
            candidates=[{'view':arm,'method':'linguistic','partition':part,'states':states,'B':cfg['B'],'l1':pen,'lg':pen/2}
                        for part in cfg['pg_partitions'] for states in cfg['pg_states'] for pen in cfg['pg_penalties']]
            ranking=[]
            for c in candidates:
                key=digest(c)[:12];y=[];p=[]
                for j,(it,iv) in enumerate(inner):
                    def build():
                        m=Model(c,opts).fit([frames[i] for i in it]);return {'y':[frames[i]['y'] for i in iv],'p':list(m.predict([frames[i] for i in iv]))}
                    r=store.task(f'matched_pg/inner/{fold}/{arm}/{key}/{j}',build)
                    y+=r['y'];p+=r['p']
                value=float(f1_score(y,p,labels=labels,average='macro',zero_division=0))
                ranking.append((value,key,c));search.append({'fold':fold,'arm':arm,'candidate_id':key,'inner_macro_f1':value,'parameters':json.dumps(c,sort_keys=True)})
            value,_,best=sorted(ranking,key=lambda x:(-x[0],x[1]))[0]
            m=store.task(f'matched_pg/outer/{fold}/{arm}',lambda:Model(best,opts).fit([frames[i] for i in tr]))
            fp=m.predict([frames[i] for i in te]);selected.append({'fold':fold,'arm':arm,'inner_macro_f1':value,'parameters':json.dumps(best,sort_keys=True),'B':cfg['B'],'max_cycles':cfg['max_cycles']})
            rows += [dict(id=frames[i]['id'],subject=frames[i]['subject'],y=frames[i]['y'],pred=str(q),fold=fold,arm='PG_'+arm) for i,q in zip(te,fp)]
            for name,r in [('matched_pg_oof_predictions',rows),('matched_pg_inner_search',search),('matched_pg_selected_configs',selected)]:csv_write(out/(name+'.csv'),r)
    for arm in cfg['pg_arms']:
        csv_write(out/f'metrics_PG_{arm}.csv',summary([r for r in rows if r['arm']=='PG_'+arm],cfg['bootstrap'],cfg['seed']))
    return rows

def compare_matched(baselines,pg,cfg,out):
    outrows=[]
    for a in sorted({r['arm'] for r in pg}):
        for b in sorted({r['arm'] for r in baselines}):
            ar=[r for r in pg if r['arm']==a];br=[r for r in baselines if r['arm']==b]
            for level in ('segment','subject'):
                outrows.append({'method_a':a,'method_b':b,**paired(ar,br,cfg['bootstrap'],cfg['seed'],level)})
    csv_write(out/'matched_paired_deltas.csv',outrows)

def load_models(source,frames,folds,arm):
    import joblib
    result=[]
    for fold,(tr,te) in enumerate(folds):
        p=source/'checkpoints'/f'outer__{fold}__{arm}.joblib';receipt=p.with_suffix('.done.json')
        if not p.exists() or not receipt.exists() or sha(p)!=json.loads(receipt.read_text())['sha256']:
            raise FileNotFoundError('Missing verified held-out model '+str(p))
        m=joblib.load(p);result.append((fold,m,te))
    return result

def run_robust(frames,source,cfg,store,out,folds):
    from .phase import cycles
    results=[];diagnostics=[];metrics=[]
    for arm in cfg['pg_arms']:
        for fold,m,te in load_models(source,frames,folds,arm):
            for condition in cfg['perturbations']:
                def build():
                    records=[];diag=[]
                    for i in te:
                        f=frames[i];s=seed_of(cfg['seed'],fold,arm,condition['name'],f['id'])
                        g=corrupt_frame(f,condition,s)
                        boundary=condition if condition['kind'] not in ('clean','awgn','radar_phase_jitter') else None
                        _,st=cycles(g['amp'],g['cpx'],m.c.get('B',cfg['B']),m.options['max_cycles'],boundary,s,True)
                        accepted=not st['segment_rejected'];p=str(m.predict([g],boundary,s)[0]) if accepted else '__ABSTAIN__'
                        records.append({'id':f['id'],'subject':f['subject'],'y':f['y'],'pred':p,'accepted':int(accepted),'arm':arm,'fold':fold,'condition':condition['name']})
                        diag.append({'id':f['id'],'subject':f['subject'],'arm':arm,'fold':fold,'condition':condition['name'],**st})
                    return records,diag
                rows,diags=store.task(f'robust/{fold}/{arm}/{condition["name"]}',build);results+=rows;diagnostics+=diags
                csv_write(out/'robustness_predictions.csv',results);csv_write(out/'robustness_cycle_diagnostics.csv',diagnostics)
        clean=[r for r in results if r['arm']==arm and r['condition']=='clean']
        for condition in cfg['perturbations']:
            r=[x for x in results if x['arm']==arm and x['condition']==condition['name']]
            if not r:continue
            labels=sorted({x['y'] for x in r})
            metrics.append({'arm':arm,'condition':condition['name'],'n':len(r),'coverage':np.mean([x['accepted'] for x in r]),'macro_f1_abstain_is_error':float(f1_score([x['y'] for x in r],[x['pred'] for x in r],labels=labels,average='macro',zero_division=0)),**({k:v for k,v in paired(r,clean,cfg['bootstrap'],cfg['seed']).items() if k in ('delta_macro_f1','lo','hi')} if condition['name']!='clean' else {})})
            csv_write(out/'robustness_metrics.csv',metrics)
    return results

def run_fidelity(frames,source,cfg,store,out,folds):
    from scipy.special import softmax
    records=[];rules=[];scores=[]
    for arm in cfg['pg_arms']:
        for fold,m,te in load_models(source,frames,folds,arm):
            if not hasattr(m.clf,'coef_'):
                raise ValueError('Rule fidelity requires a linear sparse-group model: '+arm)
            fs=[frames[i] for i in te];z=m.transform(fs);full=m.scores(fs);coef=np.asarray(m.clf.coef_)
            if coef.shape[0]!=len(m.classes_):raise ValueError('Multiclass coefficient alignment needs verification')
            importance=np.abs(coef)*m.train_abs
            for k in cfg['rule_k']:
                def build():
                    mask=np.zeros_like(coef,dtype=bool)
                    for c in range(len(coef)):
                        top=np.argsort(-importance[c],kind='stable')[:k];mask[c,top]=True
                    partial=z@(coef*mask).T+np.asarray(m.clf.intercept_)
                    fullpred=m.classes_[full.argmax(1)];p=m.classes_[partial.argmax(1)]
                    delta=(partial-partial.mean(1,keepdims=True))-(full-full.mean(1,keepdims=True))
                    return mask,fullpred,p,partial,delta
                mask,fp,pp,part,delta=store.task(f'faithful/{fold}/{arm}/top{k}',build)
                for j,i in zip(*np.where(mask)):
                    rules.append({'arm':arm,'fold':fold,'top_k':k,'class':str(m.classes_[j]),'feature':str(m.names[i]),'sign':int(np.sign(coef[j,i])),'coefficient':float(coef[j,i])})
                for j,f in enumerate(fs):
                    records.append({'id':f['id'],'subject':f['subject'],'y':f['y'],'fold':fold,'arm':arm,'top_k':k,'full_pred':str(fp[j]),'sparse_pred':str(pp[j]),'agreement':int(fp[j]==pp[j]),'score_rmse':float(np.sqrt(np.mean(delta[j]**2))),'probability_mae':float(np.abs(softmax(part[j])-softmax(full[j])).mean())})
                scores.append({'arm':arm,'fold':fold,'top_k':k,'n':len(fs),'agreement':float(np.mean(fp==pp)),'sparse_macro_f1':float(f1_score([f['y'] for f in fs],pp,labels=m.classes_,average='macro',zero_division=0)),'full_macro_f1':float(f1_score([f['y'] for f in fs],fp,labels=m.classes_,average='macro',zero_division=0))})
                for name,rows in [('fidelity_predictions',records),('fidelity_rules',rules),('fidelity_fold_metrics',scores)]:csv_write(out/(name+'.csv'),rows)
    import itertools
    stability=[]
    for arm in cfg['pg_arms']:
        for k in cfg['rule_k']:
            groups=[]
            for fold in sorted({r['fold'] for r in rules}):groups.append({(r['class'],r['feature'],r['sign']) for r in rules if r['arm']==arm and r['top_k']==k and r['fold']==fold})
            vals=[len(a&b)/len(a|b) if a|b else 1. for a,b in itertools.combinations(groups,2)]
            stability.append({'arm':arm,'top_k':k,'folds':len(groups),'mean_pairwise_rule_jaccard':np.mean(vals) if vals else None})
    csv_write(out/'fidelity_stability.csv',stability)
    for arm in cfg['pg_arms']:
        for k in cfg['rule_k']:
            rr=[r for r in records if r['arm']==arm and r['top_k']==k]
            if rr:
                from .protocol import SubjectBootstrap
                sampler=SubjectBootstrap(rr);v=np.array([r['agreement'] for r in rr]);rng=np.random.default_rng(seed_of(cfg['seed'],arm,k))
                draws=np.array([v[sampler.draw(rng)].mean() for _ in range(cfg['bootstrap'])]);csv_write(out/f'fidelity_ci_{arm}_top{k}.csv',[{'arm':arm,'top_k':k,'agreement':v.mean(),'lo':np.quantile(draws,.025),'hi':np.quantile(draws,.975),'independent_units':len({r['subject'] for r in rr})}])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source-results',required=True);ap.add_argument('--config',default='configs/reviewer_closure.json');ap.add_argument('--out',default='runs_reviewer_closure');ap.add_argument('--stage',choices=['matched','robustness','fidelity','all'],default='all');ap.add_argument('--stop-after',type=int,default=0);a=ap.parse_args()
    source=Path(a.source_results).resolve();cfg=json.loads(Path(a.config).read_text(encoding='utf-8'));frames,identity,receipts=load_prepared(source)
    if identity['config']['data_mode']!='polliradar':raise ValueError('Only real Polliradar inputs are accepted for manuscript claims')
    original=identity['config'];assert original['expected_subjects']==len({f['subject'] for f in frames})
    if cfg['max_cycles']>original['max_cycles']:raise ValueError('max_cycles exceeds source extraction ceiling')
    manifest={'source':str(source),'source_run_sha':sha(source/'run_identity.json'),'prepared_receipts':receipts,'source_models':fingerprint_models(source),'config':cfg,'script_sha':sha(__file__),'python':platform.python_version()}
    out=(ROOT/a.out/('reviewer_closure_'+digest(manifest)[:16])).resolve()
    with run_lock(out),threadpool_limits(limits=cfg['threads']):
        json_write(out/'identity.json',manifest);store=Store(out/'checkpoints',a.stop_after);folds=split_subjects(frames,np.arange(len(frames)),original['outer_folds'],original['seed']);done=out/'complete.json';done.unlink(missing_ok=True)
        stages=['matched','robustness','fidelity'] if a.stage=='all' else [a.stage]
        try:
            for stage in stages:
                if stage=='matched':
                    baselines,_=run_baseline(frames,source,cfg,store,out)
                    pg=run_matched_pg(frames,source,cfg,store,out,folds)
                    compare_matched(baselines,pg,cfg,out)
                elif stage=='robustness':run_robust(frames,source,cfg,store,out,folds)
                else:run_fidelity(frames,source,cfg,store,out,folds)
                json_write(out/f'{stage}_complete.json',{'stage':stage,'completed':True,'reused_jobs':store.reused,'new_jobs':store.completed})
            json_write(done,{'stages_completed_this_invocation':stages,'n_subjects':len({f['subject'] for f in frames}),'n_frames':len(frames),'scope':'empirical only when authenticated prepared input supplied'})
            print('DONE',out)
        except InterruptedError as e:
            json_write(out/'interrupted.json',{'reason':str(e),'resume':'rerun identical command'});raise SystemExit(75)
        except Exception as e:
            json_write(out/'last_error.json',{'type':type(e).__name__,'message':str(e)});raise
if __name__=='__main__':main()
