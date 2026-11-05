"""Frozen clean-trained outer models, corrupted held-out data and explicit abstention."""
import hashlib
import numpy as np
from sklearn.metrics import f1_score
from pglts.signal.features import FeatureExtractor,PGLTS_CORE
from pglts.signal.segmentation import harmonic_ratio
from .phase import cycles
from .storage import csv_write
from .protocol import paired

def corrupt_frame(f,condition,seed):
    g=dict(f);rng=np.random.default_rng(seed);kind=condition['kind'];a=np.array(f['amp'],float);z=np.array(f['cpx'],complex);v=condition['amount']
    if kind=='awgn':
        a+=rng.normal(0,np.std(a)*10**(-v/20),len(a));sigma=np.sqrt(np.mean(np.abs(z)**2)/2)*10**(-v/20)
        z+=sigma*(rng.normal(size=len(z))+1j*rng.normal(size=len(z)))
    elif kind=='radar_phase_jitter':z*=np.exp(1j*rng.normal(0,v,len(z)))
    g['amp']=a.astype(np.float32);g['cpx']=z.astype(np.complex64)
    if kind=='awgn':g['window']=FeatureExtractor(names=PGLTS_CORE).transform(a).astype(np.float32)
    return g

def run(frames,results,cfg,store,folder):
    output=[];rates=[];metrics=[]
    for (fold,arm),entry in results.items():
        if arm not in cfg['robustness_arms']:continue
        model=entry['model'];clean=[frames[i] for i in entry['test']]
        for condition in cfg['perturbations']:
            def build(condition=condition,model=model,clean=clean):
                rows=[];diagnostics=[]
                for f in clean:
                    seed=int.from_bytes(hashlib.sha256(f"{cfg['seed']}:{condition}:{f['id']}".encode()).digest()[:4],'little')
                    g=corrupt_frame(f,condition,seed);boundary=condition if condition['kind'] not in ('awgn','radar_phase_jitter','clean') else None
                    cycle_seed=int.from_bytes(hashlib.sha256(f"{seed}:{g['id']}".encode()).digest()[:4],'little')
                    _,st=cycles(g['amp'],g['cpx'],model.c.get('B',8),cfg['max_cycles'],boundary,cycle_seed,True)
                    uses_cycle=model.c['view'] in ('combined','phase_ordered','phase_occ');accepted=st['segment_rejected']==0 or not uses_cycle
                    pred=str(model.predict([g],boundary,seed)[0]) if accepted else '__ABSTAIN__'
                    rows.append({'id':f['id'],'subject':f['subject'],'y':f['y'],'pred':pred,'cycle_accepted':int(accepted),'hr_accepted':int(harmonic_ratio(g['amp'])>cfg['hr_threshold'])})
                    diagnostics.append({'id':f['id'],'subject':f['subject'],**st})
                return rows,diagnostics
            rows,diag=store.task(f'robustness/{fold}/{arm}/{condition["name"]}',build)
            output.extend(dict(r,arm=arm,fold=fold,condition=condition['name']) for r in rows)
            rates.extend(dict(r,arm=arm,fold=fold,condition=condition['name']) for r in diag)
    labels=sorted({f['y'] for f in frames})
    for arm in cfg['robustness_arms']:
        cleanrows=[r for (fold,a),e in results.items() if a==arm for r in e['rows']]
        for cond in cfg['perturbations']:
            rows=[r for r in output if r['arm']==arm and r['condition']==cond['name']]
            if not rows:continue
            valid=[r for r in rows if r['cycle_accepted']];retained=[r for r in valid if r['hr_accepted']]
            for cohort,data in [('fixed_clean_cohort_abstention_as_error',rows),('cycle_accepted_only',valid),('cycle_and_hr_accepted_only',retained)]:
                value=float(f1_score([r['y'] for r in data],[r['pred'] for r in data],labels=labels,average='macro',zero_division=0)) if data else ''
                metrics.append({'arm':arm,'condition':cond['name'],'cohort':cohort,'coverage':len(data)/len(rows),'segments':len(data),'macro_f1':value})
            metrics.append({'arm':arm,'condition':cond['name'],'cohort':'paired_fixed_clean_cohort','coverage':1,**paired(rows,cleanrows,cfg['bootstrap'],cfg['seed'])})
    csv_write(folder/'robustness_predictions.csv',output);csv_write(folder/'phase_failure_rates.csv',rates);csv_write(folder/'robustness_metrics.csv',metrics)

def synthetic_cycle_detection(cfg,store,folder):
    from pglts.signal.cycle import cycle_boundaries,dominant_f0
    rows=[]
    for hz in [80.,125.,165.]:
        for snr in [40.,20.,10.,0.]:
            def build(hz=hz,snr=snr):
                rng=np.random.default_rng(cfg['seed']);t=np.arange(2000)/1000
                x=np.sin(2*np.pi*hz*t);x+=rng.normal(0,np.std(x)*10**(-snr/20),len(x))
                starts=cycle_boundaries(x,dominant_f0(x));starts=starts[(starts>=100)&(starts<1900)]
                # Compare period recovery without fitting a phase offset on test data.
                expected=hz*1.8;period=np.diff(starts);error=np.abs(period-1000/hz)
                truth=np.arange(1,int(2*hz)+1)*1000/hz;truth=truth[(truth>=100)&(truth<1900)]
                remaining=list(truth);matched=0;errors=[]
                for detected in starts:
                    if not remaining:break
                    ix=int(np.argmin(np.abs(np.array(remaining)-detected)));e=abs(remaining[ix]-detected)
                    if e<=1:matched+=1;errors.append(e);remaining.pop(ix)
                return {'data_mode':'synthetic_control','hz':hz,'snr_db':snr,'detected_boundaries_interior':len(starts),'expected_cycles_interior':expected,'expected_boundaries_interior':len(truth),'count_error':len(starts)-expected,'period_mae_samples':float(error.mean()) if len(error) else '', 'boundary_precision_1sample':matched/max(len(starts),1),'boundary_recall_1sample':matched/max(len(truth),1),'boundary_timing_mae_matched':float(np.mean(errors)) if errors else '', 'phase_scope':'known synthetic zero-crossing cycles, no fitted offset; no biomechanical phase claim'}
            rows.append(store.task(f'cycle_synthetic/{hz}/{snr}',build))
    csv_write(folder/'cycle_detection_synthetic.csv',rows)
