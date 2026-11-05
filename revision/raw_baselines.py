"""Optional nested ROCKET/MultiRocket on prepared I/Q or complex STFT power.
Reuses the old accepted cohort and exact subject folds. No new measurement data.
"""
from pathlib import Path
import argparse,itertools,json,time
import joblib
import numpy as np
from scipy.signal import stft
from threadpoolctl import threadpool_limits
from . import ROOT
from .models import Model
from .followup import legacy_identity,read_csv
from .storage import Store,run_lock,digest,sha,json_write,csv_write
from .protocol import split_subjects,score,paired
from .experiments import analyze
from .cli import environment


def raw_tensor(frames,view):
    z=np.stack([f['cpx'] for f in frames])
    if view=='raw_iq':return np.stack([z.real,z.imag],axis=1).astype(np.float32)
    if view=='spectrogram':
        # Fixed STFT, no fitting on test data: two-sided complex-return power.
        _,_,x=stft(z,fs=1000,nperseg=128,noverlap=96,boundary=None,padded=False,return_onesided=False,axis=-1)
        return (10*np.log10(np.maximum(np.abs(x)**2,1e-30))).astype(np.float32)
    raise ValueError('Unknown raw view')

class RawModel(Model):
    def _describe(self,frames,fit=False,pert=None,seed=0):
        if pert:raise ValueError('Raw baseline perturbation not implemented')
        if fit:
            from aeon.transformations.collection.convolution_based import Rocket,MultiRocket
            cls=Rocket if self.c['method']=='rocket' else MultiRocket
            self.rocket=cls(n_kernels=self.c['kernels'],n_jobs=self.options['threads'],random_state=self.c['seed'])
            self.rocket.fit(raw_tensor(frames,self.c['view']))
        # Limit transformed output allocation; retain the fixed full accepted cohort.
        pieces=[]
        for start in range(0,len(frames),self.options['transform_batch']):
            pieces.append(np.asarray(self.rocket.transform(raw_tensor(frames[start:start+self.options['transform_batch']],self.c['view']))))
        x=np.concatenate(pieces).astype(float)
        return x,[f'{self.c["view"]}:{i}' for i in range(x.shape[1])],['raw']*x.shape[1]


def load_prepared(source):
    identity=legacy_identity(source);manifest=read_csv(source/'segment_manifest.csv');wanted={r['id'] for r in manifest};frames=[];receipts=[]
    for p in sorted((source/'checkpoints').glob('prepare__record_*.joblib')):
        receipt=p.with_suffix('.done.json')
        if not receipt.exists() or json.loads(receipt.read_text())['sha256']!=sha(p):raise ValueError('Broken preparation checkpoint: '+str(p))
        f,_=joblib.load(p);frames.extend(f);receipts.append({'name':p.name,'sha256':sha(p)})
    if len(frames)!=len(wanted) or {f['id'] for f in frames}!=wanted:raise ValueError('Preparation checkpoints are missing/incomplete. Restore original run checkpoints; summary-only ZIP is insufficient for raw baselines.')
    return frames,identity,receipts


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-results',required=True);p.add_argument('--config',default='configs/raw_baselines.json');p.add_argument('--output-root',default='runs_followup');p.add_argument('--stop-after',type=int,default=0);a=p.parse_args(argv)
    source=Path(a.source_results).resolve();frames,old,receipts=load_prepared(source);cfg=json.loads(Path(a.config).read_text());labels=sorted({f['y'] for f in frames})
    identity={'source_identity_sha256':sha(source/'run_identity.json'),'prepared_checkpoints':receipts,'legacy_oof_sha256':sha(source/'oof_predictions.csv'),'config':cfg,'module_sha256':sha(__file__),'environment':environment()}
    folder=ROOT/a.output_root/('raw_baselines-'+digest(identity)[:16]);print('RAW BASELINES DIRECTORY',folder,flush=True)
    from . import storage
    storage.CSV_CONTEXT={'data_mode':old['config']['data_mode'],'run_id':folder.name}
    with run_lock(folder),threadpool_limits(limits=cfg['threads']):
        json_write(folder/'followup_identity.json',identity);store=Store(folder/'checkpoints',a.stop_after);(folder/'complete.json').unlink(missing_ok=True)
        try:
            outer=split_subjects(frames,np.arange(len(frames)),old['config']['outer_folds'],old['config']['seed']);prior_splits=read_csv(source/'splits.csv');all_pred=[];search=[];selected=[]
            pool={}
            for view,method in itertools.product(cfg['views'],cfg['methods']):
                arm=view+'_'+method;candidates=[{'view':view,'method':method,'C':C,'kernels':k,'seed':old['config']['seed']} for C,k in itertools.product(cfg['C_values'],cfg['kernels'])]
                pool[arm]=[dict(c,candidate_id=digest(c)[:12]) for c in candidates]
            json_write(folder/'search_space.json',pool)
            for fold,(tr,te) in enumerate(outer):
                expected={r['subject'] for r in prior_splits if r['level']=='outer' and int(r['outer_fold'])==fold and r['role']=='test'}
                if {frames[i]['subject'] for i in te}!=expected:raise ValueError('Old outer splits do not reproduce')
                inner=split_subjects(frames,tr,old['config']['inner_folds'],old['config']['seed']+fold+1)
                for arm,candidates in pool.items():
                    ranks=[]
                    for c in candidates:
                        ys=[];ps=[]
                        for j,(it,iv) in enumerate(inner):
                            def build(it=it,iv=iv,c=c):
                                m=RawModel(c,cfg).fit([frames[i] for i in it]);return {'y':[frames[i]['y'] for i in iv],'pred':m.predict([frames[i] for i in iv]).tolist(),'converged':m.converged}
                            r=store.task(f'inner/{fold}/{arm}/{c["candidate_id"]}/{j}',build);ys+=r['y'];ps+=r['pred']
                        from sklearn.metrics import f1_score
                        val=float(f1_score(ys,ps,labels=labels,average='macro',zero_division=0));ranks.append((val,c['candidate_id'],c));search.append({'fold':fold,'arm':arm,'candidate_id':c['candidate_id'],'inner_macro_f1':val,'config':json.dumps(c,sort_keys=True)})
                    val,_,best=sorted(ranks,key=lambda x:(-x[0],x[1]))[0]
                    m=store.task(f'outer/{fold}/{arm}',lambda best=best:RawModel(best,cfg).fit([frames[i] for i in tr]));scores=m.scores([frames[i] for i in te]);pred=m.classes_[scores.argmax(1)]
                    all_pred.extend({'id':frames[i]['id'],'subject':frames[i]['subject'],'y':frames[i]['y'],'pred':str(y),'fold':fold,'arm':arm} for i,y in zip(te,pred));selected.append({'fold':fold,'arm':arm,'inner_macro_f1':val,'config':json.dumps(best,sort_keys=True),'input_features':len(m.names),'converged':m.converged})
                    for name,rows in [('oof_predictions',all_pred),('inner_search',search),('selected_configs',selected)]:csv_write(folder/(name+'.csv'),rows)
            analyze(all_pred,cfg,folder)
            legacy=read_csv(source/'oof_predictions.csv');deltas=[]
            for arm in pool:
                for baseline in ['occupancy','combined','phase_ordered']:
                    aa=[r for r in all_pred if r['arm']==arm];bb=[r for r in legacy if r['arm']==baseline]
                    if not bb:continue
                    for level in ['segment','subject']:
                        r=store.task(f'paired/{arm}/{baseline}/{level}',lambda aa=aa,bb=bb,level=level:paired(aa,bb,cfg['bootstrap'],cfg['seed'],level));deltas.append({'a':arm,'b':baseline,**r,'scope':'exploratory follow-up; unadjusted comparisons; prepared I/Q cohort'})
            csv_write(folder/'paired_vs_legacy.csv',deltas);json_write(folder/'complete.json',{'new_tasks':store.completed,'reused_tasks':store.reused,'legacy_models_refitted':False,'scope':'Prepared high-pass/normalized I/Q or fixed STFT; no phase-cycle extraction in these baselines; same previously accepted cohort, including legacy cycle eligibility.'});print('COMPLETE raw baselines',store.completed,store.reused)
        except InterruptedError as e:json_write(folder/'interrupted.json',{'reason':str(e)});raise SystemExit(75)
        except Exception as e:json_write(folder/'last_error.json',{'type':type(e).__name__,'message':str(e)});raise
if __name__=='__main__':main()
