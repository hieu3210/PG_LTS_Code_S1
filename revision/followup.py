"""Additive follow-up; old result/model directories are read-only.
python -m revision.followup --stage summarize --source-results /path/to/old_run
python -m revision.followup --stage calibration --config configs/followup_calibration.json
"""
from pathlib import Path
import argparse,json,platform,sys,traceback,csv
import numpy as np
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
from . import ROOT
from .storage import Store,run_lock,json_write,csv_write,sha,digest
from .cli import environment
from .protocol import paired
from .followup_calibration import run as calibrate
VERSION='1.1.0'


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def legacy_identity(source):
    identity=json.loads((source/'run_identity.json').read_text());expected=json.loads((ROOT/'LEGACY_COMPATIBILITY.json').read_text())
    if identity['sources']!=expected['sources']:raise ValueError('Unsupported legacy source fingerprint; do not mix an unknown implementation.')
    # Every legacy source file must still be byte-identical. Additive modules are permitted.
    for rel,h in expected['files'].items():
        if sha(ROOT/rel)!=h:raise ValueError('Legacy computational source changed: '+rel)
    return identity


def summarize(source,folder,store,n_boot,seed):
    identity=legacy_identity(source);pred=read_csv(source/'oof_predictions.csv');rules=read_csv(source/'rule_predictions.csv')
    labels=sorted({r['y'] for r in pred});expected=set(r['id'] for r in pred if r['arm']==pred[0]['arm'])
    for arm in sorted({r['arm'] for r in pred}):
        data=[r for r in pred if r['arm']==arm]
        if len(data)!=len(expected) or {r['id'] for r in data}!=expected:raise ValueError('Incomplete or nonmatched OOF cohort: '+arm)
    pooled=[]
    for arm,k in sorted({(r['arm'],int(r['top_k_per_class'])) for r in rules}):
        g=[r for r in rules if r['arm']==arm and int(r['top_k_per_class'])==k]
        if len(g)!=len(expected) or {r['id'] for r in g}!=expected:raise ValueError('Incomplete rule predictions: '+arm)
        pooled.append({'arm':arm,'top_k_per_class':k,'segments':len(g),'pooled_full_macro_f1':f1_score([r['y'] for r in g],[r['full_pred'] for r in g],labels=labels,average='macro',zero_division=0),'pooled_rule_macro_f1':f1_score([r['y'] for r in g],[r['pred'] for r in g],labels=labels,average='macro',zero_division=0),'agreement':np.mean([r['pred']==r['full_pred'] for r in g]),'scope':'pooled fixed OOF predictions; not mean fold F1'})
    csv_write(folder/'rule_metrics_pooled.csv',pooled)
    contrasts=[]
    for a,b in [('combined','occupancy'),('phase_ordered','phase_occ'),('phase_ordered','phase_rocket'),('combined','combined_rocket'),('phase_ordered','phase_linear'),('combined','combined_linear'),('phase_ordered','phase_multirocket'),('combined','combined_multirocket')]:
        if a not in {r['arm'] for r in pred} or b not in {r['arm'] for r in pred}:continue
        for level in ['segment','subject']:
            value=store.task(f'paired/{a}/{b}/{level}',lambda a=a,b=b,level=level:paired([r for r in pred if r['arm']==a],[r for r in pred if r['arm']==b],n_boot,seed,level))
            contrasts.append({'a':a,'b':b,**value,'comparison_scope':'follow-up exploratory; no familywise multiplicity correction'})
    csv_write(folder/'paired_deltas_extended.csv',contrasts)
    # Pool squared score errors by segment count, not an arithmetic mean of fold RMSE.
    from collections import Counter
    weights_by_fold=Counter((r['arm'],int(r['top_k_per_class']),r['fold']) for r in rules)
    fidelity=read_csv(source/'rule_fidelity.csv');score=[]
    for arm,k in sorted({(r['arm'],int(r['top_k_per_class'])) for r in fidelity}):
        g=[r for r in fidelity if r['arm']==arm and int(r['top_k_per_class'])==k];weights=[]
        for r in g:weights.append(weights_by_fold[(arm,k,r['fold'])])
        if not all(weights):raise ValueError('Missing rule fold predictions')
        score.append({'arm':arm,'top_k_per_class':k,'centered_score_rmse':float(np.sqrt(np.average([float(r['centered_score_rmse'])**2 for r in g],weights=weights))),'probability_mae':float(np.average([float(r['probability_mae']) for r in g],weights=weights)),'scope':'segment-weighted score errors; same class count across folds'})
    csv_write(folder/'rule_score_fidelity_pooled.csv',score)
    return {'source_run_name':identity['config']['name'],'segments':len(expected),'subjects':len({r['subject'] for r in pred}),'legacy_models_refitted':False}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['calibration','summarize','status'],required=True);p.add_argument('--config');p.add_argument('--source-results');p.add_argument('--output-root',default='runs_followup');p.add_argument('--bootstrap',type=int,default=5000);p.add_argument('--seed',type=int,default=20261031);p.add_argument('--stop-after',type=int,default=0);a=p.parse_args(argv)
    if a.stage=='calibration' or (a.stage=='status' and a.config):
        if not a.config:p.error('--config required')
        cfg=json.loads(Path(a.config).read_text());validate(cfg);inputs=[];kind='aggregation_calibration'
    else:
        if not a.source_results:p.error('--source-results required')
        source=Path(a.source_results).resolve();legacy_identity(source)
        names=['run_identity.json','oof_predictions.csv','rule_predictions.csv','rule_fidelity.csv']
        inputs=[{'name':n,'sha256':sha(source/n)} for n in names];cfg={'bootstrap':a.bootstrap,'seed':a.seed,'threads':1};kind='saved_result_summary'
    modules=['followup.py','followup_calibration.py','storage.py','order.py','protocol.py'];signature={'extension_version':VERSION,'kind':kind,'config':cfg,'inputs':inputs,'modules':{n:sha(ROOT/'revision'/n) for n in modules},'environment':environment()}
    run_id=kind+'-'+digest(signature)[:16];folder=ROOT/a.output_root/run_id;print('FOLLOWUP DIRECTORY',folder,flush=True)
    from . import storage
    storage.CSV_CONTEXT={'data_mode':'synthetic_aggregated_control' if kind=='aggregation_calibration' else 'saved_results_followup','run_id':run_id}
    with run_lock(folder),threadpool_limits(limits=cfg['threads']):
        json_write(folder/'followup_identity.json',signature);store=Store(folder/'checkpoints',a.stop_after)
        if a.stage=='status':
            print('Committed receipts:',len(list(store.folder.glob('*.done.json'))));print('Status does not validate checkpoint payloads; hashes checked on resume.');return
        for partial in folder.rglob('*.partial'):partial.unlink(missing_ok=True)
        # Invalidate stale completion before potentially rebuilding any task.
        (folder/'complete.json').unlink(missing_ok=True)
        try:
            result=calibrate(cfg,store,folder) if kind=='aggregation_calibration' else summarize(source,folder,store,a.bootstrap,a.seed)
            json_write(folder/'complete.json',{'stage':a.stage,'new_tasks':store.completed,'reused_tasks':store.reused,**result});print('COMPLETE',a.stage,'new=',store.completed,'reused=',store.reused,flush=True)
        except InterruptedError as e:
            json_write(folder/'interrupted.json',{'reason':str(e),'new_tasks':store.completed});raise SystemExit(75)
        except Exception as e:
            json_write(folder/'last_error.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()});raise


def validate(c):
    for n in ['repetitions','null_permutations','draw_batch','channels','threads']:
        if not isinstance(c[n],int) or c[n]<1:raise ValueError('Positive integer required: '+n)
    if c['states']<2 or not 0<c['alpha']<1 or min(c['bins'])<2:raise ValueError('Invalid states/bins/alpha')
    if not set(c['groups'])<= {'uniform_bins','cyclic_shift'}:raise ValueError('Invalid groups')
    if not set(c['overlap_couplings'])<= {'coherent_cycle_id','independent_occurrence'}:raise ValueError('Invalid couplings')
    for x in c['profiles']:
        if x['segments']<1 or x['cycles']<1:raise ValueError('Invalid aggregation profile')
    for x in c['scenarios']:
        if x['generator'] not in ['iid','smooth'] or not 0<=x['strength']<=1:raise ValueError('Invalid scenario')
if __name__=='__main__':main()
