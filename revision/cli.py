"""python -m revision.cli --config configs/smoke.json --stage all"""
from pathlib import Path
import argparse,importlib.metadata,json,platform,sys,traceback
import numpy as np
from threadpoolctl import threadpool_limits
from . import ROOT,__version__
from .storage import digest,sha,json_write,csv_write,Store,run_lock
from .data import input_identity,prepare
from .experiments import nested,analyze
from .order import synthetic_calibration,measured_order,permutation_classification
from . import robustness,fidelity

def source_identity():
    return digest({str(p.relative_to(ROOT)):sha(p) for folder in ['revision','pglts','third_party/pyLTS/lts'] for p in sorted((ROOT/folder).rglob('*.py'))})

def environment():
    out={'python':platform.python_version(),'platform':platform.platform(),'processor':platform.processor(),'package_version':__version__}
    for name in ['numpy','scipy','scikit-learn','aeon','numba','joblib']:
        try:out[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:out[name]='not installed'
    return out

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--stage',choices=['prepare','nested','analyze','calibration','permutation','robustness','fidelity','all','status'],default='all')
    parser.add_argument('--stop-after',type=int,default=0,help='Testing: stop after N NEW task commits; rerun without flag.')
    args=parser.parse_args(argv);cfg=json.loads(Path(args.config).read_text(encoding='utf-8'))
    if cfg['data_mode'] not in ('synthetic','polliradar'):raise ValueError('data_mode must be explicit.')
    cfg['data_root']=str((ROOT/cfg['data_root']).resolve()) if not Path(cfg['data_root']).is_absolute() else cfg['data_root']
    inputs=input_identity(cfg);env=environment()
    signature={'config':cfg,'sources':source_identity(),'inputs':inputs,'environment':env}
    run_id=cfg['name']+'-'+digest(signature)[:16];folder=ROOT/cfg['output_root']/run_id
    from . import storage
    storage.CSV_CONTEXT={'data_mode':cfg['data_mode'],'run_id':run_id}
    print('RUN DIRECTORY',folder,flush=True)
    with run_lock(folder),threadpool_limits(limits=cfg['threads']):
        json_write(folder/'run_identity.json',signature);store=Store(folder/'checkpoints',args.stop_after)
        for partial in folder.rglob('*.partial'):partial.unlink(missing_ok=True)
        if args.stage=='status':
            receipts=list((folder/'checkpoints').glob('*.done.json'));print('Committed tasks:',len(receipts));print('No checkpoint deserialization required for status.');return
        try:
            if args.stage=='calibration':
                synthetic_calibration(cfg,store,folder);json_write(folder/'calibration_complete.json',{'data_mode':'synthetic calibration, not radar classification'});return
            frames,audit=prepare(cfg,store);csv_write(folder/'preparation_audit.csv',audit)
            csv_write(folder/'segment_manifest.csv',[{'id':f['id'],'subject':f['subject'],'species':f['y'],'data_mode':cfg['data_mode']} for f in frames])
            csv_write(folder/'input_checksums.csv',inputs)
            if cfg.get('expected_subjects') is not None and len({f['subject'] for f in frames})!=cfg['expected_subjects']:
                raise ValueError('Accepted subject count differs from expected_subjects. Inspect audit before changing configuration.')
            if args.stage=='prepare':return
            results,preds=nested(frames,cfg,store,folder)
            if args.stage in ('all','nested','analyze'):analyze(preds,cfg,folder)
            if args.stage in ('all','permutation'):
                synthetic_calibration(cfg,store,folder);measured_order(frames,results,cfg,store,folder);permutation_classification(frames,results,cfg,store,folder)
            if args.stage in ('all','robustness'):
                robustness.run(frames,results,cfg,store,folder);robustness.synthetic_cycle_detection(cfg,store,folder)
            if args.stage in ('all','fidelity'):fidelity.run(frames,results,cfg,store,folder)
            json_write(folder/(args.stage+'_complete.json'),{'stage':args.stage,'data_mode':cfg['data_mode'],'new_tasks':store.completed,'reused_tasks':store.reused})
            print('COMPLETE',args.stage,'new=',store.completed,'reused=',store.reused,flush=True)
        except InterruptedError as e:
            json_write(folder/'interrupted.json',{'reason':str(e),'completed_new_tasks':store.completed});print(str(e),flush=True);raise SystemExit(75)
        except Exception as e:
            json_write(folder/'last_error.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()});raise
if __name__=='__main__':main()
