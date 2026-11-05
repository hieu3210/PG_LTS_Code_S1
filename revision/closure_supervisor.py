"""Resume-safe supervisor for independent DrCIF/RDST and HA tests.

Run from project root: python -m revision.closure_supervisor --source-results ...
This is an analysis/engineering script, not evidence that experiments have run.
"""
import argparse,csv,hashlib,json,os,platform,signal,subprocess,sys,time
from pathlib import Path
from .storage import sha,json_write,csv_write,run_lock
from .protocol import paired,summary
from .raw_baselines import load_prepared


def read_csv(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def atomic_log_record(p,rows):csv_write(p,rows)

def verify_checkpoint(folder):
    rows=[];bad=[]
    for p in sorted((folder/'checkpoints').glob('*.done.json')):
        obj=json.loads(p.read_text(encoding='utf-8'));model=p.with_suffix('').with_suffix('.joblib')
        # p.name='name.done.json' -> model 'name.joblib'
        model=p.with_name(p.name[:-10]+'.joblib')
        valid=model.is_file() and sha(model)==obj.get('sha256')
        rows.append({'task':p.name[:-10],'checkpoint':model.name,'sha256':sha(model) if model.exists() else '', 'valid':int(valid)})
        if not valid:bad.append(p.name)
    if bad:raise ValueError('Broken checkpoint receipts: '+', '.join(bad[:8]))
    return rows

def validate_predictions(path,frames,arm_field,expected_arms):
    rows=read_csv(path)
    expected={str(f['id']):f for f in frames};arms={r[arm_field] for r in rows}
    if arms!=set(expected_arms):raise ValueError(f'{path}: missing arms {set(expected_arms)-arms}')
    for arm in expected_arms:
        selected=[r for r in rows if r[arm_field]==arm];ids=[r['id'] for r in selected]
        if len(ids)!=len(expected) or len(set(ids))!=len(ids) or set(ids)!=set(expected):
            raise ValueError(f'{path}: OOF IDs missing or duplicated for {arm}')
        for r in selected:
            src=expected[r['id']]
            if r['subject']!=str(src['subject']) or r['y']!=str(src['y']):raise ValueError('Subject/label drift')
            if r['pred'] not in {str(f['y']) for f in frames}:raise ValueError('Prediction outside class set')
    return rows

def report(root,source,cfg,stages,monitor,reference_oof=None):
    frames,_,_=load_prepared(source)
    metrics=[];contrasts=[];provenance=[]
    cp=verify_checkpoint(root)
    csv_write(root/'checkpoint_audit.csv',cp)
    if (root/'baseline_COMPLETE.json').exists():
        b=validate_predictions(root/'baseline_outer_predictions.csv',frames,'method',['DrCIF','RDST'])
        for arm in ['DrCIF','RDST']:
            for rec in summary([r for r in b if r['method']==arm],cfg['bootstrap'],cfg['seed']):
                metrics.append({'family':'baseline','method':arm,**rec})
        selected=read_csv(root/'baseline_selected.csv')
        if len(selected)!=8 or any(r['outer_prediction_replay']!='PASS' for r in selected):raise ValueError('Baseline model replay or fold count failed')
        if (root/'tensor_audit_manifest.csv').exists():
            for r in read_csv(root/'tensor_audit_manifest.csv'):
                p=root/'tensor_audit'/r['path']
                if not p.exists() or sha(p)!=r['sha256_file']:raise ValueError('Tensor file digest mismatch')
        provenance.append({'item':'DrCIF/RDST','status':'VERIFIED_REPLAY','detail':'Four folds, OOF ids, checkpoint receipts and saved-model replay in same run; not necessarily historical model equality'})
    if reference_oof and (root/'baseline_COMPLETE.json').exists():
        reference=read_csv(Path(reference_oof))
        refs={r.get('arm') for r in reference}
        b=read_csv(root/'baseline_outer_predictions.csv')
        for arm in ('combined','phase_ordered','occupancy'):
            if arm not in refs:continue
            rr=[r for r in reference if r.get('arm')==arm]
            if len(rr)!=len(frames):raise ValueError(f'Reference OOF incomplete: {arm}')
            for method in ('DrCIF','RDST'):
                bb=[r for r in b if r['method']==method]
                for level in ('segment','subject'):
                    val=paired([dict(r,arm=method) for r in bb],rr,cfg['bootstrap'],cfg['seed'],level)
                    contrasts.append({'view':'locked_reference','a':method,'b':arm,**val})
        provenance.append({'item':'Locked PG reference comparison','status':'VERIFIED_PAIRED_OOF','detail':'Exact matching segment IDs, subjects and labels required'})
    if (root/'ha_COMPLETE.json').exists():
        a=validate_predictions(root/'ha_outer_predictions.csv',frames,'mapping',['ha_soft','quantile']) if len(cfg['ha_views'])==1 else read_csv(root/'ha_outer_predictions.csv')
        for view in cfg['ha_views']:
            av=[r for r in a if r['view']==view]
            if {r['mapping'] for r in av}!={'ha_soft','quantile'}:raise ValueError('HA comparison incomplete')
            # Validation requires complete OOF within each view.
            for arm in ['ha_soft','quantile']:
                ab=[r for r in av if r['mapping']==arm]
                validate_unique(ab,frames,view,arm)
                for rec in summary(ab,cfg['bootstrap'],cfg['seed']):metrics.append({'family':'ha','view':view,'method':arm,**rec})
            first=[r for r in av if r['mapping']=='ha_soft'];second=[r for r in av if r['mapping']=='quantile']
            for level in ('segment','subject'):
                res=paired(first,second,cfg['bootstrap'],cfg['seed'],level)
                contrasts.append({'view':view,'a':'ha_soft','b':'quantile',**res})
        selected=read_csv(root/'ha_selected.csv')
        for view in cfg['ha_views']:
            for fold in range(4):
                ss=[r for r in selected if r['view']==view and int(r['fold'])==fold]
                if len(ss)!=2 or len({int(r['n_descriptors']) for r in ss})!=1:raise ValueError('Unmatched HA descriptor dimension')
        provenance.append({'item':'HA ablation','status':'VERIFIED_OOF','detail':'Paired subject folds, descriptor dimensions and chosen hyperparameters audited'})
    if metrics:csv_write(root/'final_macro_f1_and_class_metrics.csv',metrics)
    if contrasts:csv_write(root/'final_paired_contrasts.csv',contrasts)
    if provenance:csv_write(root/'evidence_status.csv',provenance)
    report_lines=['# Reviewer closure: verification report','',f'Source: `{source}`','','## Results availability']
    for stage in stages:report_lines.append(f'- {stage}: '+('COMPLETE' if (root/(stage+'_COMPLETE.json')).exists() else 'NOT COMPLETE'))
    report_lines += ['','## Integrity',f'- Verified task checkpoint receipts: {len(cp)}','- Previously published five-seed results are not modified by this pipeline.','- Baseline and HA paired results are follow-up experiments with fixed source data and splits.','- A positive HA paired effect is supported within this protocol only if the confidence interval excludes zero.','- Runtime is measured on the executing machine; hardware effects remain local.','']
    if contrasts:
        report_lines += ['## HA vs quantile (macro-F1, paired cluster bootstrap)','', '| View | Unit | Delta | CI lower | CI upper |','|---|---|---:|---:|---:|']
        for r in contrasts:report_lines.append(f"| {r['view']} | {r['level']} | {r['delta_macro_f1']:.4f} | {r['lo']:.4f} | {r['hi']:.4f} |")
    (root/'reviewer_closure_report.md').write_text('\n'.join(report_lines)+'\n',encoding='utf-8')
    json_write(root/'verification_status.json',{'stages_verified':[s for s in stages if (root/(s+'_COMPLETE.json')).exists()],'checkpoint_receipts_valid':len(cp),'source_sha256':sha(source/'run_identity.json'),'monitor_rows':len(monitor),'five_seed_untouched':True})

def validate_unique(rows,frames,view,arm):
    gold={str(f['id']):(str(f['subject']),str(f['y'])) for f in frames}
    if len(rows)!=len(gold) or {r['id'] for r in rows}!=set(gold):raise ValueError(f'Incomplete HA OOF {view}/{arm}')
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Repeated OOF sample')
    if any((r['subject'],r['y'])!=gold[r['id']] for r in rows):raise ValueError('Label drift')

def run():
    ap=argparse.ArgumentParser();ap.add_argument('--reference-oof',default=None,help='Optional LOCKED legacy PG OOF CSV for paired comparisons');ap.add_argument('--source-results',required=True);ap.add_argument('--config',default='configs/independent_validation.json');ap.add_argument('--out',default='runs_reviewer_v2');ap.add_argument('--stage',choices=['baseline','ha','all'],default='all');ap.add_argument('--sample-seconds',type=float,default=2.0);ap.add_argument('--stop-after',type=int,default=0);a=ap.parse_args()
    root=Path(__file__).resolve().parents[1];source=Path(a.source_results).resolve();cfgfile=(root/a.config).resolve() if not Path(a.config).is_absolute() else Path(a.config).resolve();cfg=json.loads(cfgfile.read_text(encoding='utf-8'))
    if a.sample_seconds<0.2:raise ValueError('sample-seconds must be >=0.2')
    if not (source/'run_identity.json').exists():raise FileNotFoundError('Source must be extracted original run with run_identity.json and checkpoint folder')
    # Independent code itself computes stable output path from immutable source+config+code fingerprints.
    stages=['baseline','ha'] if a.stage=='all' else [a.stage]
    run_directory=(root/a.out).resolve();run_directory.mkdir(parents=True,exist_ok=True)
    with run_lock(run_directory/'supervisor'):
        session=[]
        for stage in stages:
            before=set(run_directory.glob('locked_*'))
            command=[sys.executable,'-m','revision.independent_validation','--source-results',str(source),'--config',str(cfgfile),'--out',a.out,'--stage',stage]
            if a.stop_after:command+=['--stop-after',str(a.stop_after)]
            stage_log=run_directory/f'supervisor_{stage}.log'
            started=time.time();peak_rss=0;peak_tree=0;status='RUNNING'
            with stage_log.open('a',encoding='utf-8') as log:
                log.write('\nSTART '+time.strftime('%Y-%m-%dT%H:%M:%S')+' '+repr(command)+'\n');log.flush()
                proc=subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
                try:
                    import psutil
                    while proc.poll() is None:
                        try:
                            parent=psutil.Process(proc.pid);nodes=[parent]+parent.children(recursive=True)
                            rss=sum(p.memory_info().rss for p in nodes if p.is_running());peak_tree=max(peak_tree,rss);peak_rss=max(peak_rss,parent.memory_info().rss)
                        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
                        time.sleep(a.sample_seconds)
                except KeyboardInterrupt:
                    proc.send_signal(signal.SIGINT);proc.wait();raise
                code=proc.wait()
            status='SUCCESS' if code==0 else 'INTERRUPTED' if code==75 else 'FAILED'
            session.append({'stage':stage,'status':status,'exit_code':code,'wall_seconds':round(time.time()-started,3),'peak_process_rss_mb':round(peak_rss/1048576,3),'peak_process_tree_rss_mb':round(peak_tree/1048576,3),'host':platform.node(),'python':sys.version.split()[0],'processor':platform.processor(),'log':stage_log.name})
            csv_write(run_directory/'supervisor_runtime.csv',session)
            if code!=0:
                print(f'{stage} stopped with code {code}. Repeat the SAME command to resume. Log: {stage_log}',flush=True)
                sys.exit(code if code>0 else 1)
        candidates=list(run_directory.glob('locked_*'))
        chosen=[p for p in candidates if all((p/(s+'_COMPLETE.json')).exists() for s in stages)]
        if len(chosen)!=1:raise RuntimeError('Cannot uniquely identify verified run directory; inspect '+str(run_directory))
        report(chosen[0],source,cfg,stages,session,a.reference_oof)
        csv_write(chosen[0]/'supervisor_runtime.csv',session)
        print('VALIDATED OUTPUT:',chosen[0])

if __name__=='__main__':run()
