#!/usr/bin/env python3
"""Read-only capture of a PG-LTS v140 run; never fits or silently invents evidence."""
import argparse, csv, hashlib, importlib.metadata, importlib.util, io, json, os
from pathlib import Path
import platform, shutil, sys, tempfile, time, zipfile
from contextlib import contextmanager
import numpy as np
VERSION='1.4.1'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def atomic(p,writer):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    fd,t=tempfile.mkstemp(prefix='.'+p.name,suffix='.partial',dir=p.parent);os.close(fd)
    try:
        writer(Path(t))
        with open(t,'rb') as f:os.fsync(f.fileno())
        os.replace(t,p)
        if os.name=='posix':
            d=os.open(p.parent,os.O_RDONLY)
            try:os.fsync(d)
            finally:os.close(d)
    finally:
        if os.path.exists(t):os.unlink(t)

def jsonwrite(p,x):atomic(p,lambda t:t.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf8'))
def readcsv(p):
    with open(p,newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def csvwrite(p,rows,fields=None):
    rows=list(rows);fields=fields or list(dict.fromkeys(k for r in rows for k in r))
    def w(t):
        with open(t,'w',newline='',encoding='utf8') as f:
            a=csv.DictWriter(f,fieldnames=fields);a.writeheader();a.writerows(rows)
    atomic(p,w)

@contextmanager
def lock(p):
    with open(p,'a+b') as f:
        f.seek(0);f.write(b'0');f.flush();f.seek(0)
        try:
            if os.name=='nt':
                import msvcrt;msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl;fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:raise RuntimeError('Another exporter owns this output folder.')
        try:yield
        finally:
            f.seek(0)
            if os.name=='nt':
                import msvcrt;msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(f,fcntl.LOCK_UN)

class Store:
    def __init__(self,out,stop=0):self.out=out;self.stop=stop;self.done=0
    def task(self,key,build):
        p=self.out/'receipts'/(hashlib.sha256(key.encode()).hexdigest()+'.json')
        if p.exists():
            try:
                r=json.loads(p.read_text());assert r['key']==key and all((self.out/q).is_file() and sha(self.out/q)==v for q,v in r['outputs'].items())
                print('REUSE',key,flush=True);return
            except (ValueError,KeyError,AssertionError,OSError):pass
        print('RUN',key,flush=True);outputs=build()
        jsonwrite(p,{'key':key,'outputs':{str(q.relative_to(self.out)):sha(q) for q in outputs}})
        self.done+=1
        if self.stop and self.done>=self.stop:raise InterruptedError('Intentional stop; rerun the same command to resume.')
    def capture(self,src,rel,expected):
        dest=self.out/rel
        def b():
            if sha(src)!=expected:raise RuntimeError('Input changed while capturing: '+str(src))
            atomic(dest,lambda t:shutil.copyfile(src,t))
            if sha(dest)!=expected:raise RuntimeError('Input changed during copy: '+str(src))
            return [dest]
        self.task('copy/'+rel,b)

def metrics(y,p):
    labels=sorted(set(y));return float(np.mean([2*sum(a==c and b==c for a,b in zip(y,p))/(sum(a==c for a in y)+sum(b==c for b in p)) if sum(a==c for a in y)+sum(b==c for b in p) else 0 for c in labels]))

def array_info(v):
    a=np.asarray(v);h=hashlib.sha256()
    h.update(str(a.dtype).encode());h.update(json.dumps(a.shape).encode())
    h.update(json.dumps(a.tolist(),ensure_ascii=False,default=str).encode() if a.dtype.hasobject else np.ascontiguousarray(a).tobytes())
    return {'shape':json.dumps(a.shape),'dtype':str(a.dtype),'sha256':h.hexdigest()}

def inspect_payload(path,rel):
    import joblib
    obj=joblib.load(path);models=[];frames=[];inner=[];seen=set()
    def walk(x,loc='root',depth=0):
        if depth>6 or id(x) in seen:return
        seen.add(id(x))
        if isinstance(x,dict):
            if all(k in x for k in ('id','subject','y','amp','cpx','window')):
                for k in ('amp','cpx','window'):
                    frames.append(dict(payload=rel,location=loc,id=x['id'],subject=x['subject'],y=x['y'],component=k,**array_info(x[k])))
                return
            if 'y' in x and 'pred' in x and not isinstance(x['y'],str):
                yy=list(x['y']);pp=list(x['pred'])
                if len(yy)!=len(pp):raise ValueError('Inner y/pred length mismatch')
                for i,(y,p) in enumerate(zip(yy,pp)):inner.append(dict(payload=rel,location=loc,ordinal=i,y=str(y),pred=str(p),id_scope='ordinal only; sample ID not authenticated',converged=x.get('converged','unknown')))
            for k,v in x.items():
                if k not in ('y','pred'):walk(v,loc+'.'+str(k),depth+1)
        elif isinstance(x,(tuple,list)):
            # Prepared frames can exceed 1000; never truncate.
            for i,v in enumerate(x):walk(v,loc+f'[{i}]',depth+1)
        elif hasattr(x,'predict') or hasattr(x,'scores'):
            r={'payload':rel,'location':loc,'class':type(x).__module__+'.'+type(x).__name__,'scope':'attributes only; not inference replay'}
            for k in ('c','options','classes_','names','groups','converged','converged_','n_iter_','objective_','max_iter','tol','n_estimators','max_shapelets'):
                if hasattr(x,k):
                    v=getattr(x,k);r[k]=json.dumps(v.tolist() if isinstance(v,np.ndarray) else v,default=str,ensure_ascii=False)
            models.append(r)
            for k in ('clf','estimator','model'):
                if hasattr(x,k):walk(getattr(x,k),loc+'.'+k,depth+1)
    walk(obj);return models,frames,inner

def export_bootstrap(rows,store,seed,nboot,chunk):
    keys=sorted({(r['arm'],int(r['top_k'])) for r in rows})
    for arm,k in keys:
        rr=[r for r in rows if r['arm']==arm and int(r['top_k'])==k]
        if len({r['id'] for r in rr})!=len(rr):raise ValueError('Duplicate fidelity IDs')
        by={}
        for r in rr:
            s=r['subject'];d=by.setdefault(s,{'y':r['y'],'n':0,'agree':0})
            if d['y']!=r['y']:raise ValueError('Subject with multiple species')
            d['n']+=1;d['agree']+=int(r['full_pred']==r['sparse_pred'])
            if 'agreement' in r and int(r['agreement'])!=int(r['full_pred']==r['sparse_pred']):raise ValueError('Invalid stored agreement')
        subs=sorted(by);groups=[sorted(s for s in subs if by[s]['y']==c) for c in sorted({v['y'] for v in by.values()})]
        si={s:i for i,s in enumerate(subs)};ns=np.array([by[s]['n'] for s in subs]);ok=np.array([by[s]['agree'] for s in subs])
        tag=hashlib.sha256(f'{arm}/{k}'.encode()).hexdigest()[:12];rng=np.random.default_rng(seed)
        chunks=[]
        for start in range(0,nboot,chunk):
            end=min(start+chunk,nboot);counts=np.zeros((end-start,len(subs)),np.int32)
            # Advance RNG even when a committed chunk will be reused.
            for i in range(end-start):
                for ss in groups:
                    for s in rng.choice(ss,len(ss),replace=True):counts[i,si[s]]+=1
            dp=store.out/f'bootstrap/{tag}/draws_{start:06}.csv';vp=store.out/f'bootstrap/{tag}/values_{start:06}.csv';chunks.append(vp)
            def build(counts=counts,start=start,dp=dp,vp=vp):
                csvwrite(dp,({'arm':arm,'top_k':k,'replicate':start+i,'subject':s,'multiplicity':int(counts[i,j])} for i in range(len(counts)) for j,s in enumerate(subs)),['arm','top_k','replicate','subject','multiplicity'])
                den=counts@ns;num=counts@ok
                csvwrite(vp,[{'arm':arm,'top_k':k,'replicate':start+i,'matching_segments':int(num[i]),'sampled_segments':int(den[i]),'agreement':float(num[i]/den[i])} for i in range(len(counts))]);return [dp,vp]
            store.task(f'bootstrap/{tag}/{start}',build)
        def finish(chunks=chunks,tag=tag,arm=arm,k=k,by=by,ns=ns,ok=ok,subs=subs):
            vv=[float(r['agreement']) for p in chunks for r in readcsv(p)];lo,hi=np.quantile(vv,[.025,.975],method='linear')
            p=store.out/f'bootstrap/{tag}/summary.json';jsonwrite(p,dict(arm=arm,top_k=k,agreement=float(ok.sum()/ns.sum()),lo=float(lo),hi=float(hi),seed=seed,n_boot=nboot,n_subjects=len(subs),method='canonical audit recomputation; NOT claimed to be original run resampling',stratified_by_species=True,subject_order=subs,quantile='numpy linear',rng='numpy default_rng PCG64',weighting='segment weighted; repeated subject repeats all its segments'))
            return [p]
        store.task('bootstrap-summary/'+tag,finish)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results',required=True,type=Path);ap.add_argument('--source-root',required=True,type=Path)
    ap.add_argument('--source-run',type=Path);ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--extra-source-root',type=Path,action='append',default=[])
    ap.add_argument('--seed',type=int,default=20261102);ap.add_argument('--bootstrap',type=int,default=5000);ap.add_argument('--chunk',type=int,default=250)
    ap.add_argument('--trust-joblib',action='store_true');ap.add_argument('--stop-after',type=int,default=0)
    ap.add_argument('--pack',action='store_true');args=ap.parse_args()
    if args.bootstrap<2 or args.chunk<1:ap.error('bootstrap >=2 and chunk >=1 required')
    result=args.results.resolve();source=args.source_root.resolve();out=args.out.resolve()
    if not result.is_dir() or not source.is_dir():ap.error('results and source-root must exist')
    identity=json.loads((result/'identity.json').read_text());run=args.source_run or Path(identity['source']);run=run.resolve()
    if not run.is_dir():ap.error('Original source run missing; specify --source-run on this machine')
    if out==source or out==result or out==run:ap.error('Output must be a separate folder')
    out.mkdir(parents=True,exist_ok=True)
    with lock(out/'export.lock'):
        ignored={'.git','.venv','venv','__pycache__','node_modules','runs','runs_reviewer_closure','checkpoints','output','audit_export'}
        inputs=[]
        def add(p,rel):
            if p.is_symlink():return
            inputs.append({'source':str(p),'export_path':rel,'sha256':sha(p),'bytes':p.stat().st_size})
        for p in sorted(result.rglob('*')):
            if p.is_file() and out not in p.parents and p.suffix.lower() in {'.csv','.json','.log','.diff','.md','.py'}:add(p,'results/'+str(p.relative_to(result)))
        for root,dirs,files in os.walk(source):
            dirs[:]=sorted(d for d in dirs if d not in ignored and not d.startswith('.') and (Path(root)/d).resolve()!=out and not (Path(root)/d).is_symlink())
            for n in sorted(files):
                p=Path(root)/n
                if p.suffix in {'.py','.toml','.yaml','.yml','.ini','.cfg','.sh','.bat','.ps1'} or n.startswith('requirements') or n in {'uv.lock','poetry.lock'}:add(p,'source/'+str(p.relative_to(source)))
        for idx,extra in enumerate(args.extra_source_root):
            extra=extra.resolve()
            if not extra.is_dir():raise ValueError('Extra source folder missing: '+str(extra))
            for root,dirs,files in os.walk(extra):
                dirs[:]=sorted(d for d in dirs if d not in ignored and not d.startswith('.') and not (Path(root)/d).is_symlink() and (Path(root)/d).resolve()!=out)
                for n in sorted(files):
                    p=Path(root)/n
                    if p.suffix in {'.py','.toml','.yaml','.yml','.ini','.cfg'} or n.startswith('requirements'):
                        add(p,f'extra_source_{idx}/'+str(p.relative_to(extra)))
        for label,folder in [('original_run',run),('closure_run',result)]:
            for p in sorted(folder.rglob('*')):
                if not p.is_file() or out in p.parents:continue
                if p.suffix in {'.joblib','.pkl','.pickle','.npz','.npy','.csv','.json','.log'} or p.name.endswith('.done.json') or p.name in {'config.json','splits.csv','run.json','analysis_identity.json','model_metadata.json','nested_protocol.json','selected_configs.csv','inner_search.csv'}:
                    add(p,label+'/'+str(p.relative_to(folder)))
        # Capture the installed implementation, including a local aeon variance patch.
        for package,relative in [('aeon','base/_base_collection.py'),('aeon','__init__.py')]:
            try:
                spec=importlib.util.find_spec(package)
                if spec and spec.submodule_search_locations:
                    p=Path(next(iter(spec.submodule_search_locations)))/relative
                    if p.is_file():add(p,'environment/installed/'+package+'/'+relative)
            except (ImportError,ValueError):pass
        inputs=sorted({r['export_path']:r for r in inputs}.values(),key=lambda r:r['export_path'])
        env={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'packages':dict(sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions() if d.metadata['Name'])),'runtime_variables':{k:os.environ.get(k) for k in ['PGLTS_AEON_JOBS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','PYTHONHASHSEED']},'exporter_sha256':sha(Path(__file__))}
        config={'version':VERSION,'inputs':inputs,'environment':env,'seed':args.seed,'bootstrap':args.bootstrap,'chunk':args.chunk,'trust_joblib':args.trust_joblib}
        pin=out/'capture_identity.json'
        if pin.exists() and json.loads(pin.read_text())!=config:raise RuntimeError('Inputs/environment/options changed. Use a NEW --out folder; old export is preserved.')
        if not pin.exists():jsonwrite(pin,config)
        (out/'COMPLETE.json').unlink(missing_ok=True)
        store=Store(out,args.stop_after)
        for r in inputs:store.capture(Path(r['source']),r['export_path'],r['sha256'])
        def envtask():
            p=out/'environment/environment.json';jsonwrite(p,env);return [p]
        store.task('environment',envtask)
        payloads=[r for r in inputs if r['export_path'].startswith(('original_run/','closure_run/')) and r['source'].endswith(('.joblib','.pkl','.pickle'))]
        receipts=[]
        for r in payloads:
            src=Path(r['source']);receipt=src.with_suffix('.done.json');status='MISSING';key=''
            if receipt.is_file():
                try:
                    x=json.loads(receipt.read_text());key=x.get('key','');status='MATCH' if x.get('sha256')==r['sha256'] else 'MISMATCH'
                except Exception:status='UNREADABLE'
            receipts.append(dict(payload=r['export_path'],bytes=r['bytes'],sha256=r['sha256'],receipt_status=status,receipt_key=key))
        csvwrite(out/'checkpoint_receipt_audit.csv',receipts,['payload','bytes','sha256','receipt_status','receipt_key'])
        identity_checks=[]
        for family in ('prepared_receipts','source_models'):
            for expected in identity.get(family,[]):
                matches=[r for r in inputs if r['export_path'].startswith('original_run/') and Path(r['source']).name==expected['name']]
                status='MATCH' if len(matches)==1 and matches[0]['sha256']==expected['sha256'] else ('MISSING' if not matches else 'MISMATCH_OR_AMBIGUOUS')
                identity_checks.append({'family':family,'name':expected['name'],'expected_sha256':expected['sha256'],'observed_sha256':matches[0]['sha256'] if len(matches)==1 else '', 'status':status})
        csvwrite(out/'identity_payload_audit.csv',identity_checks,['family','name','expected_sha256','observed_sha256','status'])

        if args.trust_joblib:
            sys.dont_write_bytecode=True
            sys.path.insert(0,str(source));sys.path.insert(0,str(source/'src'))
            for extra in args.extra_source_root:
                sys.path.extend([str(extra.resolve()),str(extra.resolve()/'src')])
            for r in payloads:
                tag=hashlib.sha256(r['export_path'].encode()).hexdigest()[:16]
                def inspect(r=r,tag=tag):
                    files=[]
                    try:
                        models,frames,inner=inspect_payload(out/r['export_path'],r['export_path'])
                        for name,rows in [('model_attributes',models),('prepared_components',frames),('inner_validation',inner)]:
                            if rows:
                                p=out/f'inspection/{tag}/{name}.csv';csvwrite(p,rows);files.append(p)
                        status={'payload':r['export_path'],'status':'INSPECTED','models':len(models),'prepared_components':len(frames),'inner_predictions':len(inner)}
                    except Exception as e:status={'payload':r['export_path'],'status':'FAILED','error':type(e).__name__+': '+str(e)}
                    p=out/f'inspection/{tag}/status.json';jsonwrite(p,status);return files+[p]
                store.task('inspect/'+r['export_path'],inspect)
        fpath=out/'results/fidelity_predictions.csv'
        if not fpath.exists():raise RuntimeError('Missing fidelity_predictions.csv')
        fr=readcsv(fpath);export_bootstrap(fr,store,args.seed,args.bootstrap,args.chunk)
        checks=[];summaries=[]
        for p in sorted((out/'bootstrap').glob('*/summary.json')):
            q=json.loads(p.read_text());orig=out/f"results/fidelity_ci_{q['arm']}_top{q['top_k']}.csv"
            if orig.exists():
                z=readcsv(orig)[0];q.update(original_lo=float(z['lo']),original_hi=float(z['hi']),original_agreement=float(z['agreement']))
                q['ci_max_abs_difference']=max(abs(q['lo']-q['original_lo']),abs(q['hi']-q['original_hi']))
                q['comparison']='EXACT' if q['ci_max_abs_difference']<1e-12 else 'DIFFERENT_DRAW_PROCEDURE_OR_SEED_OR_ORDER; inspect captured original code'
            summaries.append({k:v for k,v in q.items() if k!='subject_order'})
        csvwrite(out/'fidelity_bootstrap_comparison.csv',summaries)
        pooled=[]
        for arm,k in sorted({(r['arm'],int(r['top_k'])) for r in fr}):
            rr=[r for r in fr if r['arm']==arm and int(r['top_k'])==k]
            pooled.append(dict(arm=arm,top_k=k,n=len(rr),subjects=len({r['subject'] for r in rr}),full_macro_f1=metrics([r['y'] for r in rr],[r['full_pred'] for r in rr]),rule_macro_f1=metrics([r['y'] for r in rr],[r['sparse_pred'] for r in rr]),agreement=float(np.mean([r['full_pred']==r['sparse_pred'] for r in rr])),pooled_score_rmse=float(np.sqrt(np.mean([float(r['score_rmse'])**2 for r in rr]))),mean_segment_score_rmse=float(np.mean([float(r['score_rmse']) for r in rr])),probability_mae=float(np.mean([float(r['probability_mae']) for r in rr]))))
        csvwrite(out/'fidelity_pooled_metrics.csv',pooled)
        checks.append({'item':'source_payload_identity','status':'PASS' if identity_checks and all(r['status']=='MATCH' for r in identity_checks) else 'INCOMPLETE_OR_MISMATCH','scope':'compares original payload bytes to v140 identity.json'})
        checks.extend([{'item':'payload_receipts','status':'PASS' if receipts and all(r['receipt_status']=='MATCH' for r in receipts) else 'INCOMPLETE_OR_MISMATCH','scope':'SHA consistency only; not proof of historical training'}, {'item':'actual_model_inputs','status':'PENDING_SOURCE_AND_PAYLOAD_REVIEW','scope':'prepared component hashes are not baseline tensor hashes'}, {'item':'inner_fit_and_outer_inference_replay','status':'NOT_PERFORMED','scope':'no retraining or generic guessed predict interface'}, {'item':'bootstrap_fidelity','status':'CANONICAL_RECOMPUTATION_EXPORTED','scope':'original procedure captured as source; exact historical draws not fabricated'}])
        failures=[json.loads(p.read_text()) for p in (out/'inspection').glob('*/status.json') if json.loads(p.read_text()).get('status')=='FAILED']
        csvwrite(out/'evidence_status.csv',checks)
        csvwrite(out/'input_file_manifest.csv',inputs)
        jsonwrite(out/'COMPLETE.json',{'export_completed':True,'scientific_audit_complete':False,'copied_files':len(inputs),'payloads':len(payloads),'payload_inspection_failures':failures,'bootstrap_groups':len(summaries),'scope':'capture and saved-output checks only; see evidence_status.csv'})
        manifest={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in {'SHA256SUMS.json','export.lock'} and not p.name.endswith('.partial')}
        jsonwrite(out/'SHA256SUMS.json',manifest)
        if args.pack:
            target=out.with_suffix('.zip')
            def pack(t):
                with zipfile.ZipFile(t,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
                    for rel in list(manifest)+['SHA256SUMS.json']:z.write(out/rel,Path(out.name)/rel)
                with zipfile.ZipFile(t) as z:
                    if z.testzip():raise RuntimeError('ZIP CRC failure')
            atomic(target,pack);print('PACKAGE',target)
        print('EXPORT COMPLETE. Scientific verification remains scoped in evidence_status.csv.')
if __name__=='__main__':
    try:main()
    except InterruptedError as e:print(e,file=sys.stderr);sys.exit(75)
    except Exception as e:print(type(e).__name__+': '+str(e),file=sys.stderr);sys.exit(2)
