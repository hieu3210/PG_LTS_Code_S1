"""Fixed-lattice disjoint cohorts and recording-shared conditional order controls.
No fitting or outer-test model selection. Whole-record filtering is NOT undone.
"""
from pathlib import Path
import argparse,csv,itertools,json,math,traceback
import numpy as np
import joblib
from threadpoolctl import threadpool_limits
from . import ROOT
from .storage import Store,run_lock,json_write,csv_write,sha,digest

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def environment():
    import platform,importlib.metadata
    out={'python':platform.python_version(),'platform':platform.platform()}
    for name in ['numpy','scipy','scikit-learn','joblib','librosa']:
        try:out[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:out[name]='not installed'
    return out


def bh(p):
    p=np.array(p,float);order=np.argsort(p);q=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1];out=np.empty_like(p);out[order]=np.minimum(q,1);return out


def statistic(mu):
    # Identical aggregation to revision.order.statistic; no heavy front-end imports.
    mu=np.asarray(mu,float);L=mu.shape[1];O=mu.mean(1)
    M=np.einsum('nlra,nlrb->nrab',mu[:,:-1],mu[:,1:])/(L-1)
    E=(L*L*np.einsum('nra,nrb->nrab',O,O)-np.einsum('nlra,nlrb->nrab',mu,mu))/(L*(L-1))
    return float(np.abs((M-E).mean(0)).sum((-2,-1)).mean())


VERSION='1.2.0'
GROUPS=['recording_cyclic_shared','recording_block_shared']


def parse_id(identity):
    recording,start,length=identity.rsplit(':',2);start=int(start);length=int(length)
    if not recording or start<0 or length!=2000 or start%1000:
        raise ValueError('Expected legacy IDs recording:start:2000, on the 1000-sample grid: '+identity)
    return recording,start,length


def read_manifest(path):
    """Accept legacy segment_manifest or saved OOF; dedup repeated arms only."""
    records={};seen_arm_ids=set()
    for row in read_csv(path):
        identity=row['id'];rec,start,length=parse_id(identity);species=row.get('species') or row.get('y')
        if 'arm' in row:
            key=(row['arm'],identity)
            if key in seen_arm_ids:raise ValueError('Duplicate ID within one OOF arm: '+identity)
            seen_arm_ids.add(key)
        if not species or not row.get('subject'):raise ValueError('Missing subject/species')
        item={'id':identity,'subject':row['subject'],'species':species,'recording':rec,'start':start,'stop':start+length}
        if identity in records:
            if records[identity]!=item:raise ValueError('Inconsistent metadata for duplicate ID: '+identity)
            if 'arm' not in row:raise ValueError('Duplicate ID in a non-OOF manifest: '+identity)
        records[identity]=item
    if not records:raise ValueError('Empty manifest')
    byrecord={}
    for x in records.values():
        byrecord.setdefault(x['recording'],set()).add((x['subject'],x['species']))
    if any(len(v)!=1 for v in byrecord.values()):raise ValueError('Recording assigned to different subjects/species')
    return sorted(records.values(),key=lambda x:(x['recording'],x['start']))


def cohort_names(include_reference=False):
    return ['grid0','grid1000']+(['overlap_reference'] if include_reference else [])


def selected(row,cohort,gap=0):
    if cohort=='overlap_reference':return True
    offset={'grid0':0,'grid1000':1000}[cohort]
    return row['start']%(2000+gap)==offset


def plan(rows,folder,gap=0,include_reference=False):
    keep=[];audit=[];subjects=[];mapping={}
    for cohort in cohort_names(include_reference):
        mapping[cohort]=[x for x in rows if selected(x,cohort,gap)]
        for x in rows:keep.append({'cohort':cohort,**x,'selected':selected(x,cohort,gap),'rule':'fixed timestamp lattice; no fallback or performance-based choice','interval_convention':'[start,stop), samples at 1 kHz'})
        for rec in sorted({x['recording'] for x in rows}):
            items=[x for x in mapping[cohort] if x['recording']==rec];overlap=sum(max(0,a['stop']-b['start']) for a,b in zip(items,items[1:]));gap_min=min((b['start']-a['stop'] for a,b in zip(items,items[1:])),default=None)
            if cohort!='overlap_reference' and overlap:raise AssertionError('Disjoint cohort overlaps')
            audit.append({'cohort':cohort,'recording':rec,'selected_segments':len(items),'adjacent_pair_overlap_samples':overlap,'minimum_gap_samples':gap_min,'raw_intervals_disjoint':overlap==0})
        for subject in sorted({x['subject'] for x in rows}):
            all_s=[x for x in rows if x['subject']==subject];chosen=[x for x in mapping[cohort] if x['subject']==subject]
            subjects.append({'cohort':cohort,'subject':subject,'species':all_s[0]['species'],'original_segments':len(all_s),'selected_segments':len(chosen),'recordings_with_segments':len({x['recording'] for x in chosen}),'status':'available' if chosen else 'no_segments_on_fixed_grid'})
    csv_write(folder/'cohort_manifest.csv',keep);csv_write(folder/'cohort_overlap_audit.csv',audit);csv_write(folder/'cohort_subject_counts.csv',subjects)
    return mapping,subjects


def patterns(B,group,block_size=2):
    if B not in (4,8):raise ValueError('This locked follow-up supports the evaluated B=4/8 only.')
    if group=='recording_cyclic_shared':return [tuple((np.arange(B)+s)%B) for s in range(B)]
    if group=='recording_block_shared':
        if block_size<1 or B%block_size:raise ValueError('Block size must divide B')
        n=B//block_size
        if math.factorial(n)>720:raise ValueError('Too many shared block permutations; retain the declared block size.')
        return [tuple(j*block_size+k for j in perm for k in range(block_size)) for perm in itertools.permutations(range(n))]
    raise ValueError('Unknown group')


def recording_table(memberships,permutations):
    """One permutation is shared over all cycles/segments/channels in a record."""
    values=[]
    for index in permutations:
        values.append(float(sum(statistic(mu[:,index,:,:]) for mu in memberships)))
    return np.array(values)


def null_from_tables(tables,Q,seed,exact_limit=4096):
    """Each table is a segment-weighted contribution of one recording."""
    if not tables or Q<1:raise ValueError('Empty recording tables or invalid Q')
    cardinality=math.prod(len(v) for v in tables);observed=float(sum(v[0] for v in tables));rng=np.random.default_rng(seed)
    if cardinality<=exact_limit:
        ix=list(itertools.product(*[range(len(v)) for v in tables]));kind='exact';minimum=1/cardinality
    else:
        ix=np.stack([rng.integers(len(v),size=Q) for v in tables],axis=1).tolist();kind='monte_carlo';minimum=1/(Q+1)
    null=np.array([sum(table[j] for table,j in zip(tables,index)) for index in ix],float)
    k=int(np.sum(null>=observed-1e-12));p=k/len(null) if kind=='exact' else (1+k)/(len(null)+1)
    return {'statistic':observed,'p_value':float(p),'null_values':null.tolist(),'method':kind,'null_draws':len(null),'group_cardinality':str(cardinality),'unique_randomization_vectors':len({tuple(x) for x in ix}),'p_resolution_lower_bound':minimum,'exceedances':k}


def checked_model(source,fold):
    file=source/'checkpoints'/f'outer__{fold}__phase_ordered.joblib';receipt=file.with_suffix('.done.json')
    if not file.exists() or not receipt.exists():raise ValueError('Missing outer phase_ordered model/receipt: '+str(file))
    if json.loads(receipt.read_text())['sha256']!=sha(file):raise ValueError('Invalid model checksum: '+str(file))
    model=joblib.load(file)
    candidates=[json.loads(r['config']) for r in read_csv(source/'selected_configs.csv') if r['arm']=='phase_ordered' and int(r['outer_fold'])==fold]
    if len(candidates)!=1 or model.c!=candidates[0] or model.c.get('view')!='phase_ordered' or not hasattr(model,'cyc_sem'):
        raise ValueError('Outer model does not match recorded selected phase configuration')
    return model,{'name':file.name,'sha256':sha(file)}


def analyze(source,rows,mapping,cfg,store,folder):
    from .raw_baselines import load_prepared
    from .models import get_cycles
    frames,old,receipts=load_prepared(source);lookup={f['id']:f for f in frames};oof=read_csv(source/'oof_predictions.csv');splits=read_csv(source/'splits.csv')
    phase=[r for r in oof if r['arm']=='phase_ordered'];fold_for={}
    if {r['id'] for r in phase}!={x['id'] for x in rows} or len(phase)!=len(rows):raise ValueError('Legacy phase OOF cohort differs from manifest')
    for r in phase:
        if r['subject'] in fold_for and fold_for[r['subject']]!=int(r['fold']):raise ValueError('Subject spans outer test folds')
        fold_for[r['subject']]=int(r['fold'])
    models={fold:checked_model(source,fold)[0] for fold in sorted(set(fold_for.values()))}
    for subject,fold in fold_for.items():
        test={r['subject'] for r in splits if r['level']=='outer' and int(r['outer_fold'])==fold and r['role']=='test'}
        train={r['subject'] for r in splits if r['level']=='outer' and int(r['outer_fold'])==fold and r['role']=='train'}
        if subject not in test or subject in train or train&test:raise ValueError('Invalid held-out membership')
    for item in rows:
        frame=lookup[item['id']]
        if frame['subject']!=item['subject'] or frame['y']!=item['species']:raise ValueError('Frame metadata differs from manifest')
    subjects=sorted(fold_for);tests=[];null_rows=[];table_rows=[];cycle_rows=[]
    for cohort,chosen in mapping.items():
      for subject in subjects:
        selected_s=[x for x in chosen if x['subject']==subject];fold=fold_for[subject];model=models[fold];B=model.c['B'];mus={}
        for item in selected_s:
            f=lookup[item['id']];cy=get_cycles(f,B,model.options['max_cycles'])
            if len(cy)<2:raise ValueError('Legacy accepted frame is now rejected; do not silently change cohort.')
            mu=model.cyc_sem.transform(cy);mus.setdefault(item['recording'],[]).append(mu)
            cycle_rows.append({'cohort':cohort,'subject':subject,'recording':item['recording'],'id':item['id'],'fold':fold,'bins':B,'cycles':len(cy)})
        for group in cfg['groups']:
            base={'cohort':cohort,'subject':subject,'fold':fold,'group':group,'bins':B,'segments':len(selected_s),'recordings':len(mus),'conditional_null_assumption':'recording-wise joint group invariance; not established by disjoint sample support','test_scope':'phase-origin sensitivity' if group=='recording_cyclic_shared' else 'ordering of size-2 bin blocks (fixed B=4/8)'}
            if not selected_s:
                tests.append({**base,'status':'no_segments_on_fixed_grid','p_value':None});continue
            permutations=patterns(B,group,cfg['block_size']);tables=[]
            for record,mm in sorted(mus.items()):
                def build(mm=mm,permutations=permutations):return recording_table(mm,permutations)
                t=store.task(f'patterns/{cohort}/{subject}/{group}/{record}',build)/len(selected_s);tables.append(t)
                table_rows.extend({'cohort':cohort,'subject':subject,'recording':record,'group':group,'pattern':j,'permutation':json.dumps(list(map(int,index))),'weighted_contribution':float(v),'segments':len(mm)} for j,(index,v) in enumerate(zip(permutations,t)))
            seed=int(digest({'seed':cfg['seed'],'subject':subject,'cohort':cohort,'group':group})[:16],16)
            result=store.task(f'test/{cohort}/{subject}/{group}',lambda tables=tables,seed=seed:null_from_tables(tables,cfg['permutations'],seed,cfg['exact_limit']))
            null_rows.extend({'cohort':cohort,'subject':subject,'group':group,'draw':j,'statistic':v,'method':result['method']} for j,v in enumerate(result['null_values']))
            tests.append({**base,'status':'tested',**{k:v for k,v in result.items() if k!='null_values'}})
        export(tests,null_rows,table_rows,cycle_rows,folder,len(subjects))
    return {'subjects_planned':len(subjects),'tested_rows':sum(r['status']=='tested' for r in tests),'no_classifier_refit':True,'raw_independence_claim':False}


def export(tests,nulls,tables,cycles,folder,family_n):
    # Fixed family = all legacy subjects, per cohort/group, missing subjects p=1.
    for cohort,group in sorted({(r['cohort'],r['group']) for r in tests}):
        current=[r for r in tests if r['cohort']==cohort and r['group']==group]
        pp=[r['p_value'] if r['p_value'] is not None else 1. for r in current]+[1.]*(family_n-len(current));qq=bh(pp)
        for r,q in zip(current,qq):r.update(q_bh_exploratory=float(q),p_bonferroni=float(min(1,(r['p_value'] if r['p_value'] is not None else 1)*family_n)),family_size_planned=family_n)
    for name,data in [('subject_order_tests',tests),('subject_null_statistics',nulls),('recording_pattern_statistics',tables),('cycle_manifest',cycles)]:csv_write(folder/(name+'.csv'),data)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['plan','test'],required=True);p.add_argument('--manifest');p.add_argument('--source-results');p.add_argument('--output-root',default='runs_radar_order');p.add_argument('--seed',type=int,default=20261101);p.add_argument('--permutations',type=int,default=999);p.add_argument('--exact-limit',type=int,default=4096);p.add_argument('--gap-samples',type=int,default=0);p.add_argument('--include-overlap-reference',action='store_true');p.add_argument('--stop-after',type=int,default=0);a=p.parse_args(argv)
    if a.gap_samples<0 or a.gap_samples%1000 or a.permutations<1 or a.exact_limit<1:p.error('Invalid gap or number of randomizations')
    source=Path(a.source_results).resolve() if a.source_results else None
    manifest=Path(a.manifest).resolve() if a.manifest else source/'segment_manifest.csv' if source else None
    if manifest is None:p.error('Supply --manifest or --source-results')
    if a.stage=='test' and (source is None or a.manifest):p.error('test requires --source-results only; it uses the original segment_manifest.csv')
    rows=read_manifest(manifest);cfg={'seed':a.seed,'permutations':a.permutations,'exact_limit':a.exact_limit,'gap_samples':a.gap_samples,'include_overlap_reference':a.include_overlap_reference,'groups':GROUPS,'block_size':2}
    inputs=[{'name':'manifest','sha256':sha(manifest)}]
    if a.stage=='test':
        from .followup import legacy_identity
        legacy_identity(source)
        # Pin every consumed legacy model/preparation payload and all manifests.
        for name in ['run_identity.json','splits.csv','oof_predictions.csv','selected_configs.csv']:
            inputs.append({'name':name,'sha256':sha(source/name)})
        for payload in sorted((source/'checkpoints').glob('*.joblib')):
            if payload.name.startswith('prepare__record_') or (payload.name.startswith('outer__') and payload.name.endswith('__phase_ordered.joblib')):
                inputs.append({'name':payload.name,'sha256':sha(payload)})
    modules=['radar_order.py','order.py','models.py','phase.py','storage.py','followup.py','raw_baselines.py']
    identity={'version':VERSION,'stage':a.stage,'config':cfg,'inputs':inputs,'sources':{n:sha(ROOT/'revision'/n) for n in modules},'environment':environment()}
    folder=ROOT/a.output_root/('radar_order_'+a.stage+'-'+digest(identity)[:16]);print('RADAR ORDER DIRECTORY',folder,flush=True)
    from . import storage
    storage.CSV_CONTEXT={'data_mode':'saved_radar_cohort_plan' if a.stage=='plan' else json.loads((source/'run_identity.json').read_text())['config']['data_mode'],'run_id':folder.name}
    with run_lock(folder),threadpool_limits(limits=1):
        (folder/'complete.json').unlink(missing_ok=True);json_write(folder/'analysis_identity.json',identity);store=Store(folder/'checkpoints',a.stop_after)
        json_write(folder/'SCOPE.json',{'primary_cohort':'grid0','primary_group':'recording_block_shared','confirmatory_claim':False,'sensitivity_cohort':'grid1000','sampling_rate_hz':1000,'segment_length':2000,'lattice_period_samples':2000+a.gap_samples,'claim':'raw sample support is disjoint inside each grid cohort; no claim of independent filtered trajectories','front_end_limitation':'legacy forward-backward filtering uses full recording; normalized cycles have resampling dependence','nulls':'shared rotation tests phase origin; shared block permutation tests block ordering under recording-wise invariance; neither proves biological phase accuracy','missing_subjects':'report counts; no fallback; fixed multiplicity family includes all legacy subjects','interpretation':'conditional exploratory controls after prior dataset development; not a new untouched biological experiment'})
        try:
            mapping,counts=plan(rows,folder,a.gap_samples,a.include_overlap_reference)
            result={'stage':a.stage,'original_segments':len(rows),'subjects':len({x['subject'] for x in rows}),'counts':{k:len(v) for k,v in mapping.items()}}
            if a.stage=='test':result.update(analyze(source,rows,mapping,cfg,store,folder))
            json_write(folder/'complete.json',{**result,'new_tasks':store.completed,'reused_tasks':store.reused});print('COMPLETE',json.dumps(result),'new=',store.completed,'reused=',store.reused)
        except InterruptedError as e:json_write(folder/'interrupted.json',{'reason':str(e)});raise SystemExit(75)
        except Exception as e:json_write(folder/'last_error.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()});raise
if __name__=='__main__':main()
