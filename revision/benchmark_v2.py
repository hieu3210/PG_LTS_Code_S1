"""Resumable offline benchmark, one isolated worker per measured repetition.
This is a target-device measurement tool, not a claim of embedded performance.
"""
from pathlib import Path
import argparse,json,os,platform,subprocess,sys,time,tempfile
import numpy as np
import joblib
from threadpoolctl import threadpool_limits
from pglts.signal.preprocess import highpass,normalise
from pglts.signal.features import FeatureExtractor,PGLTS_CORE
from pglts.signal.segmentation import harmonic_ratio,frame_bounds
from .models import _cached_cycles,get_cycles
from .storage import Store,run_lock,sha,digest,json_write,csv_write
from .cli import environment


def measured_pipeline(model,amplitude,phase,cohort):
    _cached_cycles.cache_clear();timing={};start_total=time.perf_counter()
    t=time.perf_counter();amp=normalise(highpass(amplitude),'std');z=normalise(highpass(amplitude*np.exp(1j*phase)),'std');timing['front_end_s']=time.perf_counter()-t
    frames=[];extractor=FeatureExtractor(names=PGLTS_CORE);times=dict(hr_gating_s=0.,window_features_s=0.,phase_and_resampling_s=0.)
    native_cycle=model.c['view'] in ('combined','phase_ordered','phase_occ') or model.c['method'] in ('linear','rocket','multirocket')
    total=0;hr_ok=0;cycle_ok=0
    for index in frame_bounds(len(amp),2000,1000):
        total+=1;a=amp[index:index+2000];c=z[index:index+2000];t=time.perf_counter();keep=harmonic_ratio(a)>model.options['hr_threshold'];times['hr_gating_s']+=time.perf_counter()-t
        if not keep:continue
        hr_ok+=1;f={'id':f'benchmark:{index}','amp':a.astype(np.float32),'cpx':c.astype(np.complex64)}
        if native_cycle or cohort=='matched_cycle':
            t=time.perf_counter();B=model.c.get('B',8) if native_cycle else 8;cy=get_cycles(f,B,model.options['max_cycles']);times['phase_and_resampling_s']+=time.perf_counter()-t
            if len(cy)<2:continue
        cycle_ok+=1;t=time.perf_counter();f['window']=extractor.transform(a).astype(np.float32);times['window_features_s']+=time.perf_counter()-t;frames.append(f)
    if not frames:raise ValueError('No accepted segments in benchmark record')
    timing.update(times)
    # _describe includes linguistic mapping/cycle pooling; phase cache is populated
    # within THIS repetition, not inherited from warmup or a previous repetition.
    t=time.perf_counter();x=model._describe(frames,False)[0];timing['descriptor_s']=time.perf_counter()-t
    t=time.perf_counter();x=model.scaler.transform(x);timing['scaling_s']=time.perf_counter()-t
    t=time.perf_counter();scores=model.clf.decision_function(x)
    if scores.ndim==1:scores=np.c_[-scores/2,scores/2]
    pred=model.classes_[scores.argmax(1)];timing['classifier_s']=time.perf_counter()-t
    timing['total_pipeline_s']=time.perf_counter()-start_total
    return {**timing,'candidate_segments':total,'hr_accepted':hr_ok,'accepted_segments':len(frames),'cohort_policy':cohort,'cycle_eligible':cycle_ok,'accepted_id_sha256':digest([f['id'] for f in frames]),'prediction_sha256':digest(pred.tolist())}


def worker(model_path,record,cohort,threads,warmup):
    with threadpool_limits(limits=threads):
        t=time.perf_counter();model=joblib.load(model_path);model_load=time.perf_counter()-t
        t=time.perf_counter();data=np.loadtxt(record,delimiter=',',skiprows=1);io=time.perf_counter()-t
        for _ in range(warmup):measured_pipeline(model,data[:,0],data[:,1],cohort)
        row=measured_pipeline(model,data[:,0],data[:,1],cohort)
        row.update(model_load_s=model_load,input_io_s=io,recording_samples=len(data),acquisition_s=len(data)/1000)
        try:
            import resource
            value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;row['process_peak_rss_bytes']=int(value*(1 if sys.platform=='darwin' else 1024))
        except ImportError:row['process_peak_rss_bytes']=None
        return row


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True);p.add_argument('--record',required=True);p.add_argument('--output-root',default='benchmarks_v110');p.add_argument('--repeats',type=int,default=30);p.add_argument('--threads',type=int,default=1);p.add_argument('--warmup',type=int,default=1);p.add_argument('--cohort',choices=['native','matched_cycle'],default='native');p.add_argument('--device-label',default='UNSPECIFIED');p.add_argument('--stop-after',type=int,default=0);p.add_argument('--worker-output',help=argparse.SUPPRESS);a=p.parse_args(argv)
    if a.repeats<1 or a.threads<1 or a.warmup<0:p.error('Invalid repeats/threads/warmup')
    if a.worker_output:
        json_write(a.worker_output,worker(a.model,a.record,a.cohort,a.threads,a.warmup));return
    # Hardware label is descriptive, never inferred to be an embedded device.
    identity={'model_sha256':sha(a.model),'record_sha256':sha(a.record),'repeats':a.repeats,'threads':a.threads,'warmup_per_worker':a.warmup,'cohort':a.cohort,'device_label':a.device_label,'environment':environment(),'source_sha256':sha(__file__)}
    folder=Path(a.output_root)/('benchmark-'+digest(identity)[:16]);print('BENCHMARK DIRECTORY',folder,flush=True)
    from . import storage
    storage.CSV_CONTEXT={'run_id':folder.name,'data_mode':'device_benchmark_unclassified'}
    with run_lock(folder):
        json_write(folder/'benchmark_identity.json',identity);store=Store(folder/'checkpoints',a.stop_after);rows=[]
        (folder/'complete.json').unlink(missing_ok=True)
        try:
            for rep in range(a.repeats):
                def build(rep=rep):
                    temp=folder/f'worker_{rep:04d}.json';t=time.perf_counter()
                    subprocess.run([sys.executable,'-m','revision.benchmark_v2','--model',str(Path(a.model).resolve()),'--record',str(Path(a.record).resolve()),'--cohort',a.cohort,'--threads',str(a.threads),'--warmup',str(a.warmup),'--worker-output',str(temp.resolve())],check=True)
                    row=json.loads(temp.read_text());row['process_wall_s']=time.perf_counter()-t;temp.unlink();return {'rep':rep,**row}
                rows.append(store.task(f'repetition/{rep:04d}',build));csv_write(folder/'benchmark_repetitions.csv',rows)
            summary=[]
            for k in rows[0]:
                if (k.endswith('_s') or k.endswith('_bytes')) and all(r.get(k) is not None for r in rows):
                    x=[r[k] for r in rows];summary.append({'metric':k,'median':float(np.median(x)),'p05':float(np.quantile(x,.05)),'p95':float(np.quantile(x,.95)),'repetitions':len(rows)})
            csv_write(folder/'benchmark_summary.csv',summary)
            json_write(folder/'complete.json',{'repeats':len(rows),'new_tasks':store.completed,'reused_tasks':store.reused,'model_file_bytes':Path(a.model).stat().st_size,'energy_joules':None,'memory_scope':'isolated-worker process high-water RSS includes imports, model, input, warmup; not incremental classifier RAM','latency_scope':'offline full recording; pipeline excludes model loading and input file I/O, separately reported; process_wall includes process startup/warmup','phase_cache_scope':'cleared at every pipeline repetition; shared only inside one repetition','device_label':a.device_label,'energy_scope':'unmeasured; supply externally measured energy separately, with exact timing boundary'})
            print('COMPLETE benchmark new=',store.completed,'reused=',store.reused)
        except InterruptedError as e:json_write(folder/'interrupted.json',{'reason':str(e)});raise SystemExit(75)
        except Exception as e:json_write(folder/'last_error.json',{'type':type(e).__name__,'message':str(e)});raise
if __name__=='__main__':main()
