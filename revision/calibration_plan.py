"""Create ragged aggregation profiles from saved clean-cycle counts; no raw data needed."""
import argparse,json
from pathlib import Path
import numpy as np
from . import ROOT
from .followup import read_csv,legacy_identity
from .storage import json_write,sha

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-results',required=True);p.add_argument('--output',required=True);p.add_argument('--select',choices=['all','representative'],default='representative');a=p.parse_args()
    source=Path(a.source_results);legacy_identity(source)
    rows=[r for r in read_csv(source/'phase_failure_rates.csv') if r['arm']=='phase_ordered' and r['condition']=='clean']
    if not rows:raise ValueError('No clean phase_ordered cycle counts; run old robustness stage with ORIGINAL v1 code and checkpoints first.')
    profiles=[];configs={int(r['outer_fold']):json.loads(r['config']) for r in read_csv(source/'selected_configs.csv') if r['arm']=='phase_ordered'}
    ids=[r['id'] for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate clean phase rows')
    for subject in sorted({r['subject'] for r in rows}):
        s=sorted([r for r in rows if r['subject']==subject],key=lambda r:r['id']);counts=[int(r['used_cycles']) for r in s]
        if min(counts)<2:raise ValueError('Rejected clean segments in profile')
        folds={int(r['fold']) for r in s}
        if len(folds)!=1:raise ValueError('One subject spans folds')
        B=configs[next(iter(folds))]['B']
        profiles.append({'name':subject,'segments':len(s),'cycles':max(counts),'cycle_counts':counts,'bins':[B],'overlap':False})
    profiles.sort(key=lambda r:sum(r['cycle_counts']))
    if a.select=='representative':profiles=[profiles[i] for i in sorted({0,len(profiles)//2,len(profiles)-1})]
    cfg=json.loads((ROOT/'configs/followup_calibration.json').read_text());cfg['profiles']=profiles
    cfg['profile_provenance']={'source_run_identity_sha256':sha(source/'run_identity.json'),'cycle_counts_sha256':sha(source/'phase_failure_rates.csv'),'selected_configs_sha256':sha(source/'selected_configs.csv'),'selection':a.select,'scope':'Exact clean per-segment cycle counts and selected B; synthetic memberships; independent segments. Does not recreate preprocessing or overlap dependence.'}
    json_write(a.output,cfg);print('WROTE',a.output,'profiles=',len(profiles))
if __name__=='__main__':main()
