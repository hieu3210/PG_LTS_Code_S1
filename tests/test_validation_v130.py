import itertools,json,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
from revision.recording_calibration import orbit_tables,null_result,profiles_from_manifest,one_rep,main as calibrate
from revision.radar_order import patterns,statistic,null_from_tables
from revision.repeated_nested import macro,summarize,frozen_spaces
from revision.storage import csv_write,digest

@pytest.mark.parametrize('B,A,w',[(4,2,0.),(4,4,.3),(8,2,.8),(8,4,1.),(8,4,.8)])
def test_exact_moment_reduction_matches_explicit_cycles(B,A,w):
 rng=np.random.default_rng(8);path=rng.dirichlet(np.ones(A),size=(3,B,3));q=[rng.dirichlet(np.ones(A),size=(n,3)) for n in [2,7,11]]
 for group in ['recording_block_shared','recording_cyclic_shared']:
  ps=patterns(B,group);a=orbit_tables(path,np.stack([x.mean(0) for x in q]),ps,w)
  b=np.array([[statistic(w*path[s,ix][None]+(1-w)*q[s][:,None]) for s in range(3)] for ix in ps])
  np.testing.assert_allclose(a,b,atol=1e-14,rtol=1e-11)

@pytest.mark.parametrize('J,limit',[(1,4096),(2,4096),(4,4096),(2,1)])
def test_randomization_matches_executed_radar_algorithm(J,limit):
 rng=np.random.default_rng(3);tables=[rng.uniform(size=24) for _ in range(J)]
 for seed in [1,71]:
  a=null_result(tables,999,np.random.default_rng(seed),limit);b=null_from_tables(tables,999,seed,limit)
  for k in ['statistic','p_value','exceedances','null_draws','method','group_cardinality']:assert a[k]==b[k]

def test_small_orbit_exact_pvalues_conservative():
 for group in ['recording_block_shared','recording_cyclic_shared']:
  L=len(patterns(8,group));v=np.arange(L,dtype=float);p=np.array([(v>=x).mean() for x in v]);assert np.mean(p<=.05)<=.05
 assert null_result([np.ones(8)],999,np.random.default_rng(1),4096)['p_value']==1

def test_empirical_profile_counts_and_grouping():
 p=profiles_from_manifest(Path('inputs/order_v120/cycle_manifest.csv'));assert len(p)==66
 for cohort,N,J in [('grid0',1404,165),('grid1000',1351,164)]:
  x=[x for x in p if x['cohort']==cohort];assert len(x)==33;assert sum(r['segments'] for r in x)==N;assert sum(len(r['records']) for r in x)==J

def test_repetition_seed_independent_of_chunking():
 p=profiles_from_manifest(Path('inputs/order_v120/cycle_manifest.csv'))[0];cfg=json.loads(Path('configs/recording_calibration_full.json').read_text());cfg['null_permutations']=19
 assert one_rep(p,cfg,1)==one_rep(p,cfg,1)
 assert one_rep(p,cfg,0)!=one_rep(p,cfg,1)

def test_frozen_grids_not_regenerated_with_split_seed():
 arms=['occupancy','combined','phase_occ','phase_ordered'];a=frozen_spaces(Path('inputs/main'),arms)
 for arm in arms:
  assert len(a[arm])==12
  assert all(c['candidate_id']==digest({k:v for k,v in c.items() if k!='candidate_id'})[:12] for c in a[arm])

def test_averaged_repeat_score_and_cluster_bootstrap(tmp_path):
 # Second repeat swaps a perfect and wrong arm: equal mean F1, zero paired mean.
 pred=[]
 for seed in [1,2]:
  for a in ['a','b']:
   for s,y in [('s0','x'),('s1','x'),('s2','z'),('s3','z')]:
    good=(seed==1)==(a=='a');p=y if good else ('x' if y=='z' else 'z')
    pred.append({'repeat_seed':seed,'arm':a,'id':s+':0','subject':s,'y':y,'pred':p})
 cfg={'arms':['a','b'],'split_seeds':[1,2],'bootstrap':100,'bootstrap_seed':1,'contrasts':[['a','b']]}
 summarize(pred,cfg,tmp_path,True)
 import csv
 rows=list(csv.DictReader((tmp_path/'repeated_paired_summary.csv').open()))
 assert len(rows)==2
 for r in rows:
  assert float(r['mean_delta_macro_f1'])==0;assert float(r['lo'])==0;assert float(r['hi'])==0;assert int(r['biological_subjects'])==4


def test_calibration_interruption_reuse_and_corruption(tmp_path):
 manifest=tmp_path/'cycles.csv';csv_write(manifest,[{'cohort':'grid0','subject':'a','recording':'r','id':'r:0:2000','bins':8,'cycles':3}])
 cfg={'seed':4,'cycle_manifest':str(manifest),'output_root':str(tmp_path/'runs'),'expected_profiles':1,'channels':3,'states':4,'path_weight':.8,'scenarios':['uniform_shared'],'repetitions':4,'null_permutations':19,'exact_limit':4096,'alpha':.05,'chunk_size':2}
 conf=tmp_path/'c.json';conf.write_text(json.dumps(cfg))
 with pytest.raises(SystemExit) as e:calibrate(['--config',str(conf),'--workers','1','--stop-after','1'])
 assert e.value.code==75
 d=next((tmp_path/'runs').iterdir());assert not (d/'complete.json').exists()
 calibrate(['--config',str(conf),'--workers','1']);a=(d/'recording_calibration_raw.csv').read_bytes();r=json.loads((d/'complete.json').read_text());assert r['reused_chunks']==1 and r['new_chunks']==1
 cp=next((d/'checkpoints').glob('*.joblib'));cp.write_bytes(b'corrupt')
 calibrate(['--config',str(conf),'--workers','1']);r=json.loads((d/'complete.json').read_text());assert r['new_chunks']==1 and r['reused_chunks']==1;assert (d/'recording_calibration_raw.csv').read_bytes()==a
