import csv,json,subprocess,sys
import numpy as np
import pytest
from revision.radar_order import parse_id,read_manifest,selected,plan,patterns,recording_table,null_from_tables,statistic,export
from revision.storage import Store

def write(path,rows):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def mu(seed=1):
 x=np.random.default_rng(seed).uniform(size=(5,8,3,4));return x/x.sum(-1,keepdims=True)
def test_parse_intervals_half_open():
 assert parse_id('record:2000:2000')==('record',2000,2000)
 for x in ['record:-1000:2000','record:1:2000','record:0:1999']:
  with pytest.raises(ValueError):parse_id(x)
def test_fixed_grid_no_fallback(tmp_path):
 f=tmp_path/'m.csv';write(f,[{'id':f'r:{s}:2000','subject':'a','y':'x'} for s in [1000,2000,3000,4000]]);r=read_manifest(f)
 assert [x['start'] for x in r if selected(x,'grid0')]==[2000,4000]
 assert [x['start'] for x in r if selected(x,'grid1000')]==[1000,3000]
def test_disjoint_and_optional_gap(tmp_path):
 f=tmp_path/'m.csv';write(f,[{'id':f'r:{s}:2000','subject':'a','y':'x'} for s in range(0,9000,1000)]);r=read_manifest(f)
 for gap in [0,1000]:
  mapping,_=plan(r,tmp_path,gap)
  for items in mapping.values():assert all(b['start']-a['stop']>=gap for a,b in zip(items,items[1:]))
def test_dedup_arms_conflict_refused(tmp_path):
 f=tmp_path/'m.csv';a={'id':'r:0:2000','subject':'a','y':'x','arm':'one'};write(f,[a,dict(a,arm='two')]);assert len(read_manifest(f))==1
 write(f,[a,dict(a,subject='b')])
 with pytest.raises(ValueError):read_manifest(f)
def test_missing_subject_no_fallback(tmp_path):
 f=tmp_path/'m.csv';write(f,[{'id':'r:1000:2000','subject':'a','y':'x'}]);mapping,counts=plan(read_manifest(f),tmp_path)
 assert mapping['grid0']==[] and counts[0]['status']=='no_segments_on_fixed_grid'
def test_statistic_matches_legacy():
 from revision.order import statistic as legacy
 for seed in [1,2,3]:assert abs(statistic(mu(seed))-legacy(mu(seed)))<1e-14
def test_patterns_preserve_occupancy_and_local_adjacency():
 x=mu();ps=patterns(8,'recording_block_shared');assert len(ps)==24 and ps[0]==tuple(range(8))
 for p in ps:
  assert np.allclose(x[:,p].mean(1),x.mean(1));assert all(p[j+1]==p[j]+1 for j in range(0,8,2))
 ps=patterns(8,'recording_cyclic_shared');assert len(ps)==8
 for p in ps:assert all((p[(j+1)%8]-p[j])%8==1 for j in range(8))
def test_shared_transform_all_segments():
 a,b=mu(1),mu(2);ps=patterns(8,'recording_block_shared');t=recording_table([a,b],ps)
 for j,p in enumerate(ps):assert abs(t[j]-statistic(a[:,p])-statistic(b[:,p]))<1e-14
def test_record_weight_matches_segment_mean():
 a,b,c=mu(1),mu(2),mu(3);ps=patterns(8,'recording_cyclic_shared');r=null_from_tables([recording_table([a,b],ps)/3,recording_table([c],ps)/3],99,1)
 assert abs(r['statistic']-(statistic(a)+statistic(b)+statistic(c))/3)<1e-14
 assert r['method']=='exact' and r['group_cardinality']=='64'
def test_exact_ties_resolution():
 r=null_from_tables([np.ones(8)],999,1);assert r['p_value']==1 and r['null_draws']==8 and r['p_resolution_lower_bound']==1/8
 r=null_from_tables([np.arange(8)[::-1]],999,1);assert r['p_value']==1/8
def test_mc_plus_one_seed():
 tables=[np.arange(24),np.arange(24)];r=null_from_tables(tables,99,11,10);assert r==null_from_tables(tables,99,11,10)
 k=sum(v>=r['statistic']-1e-12 for v in r['null_values']);assert r['p_value']==(1+k)/100 and r['method']=='monte_carlo'
def test_exact_null_rank_size():
 v=np.array([1,1,3,4,5,5,7,8.]);p=np.array([(v>=x).mean() for x in v])
 for alpha in [.05,.125,.25,.5]:assert np.mean(p<=alpha)<=alpha
def test_missing_subjects_count_in_multiplicity(tmp_path):
 r=[{'cohort':'grid0','group':'x','subject':'a','p_value':.01},{'cohort':'grid0','group':'x','subject':'b','p_value':None}];export(r,[],[],[],tmp_path,3)
 assert r[0]['p_bonferroni']==.03 and r[1]['p_bonferroni']==1 and r[0]['family_size_planned']==3
def test_cli_plan(tmp_path):
 f=tmp_path/'m.csv';write(f,[{'id':f'r:{s}:2000','subject':'a','y':'x'} for s in [0,1000,2000]])
 r=subprocess.run([sys.executable,'-m','revision.radar_order','--stage','plan','--manifest',str(f),'--output-root',str(tmp_path/'out')],capture_output=True,text=True);assert r.returncode==0,r.stderr
 p=next((tmp_path/'out').iterdir());r=json.loads((p/'complete.json').read_text());assert r['counts']=={'grid0':2,'grid1000':1}
def test_restart_corrupt_table(tmp_path):
 s=Store(tmp_path,stop_after=1)
 with pytest.raises(InterruptedError):s.task('patterns/a',lambda:np.array([1.,2.]))
 s=Store(tmp_path);assert np.array_equal(s.task('patterns/a',lambda:None),[1,2]);assert s.reused==1
 (tmp_path/'patterns__a.joblib').write_bytes(b'bad');assert np.array_equal(s.task('patterns/a',lambda:np.array([3.,4.])),[3,4]);assert s.completed==1

def test_repeated_id_in_same_arm_is_not_hidden(tmp_path):
 f=tmp_path/'m.csv';a={'id':'r:0:2000','subject':'a','y':'x','arm':'one'};write(f,[a,a])
 with pytest.raises(ValueError):read_manifest(f)
