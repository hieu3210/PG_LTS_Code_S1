"""Reconstruct p-values/coverage/family counts; independently regenerate seeded cases."""
from pathlib import Path
import argparse,json,collections
import numpy as np
import pandas as pd
from revision.recording_calibration import one_rep
from revision.storage import sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--regenerate',type=int,default=64);a=p.parse_args();D=Path(a.run);identity=json.loads((D/'run_identity.json').read_text());cfg=identity['config']
 for n,h in identity['sources'].items():assert sha(Path(__file__).resolve().parents[1]/'revision'/n)==h,n
 prof=json.loads((D/'profiles.json').read_text());x=pd.read_csv(D/'recording_calibration_raw.csv',dtype={'seed':str,'null_seed':str,'group_cardinality':str})
 assert len(x)==len(prof)*len(cfg['scenarios'])*2*cfg['repetitions'];assert not x.duplicated(['cohort','subject','scenario','group','rep']).any()
 assert x.groupby(['cohort','subject','scenario','group']).size().eq(cfg['repetitions']).all()
 assert np.isfinite(x.statistic).all() and x.statistic.ge(-1e-12).all()
 exact=x.method.eq('exact');pvalue=np.where(exact,x.exceedances/x.null_draws,(x.exceedances+1)/(x.null_draws+1));np.testing.assert_allclose(x.p_value,pvalue,atol=1e-15,rtol=1e-13)
 assert ((x.p_value<=cfg['alpha']).astype(int)==x.reject).all()
 counts=x.groupby(['cohort','subject']).first()
 for q in prof:
  z=counts.loc[q['cohort'],q['subject']];assert z.segments==q['segments'];assert z.recordings==len(q['records']);assert z.cycles==sum(sum(r['cycle_counts']) for r in q['records'])
 for r in x[['bins','recordings','group','group_cardinality','method','p_resolution','null_draws']].drop_duplicates().to_dict('records'):
  import math
  K=(math.factorial(int(r['bins'])//2) if r['group']=='recording_block_shared' else int(r['bins']))**int(r['recordings']);assert str(K)==r['group_cardinality']
  assert r['method']==('exact' if K<=cfg['exact_limit'] else 'monte_carlo');assert r['null_draws']==(K if K<=cfg['exact_limit'] else cfg['null_permutations'])
 f=pd.read_csv(D/'recording_calibration_family_raw.csv');key=['cohort','scenario','group','rep']
 # Vector recomputation does not reuse the calibration summary implementation.
 y=x.assign(raw=(x.p_value<=cfg['alpha']).astype(int),bonf=(x.p_value<=cfg['alpha']/33).astype(int)).groupby(key)[['raw','bonf']].sum().sort_index()
 ff=f.set_index(key).sort_index();np.testing.assert_array_equal(y.raw.to_numpy(),ff.raw_rejections.to_numpy());np.testing.assert_array_equal(y.bonf.to_numpy(),ff.bonferroni_rejections.to_numpy());np.testing.assert_array_equal((y.bonf>0).astype(int).to_numpy(),ff.any_bonferroni_rejection.to_numpy())
 s=pd.read_csv(D/'recording_calibration_summary.csv');ss=x.groupby(['cohort','subject','scenario','group']).reject.agg(['size','sum','mean']).sort_index();t=s.set_index(['cohort','subject','scenario','group']).sort_index();np.testing.assert_array_equal(ss['sum'].to_numpy(),t.rejects.to_numpy());np.testing.assert_allclose(ss['mean'].to_numpy(),t.rejection_rate.to_numpy(),atol=1e-15)
 rng=np.random.default_rng(731);lookup={(q['cohort'],q['subject']):q for q in prof};pairs=[(q['cohort'],q['subject'],rep) for q in prof for rep in range(cfg['repetitions'])]
 for i in rng.choice(len(pairs),min(a.regenerate,len(pairs)),replace=False):
  cohort,subject,rep=pairs[i]
  for r in one_rep(lookup[cohort,subject],cfg,rep):
   row=x[(x.cohort==cohort)&(x.subject==subject)&(x.scenario==r['scenario'])&(x.group==r['group'])&(x.rep==rep)].iloc[0]
   assert int(row.exceedances)==r['exceedances'];assert row['seed']==str(r['seed']);assert abs(row.statistic-r['statistic'])<1e-13
 result={'status':'passed','rows':len(x),'cases':len(s),'profiles':len(prof),'seeded_replicates_regenerated':min(a.regenerate,len(pairs)),'all_pvalues_and_family_counts_reconstructed':True,'source_hashes_match':True}
 (D/'independent_output_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
