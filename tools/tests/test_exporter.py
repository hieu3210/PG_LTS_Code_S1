"""Storage and cluster-bootstrap tests only; fixtures are synthetic, never scientific results."""
import importlib.util,tempfile,unittest
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('audit',Path(__file__).resolve().parents[1]/'export_audit.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Checks(unittest.TestCase):
 def test_resume_tamper_and_powerloss_boundary(self):
  with tempfile.TemporaryDirectory() as d:
   out=Path(d);s=m.Store(out,1);p=out/'x';calls=[]
   def build():calls.append(1);m.atomic(p,lambda q:q.write_text('valid'));return [p]
   with self.assertRaises(InterruptedError):s.task('one',build)
   s=m.Store(out);s.task('one',build);self.assertEqual(len(calls),1)
   p.write_text('corrupt');s.task('one',build);self.assertEqual(len(calls),2)
   # A partial artifact without a committed receipt is rebuilt.
   p.write_text('half');s.task('two',build);self.assertEqual(p.read_text(),'valid')
 def test_bootstrap_subject_multiplicity_and_resume(self):
  with tempfile.TemporaryDirectory() as d:
   out=Path(d);rows=[dict(id=str(i),arm='fixture',top_k='3',subject=s,y=y,full_pred=y,sparse_pred=p,agreement=str(int(y==p))) for i,(s,y,p) in enumerate([('a','A','A'),('a','A','B'),('b','A','A'),('c','B','B'),('d','B','A')])]
   m.export_bootstrap(rows,m.Store(out),10,20,7)
   values=sorted((out/'bootstrap').glob('*/values*'));hashes=[m.sha(p) for p in values]
   m.export_bootstrap(rows,m.Store(out),10,20,7);self.assertEqual(hashes,[m.sha(p) for p in values])
   for dp in (out/'bootstrap').glob('*/draws*'):
    for rep in {r['replicate'] for r in m.readcsv(dp)}:
     rr=[r for r in m.readcsv(dp) if r['replicate']==rep];self.assertEqual(sum(int(r['multiplicity']) for r in rr),4)
 def test_macro_f1_fixed_observed_classes(self):self.assertAlmostEqual(m.metrics(['A','A','B'],['A','B','B']),2/3)
if __name__=='__main__':unittest.main()
