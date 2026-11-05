"""Export only the necessary original run inputs, including trusted prepared frames.
Run on the computer holding OLD_RUN. Does not include outer model checkpoints.
"""
import argparse,hashlib,json,zipfile
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-results',required=True);p.add_argument('--output',required=True);a=p.parse_args()
s=Path(a.source_results).resolve();out=Path(a.output).resolve()
needed=['run_identity.json','search_space.json','segment_manifest.csv','splits.csv','oof_predictions.csv','input_checksums.csv','preparation_audit.csv']
files=[s/n for n in needed];payloads=sorted((s/'checkpoints').glob('prepare__record_*.joblib'))
if not payloads:raise SystemExit('No prepared frame checkpoints: locate the original run, not a summary-only results folder.')
for f in payloads:
 r=f.with_suffix('.done.json')
 if not r.exists() or json.loads(r.read_text())['sha256']!=hashlib.sha256(f.read_bytes()).hexdigest():raise SystemExit('Invalid prepared checkpoint: '+str(f))
 files.extend([f,r])
if not all(f.is_file() for f in files):raise SystemExit('Required manifest is missing')
if out.is_relative_to(s):raise SystemExit('Write export outside the read-only original run')
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
 for f in files:z.write(f,Path('repeat_inputs')/f.relative_to(s))
print('EXPORTED',out,'prepared recordings',len(payloads),'files',len(files))
