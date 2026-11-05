#!/usr/bin/env python3
"""Verify exported bytes, independent of any fitted-model package."""
from pathlib import Path
import argparse,hashlib,json
p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args()
x=json.loads((a.folder/'SHA256SUMS.json').read_text());errors=[]
for rel,expected in x.items():
 f=a.folder/rel;h=hashlib.sha256()
 if not f.is_file():errors.append(rel+': MISSING');continue
 with f.open('rb') as stream:
  for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
 if h.hexdigest()!=expected:errors.append(rel+': HASH MISMATCH')
if errors:print('\n'.join(errors));raise SystemExit(2)
print(f'PASS: {len(x)} hashes. This verifies bytes, not model validity.')
