"""Pair benchmark runs by recording/cohort; never compare unaligned cohorts silently."""
import argparse,json
from pathlib import Path
from .followup import read_csv
from .storage import csv_write

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',nargs='+',required=True);p.add_argument('--output',required=True);a=p.parse_args();rows=[];reference=None
    for name in a.runs:
        root=Path(name);i=json.loads((root/'benchmark_identity.json').read_text());complete=json.loads((root/'complete.json').read_text());r=read_csv(root/'benchmark_repetitions.csv')
        signature=(i['record_sha256'],i['threads'],i['device_label'],i['cohort'],i['warmup_per_worker'],i['environment'],sorted({x['accepted_id_sha256'] for x in r}))
        if reference is None:reference=signature
        if signature!=reference:raise ValueError('Benchmark hardware/settings/record/cohort differ; compare separately and disclose differences.')
        for x in read_csv(root/'benchmark_summary.csv'):rows.append({'benchmark_run':root.name,'device_label':i['device_label'],'model_sha256':i['model_sha256'],'model_file_bytes':complete['model_file_bytes'],**x})
    csv_write(a.output,rows);print('WROTE',a.output)
if __name__=='__main__':main()
