import csv,json,sys,tempfile
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]/'code'
sys.path.insert(0,str(ROOT))
from revision.closure_supervisor import validate_unique,verify_checkpoint
from revision.storage import sha

def test_unique():
    frames=[{'id':'x','subject':'p1','y':'cat'},{'id':'y','subject':'p2','y':'dog'}]
    rows=[{'id':'x','subject':'p1','y':'cat','pred':'dog'},{'id':'y','subject':'p2','y':'dog','pred':'dog'}]
    validate_unique(rows,frames,'v','ha_soft')
    try:validate_unique(rows+[rows[0]],frames,'v','ha_soft')
    except ValueError:pass
    else:raise AssertionError('Duplicate should fail')

def test_receipt():
    with tempfile.TemporaryDirectory() as d:
        p=Path(d);(p/'checkpoints').mkdir()
        model=p/'checkpoints'/'a.joblib';model.write_bytes(b'model')
        (p/'checkpoints'/'a.done.json').write_text(json.dumps({'sha256':sha(model)}))
        assert len(verify_checkpoint(p))==1
        model.write_bytes(b'altered')
        try:verify_checkpoint(p)
        except ValueError:pass
        else:raise AssertionError('Tampered receipt should fail')
if __name__=='__main__':
    test_unique();test_receipt();print('ALL UNIT TESTS PASS')
