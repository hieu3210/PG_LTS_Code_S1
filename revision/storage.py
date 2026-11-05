"""Atomic, checksummed task commits and OS-released process lock."""
from pathlib import Path
from contextlib import contextmanager
import hashlib, json, os, platform, tempfile
import joblib
CSV_CONTEXT = {}

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',',':'), default=str).encode()).hexdigest()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def atomic(path, writer):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',suffix='.partial',dir=path.parent);os.close(fd)
    try:
        writer(Path(tmp))
        with open(tmp,'rb') as f: os.fsync(f.fileno())
        os.replace(tmp,path)
        if os.name=='posix':
            fd=os.open(path.parent,os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)

def json_write(path, obj):
    atomic(path,lambda p:p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=str),encoding='utf-8'))

def csv_write(path, rows):
    import csv
    rows=[{**CSV_CONTEXT,**r} for r in rows]
    fields=list(dict.fromkeys(k for r in rows for k in r))
    def write(p):
        with p.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    atomic(path,write)

@contextmanager
def run_lock(folder):
    p=Path(folder)/'run.lock';p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a+b') as f:
        f.seek(0);f.write(b'0');f.flush();f.seek(0)
        try:
            if os.name=='nt':
                import msvcrt;msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl;fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as e:raise RuntimeError('Another process owns this run; use a different run or wait.') from e
        try:yield
        finally:
            f.seek(0)
            if os.name=='nt':
                import msvcrt;msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
            else:
                import fcntl;fcntl.flock(f.fileno(),fcntl.LOCK_UN)

class Store:
    def __init__(self,folder,stop_after=0):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.stop_after=stop_after;self.completed=0;self.reused=0
    def task(self,key,build):
        name=key.replace('/','__');p=self.folder/(name+'.joblib');r=self.folder/(name+'.done.json')
        if p.exists() and r.exists():
            try:
                receipt=json.loads(r.read_text());assert receipt['sha256']==sha(p)
                result=joblib.load(p);self.reused+=1;return result
            except (Exception,):
                # A broken commit is never treated as a successful task.
                r.unlink(missing_ok=True)
        print('RUN',key,flush=True)
        result=build()
        atomic(p,lambda q:joblib.dump(result,q,compress=3))
        json_write(r,{'key':key,'sha256':sha(p),'host':platform.node()})
        self.completed+=1
        if self.stop_after and self.completed>=self.stop_after:
            raise InterruptedError('Requested checkpoint stop; rerun the identical command to resume.')
        return result
