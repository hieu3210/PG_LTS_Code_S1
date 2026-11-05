"""Make aeon 1.5.0 tolerate low-variance collections (advisory instead of fatal).

aeon 1.5.0 raises "Input collection has too little variation" inside
BaseCollectionEstimator._preprocess_collection for some of our fold tensors.
The reported experiments ran with the check downgraded to advisory
(raise_error=False). The edit must live in the installed library because joblib
worker processes do not inherit a runtime monkeypatch.

Usage:  python scripts/patch_aeon.py          (idempotent; prints what it did)
        python scripts/patch_aeon.py --check  (exit 1 if not patched)
"""
import re, sys
from pathlib import Path
import aeon
assert aeon.__version__ == "1.5.0", f"expected aeon 1.5.0, found {aeon.__version__}"
path = Path(aeon.__file__).parent / "base" / "_base_collection.py"
text = path.read_text(encoding="utf-8")
if "check_collection_variance(X, raise_error=False)" in text:
    print("already patched:", path); sys.exit(0)
if "--check" in sys.argv:
    print("NOT patched:", path); sys.exit(1)
new, n = re.subn(r"check_collection_variance\(X\)", "check_collection_variance(X, raise_error=False)", text)
if n != 1:
    sys.exit(f"expected exactly one call to patch, found {n}; inspect {path} manually")
path.write_text(new, encoding="utf-8")
print("patched:", path)
