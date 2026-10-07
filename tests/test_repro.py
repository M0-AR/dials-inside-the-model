"""Reproducibility smoke tests (numpy only). Run: python3 -m pytest tests/ -q"""
import os, numpy as np
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(BASE, "results")

def test_vocab():
    import sys
    sys.path.insert(0, os.path.join(BASE, "parts"))
    from common import WORDS
    assert len(WORDS) == 188 and len(set(WORDS)) == 188

def test_model_params():
    import sys
    sys.path.insert(0, os.path.join(BASE, "parts"))
    from common import TinyLM
    assert TinyLM().params() == 9284

def test_artifacts_exist():
    for f in ["model.npz", "hidden.npy", "sae64.npz", "features64.npy", "part6.npz", "part7.npz"]:
        assert os.path.exists(os.path.join(R, f)), f"missing {f}"

def test_swap_kept_high():
    d = np.load(os.path.join(R, "part5.npz"))
    assert float(d["kept"]) > 90, f"swap kept {d['kept']} too low"

def test_rebuild_high():
    d = np.load(os.path.join(R, "part5.npz"))
    assert float(d["varexp"]) > 0.85
