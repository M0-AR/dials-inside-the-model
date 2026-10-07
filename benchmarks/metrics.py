"""Benchmarks (SAEBench-lite, numpy only).
Metrics: variance-explained, L0, %alive, CE-recovered, absorption proxy, seed stability.
Usage: python3 benchmarks/metrics.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "parts"))
from common import TinyLM, make_windows, W2I, RESULTS

def main():
    H = np.load(os.path.join(RESULTS, "hidden.npy")).astype(float)
    F = np.load(os.path.join(RESULTS, "features64.npy")).astype(float)
    sae = np.load(os.path.join(RESULTS, "sae64.npz"))
    words = open(os.path.join(RESULTS, "corpus.txt")).read().split()
    ids = np.array([W2I[w] for w in words], dtype=np.int64)
    X, Y = make_windows(ids, 8)
    model = TinyLM.load(os.path.join(RESULTS, "model.npz"))
    D, mu, scale = sae["D"], sae["mu"], float(sae["scale"])
    bd = sae["bd"]
    Hhat = F @ D * scale + mu + bd * scale * 0  # bd already in norm units; approx
    # exact rebuild from part5 formula: (F@D+bd)*scale+mu
    Hhat = (F @ D + bd) * scale + mu
    varexp = 1 - ((Hhat - H) ** 2).mean() / ((H - H.mean(0)) ** 2).mean()
    l0 = float((F > 0).sum(1).mean())
    alive = float(((F > 0).sum(0) > 0).mean() * 100)
    # CE recovered on val tail
    nv = 5000
    Xv, Yv = X[-nv:], Y[-nv:]
    def ce(Hr):
        tot = 0.0
        for s in range(0, nv, 2048):
            Log = Hr[s:s + 2048] @ model.W2 + model.b2
            Log -= Log.max(1, keepdims=True)
            P = np.exp(Log); P /= P.sum(1, keepdims=True)
            Yb = Yv[s:s + 2048]
            tot += float(-np.log(P[np.arange(len(Yb)), Yb] + 1e-12).sum())
        return tot / nv
    base = ce(H[-nv:])
    rec = ce(Hhat[-nv:])
    mean = ce(np.repeat(H.mean(0, keepdims=True), nv, 0))
    ce_rec = (mean - rec) / (mean - base + 1e-12) * 100
    # absorption proxy: general feature (e.g. best 'thing-next' f39) should fire wherever
    # specific child fires; measure P(parent fires | child fires) gap
    p6 = np.load(os.path.join(RESULTS, "part6.npz"), allow_pickle=True)
    kind_of = [str(x) for x in p6["kind_of"]]
    thing_parents = [i for i in range(64) if kind_of[i] == "thing"][:3]
    absorb_gaps = []
    for p in thing_parents:
        pf = (F[:, p] > 0)
        for c in range(64):
            if c == p:
                continue
            cf = (F[:, c] > 0)
            if cf.sum() > 200 and (cf & pf).sum() / cf.sum() < 0.5:
                # child fires without parent -> absorption candidate
                absorb_gaps.append(float(1 - (cf & pf).sum() / cf.sum()))
    absorb = float(np.mean(absorb_gaps[:10])) if absorb_gaps else 0.0
    # seed stability: retrain tiny SAE-32 twice, match decoder cosines (like 2026 stability papers)
    from part8_scaling import train_quick
    _, F1, D1, _ = train_quick(H[:60000], 32, 3, steps=800, seed=11)
    _, F2, D2, _ = train_quick(H[:60000], 32, 3, steps=800, seed=22)
    Dn1 = D1 / (np.linalg.norm(D1, axis=1, keepdims=True) + 1e-12)
    Dn2 = D2 / (np.linalg.norm(D2, axis=1, keepdims=True) + 1e-12)
    sims = np.abs(Dn1 @ Dn2.T).max(1)
    stable = float((sims > 0.7).mean() * 100)
    print(f"variance-explained: {varexp*100:.1f}% | L0: {l0:.1f} | alive: {alive:.1f}%")
    print(f"CE-recovered: {ce_rec:.1f}% | absorption-gap: {absorb:.2f} | seed-stable(>0.7): {stable:.1f}%")
    print("interpretation: high varexp+CE with low L0 = good; absorption>0.3 warns of splitting/absorption (Chanin et al 2024); stability<80% warns of basis ambiguity (2026 findings)")
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "benchmarks.txt"), "w") as f:
        f.write(f"varexp={varexp}\nl0={l0}\nalive={alive}\nce_rec={ce_rec}\nabsorb={absorb}\nstable={stable}\n")
    print("saved results/benchmarks.txt")

if __name__ == "__main__":
    main()
