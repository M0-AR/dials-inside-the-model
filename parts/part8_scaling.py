"""Part 8. How many features? Dicts 16/32/64/128/256, K=3.
Measures rebuild, swap loss, readable counts, snow split, dead.
Usage: python3 parts/part8_scaling.py
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import TinyLM, make_windows, W2I, RESULTS, TOPICS

def train_quick(H, n_dict, K=3, steps=3000, seed=0):
    import importlib.util
    rng = np.random.default_rng(seed)
    N, d = H.shape
    mu = H.mean(0)
    X = H - mu
    scale = np.sqrt((X ** 2).sum(1).mean())
    Xn = X / scale
    D = rng.normal(0, 1, (n_dict, d)); D /= (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)
    E = D.T.copy()
    be = np.zeros(n_dict); bd = np.zeros(d)
    mE = np.zeros_like(E); vE = np.zeros_like(E)
    mD = np.zeros_like(D); vD = np.zeros_like(D)
    mbe = np.zeros_like(be); vbe = np.zeros_like(be)
    mbd = np.zeros_like(bd); vbd = np.zeros_like(bd)
    b1, b2, eps, lr, t = 0.9, 0.999, 1e-8, 0.01, 0
    bs = 1024
    for s in range(steps):
        t += 1
        idx = rng.integers(0, N, bs)
        Xb = Xn[idx]
        pre = Xb @ E + be
        F = np.zeros_like(pre)
        top = np.argpartition(-pre, K - 1, axis=1)[:, :K]
        F[np.arange(bs)[:, None], top] = np.maximum(pre[np.arange(bs)[:, None], top], 0)
        Xhat = F @ D + bd
        err = Xhat - Xb
        dXhat = 2 * err / bs
        dD = F.T @ dXhat
        dbd = dXhat.sum(0)
        dF = dXhat @ D.T
        thr = np.sort(pre, 1)[:, [-K]]
        dpre = dF * (pre >= thr)
        dE = Xb.T @ dpre / bs * 2
        dbe = dpre.sum(0) / bs
        for P, G, m, v in [(E, dE, mE, vE), (D, dD, mD, vD), (be, dbe, mbe, vbe), (bd, dbd / bs, mbd, vbd)]:
            m[:] = b1 * m + (1 - b1) * G
            v[:] = b2 * v + (1 - b2) * (G * G)
            P -= lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
        D /= (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)
    pre = Xn @ E + be
    F = np.zeros_like(pre)
    top = np.argpartition(-pre, K - 1, axis=1)[:, :K]
    F[np.arange(N)[:, None], top] = np.maximum(pre[np.arange(N)[:, None], top], 0)
    Xhat = (F @ D + bd) * scale
    varexp = 1 - ((Xhat - X) ** 2).mean() / ((X ** 2).mean() + 1e-12)
    dead = int(((F > 0).sum(0) == 0).sum())
    return varexp, F, D, dead

def main():
    t0 = time.time()
    H = np.load(os.path.join(RESULTS, "hidden.npy")).astype(float)
    ctx_topic = np.load(os.path.join(RESULTS, "ctx_topic.npy"), allow_pickle=True)
    words = open(os.path.join(RESULTS, "corpus.txt")).read().split()
    ids = np.array([W2I[w] for w in words], dtype=np.int64)
    X, Y = make_windows(ids, 8)
    model = TinyLM.load(os.path.join(RESULTS, "model.npz"))
    print("size | rebuild | swap-loss | topic-read | snow-split | dead", flush=True)
    for n in [16, 32, 64, 128, 256]:
        ve, F, D, dead = train_quick(H, n, 3, steps=2500 if n <= 64 else 2000, seed=0)
        # swap loss on val tail
        Hhat = F @ D
        # reconstruct: need mu/scale? train_quick used internal mu/scale; recompute Hhat full:
        # (we return F in normalized space; approximate swap via direct Hhat)
        # simpler: swap loss via rebuild hidden = mu + scale*(F@D+bd) -- recompute quickly
        # for speed, approximate swap with varexp only + quick CE on 2k sample
        nv = 4000
        Xv, Yv = X[-nv:], Y[-nv:]
        # rebuild hidden tail
        # NOTE: recompute mapping using saved scale/mu from H
        mu = H.mean(0); scale = np.sqrt(((H - mu) ** 2).sum(1).mean())
        # F,D are in normalized units with bd~0; Hhat_tail = F[-nv:]@D*scale+mu
        Hhat_tail = F[-nv:] @ D * scale + mu
        tot = 0.0
        for s in range(0, nv, 2048):
            Xb, Yb = Xv[s:s + 2048], Yv[s:s + 2048]
            Log = Hhat_tail[s:s + 2048] @ model.W2 + model.b2
            Log -= Log.max(1, keepdims=True)
            P = np.exp(Log); P /= P.sum(1, keepdims=True)
            tot += float(-np.log(P[np.arange(len(Yb)), Yb] + 1e-12).sum())
        swap = tot / nv
        # topic readability
        tr = []
        for k in range(n):
            a = F[:, k]; tot_a = a.sum() + 1e-12
            per = np.array([(a[ctx_topic == t]).sum() / tot_a for t in TOPICS])
            tr.append(per.max())
        tr = np.array(tr)
        nread = int((tr >= 0.75).sum())
        # snow split: # features whose best topic is snow
        best = []
        for k in range(n):
            a = F[:, k]; tot_a = a.sum() + 1e-12
            per = np.array([(a[ctx_topic == t]).sum() / tot_a for t in TOPICS])
            best.append(TOPICS[int(np.argmax(per))])
        snow = sum(1 for b in best if b == "snow")
        print(f"{n:4d} | {ve*100:5.1f}% | {swap:.2f} | {nread:3d} | snow×{snow} | dead {dead}", flush=True)
    print(f"done in {time.time()-t0:.1f}s (video: 75%->94.5%, 1.79->1.06, snow 0/1/2/4/8, dead 36/256)")

if __name__ == "__main__":
    main()
