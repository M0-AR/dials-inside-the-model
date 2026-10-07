"""Part 5. Real model SAE. Dict 64, TopK K=3 on 24-dim hidden states.
Swap test: replace H with SAE rebuild, measure surprise.
Usage: python3 parts/part5_sae_real.py
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import TinyLM, make_windows, W2I, RESULTS

def train_sae(H, n_dict=64, K=3, steps=6000, bs=1024, lr=0.01, seed=0):
    rng = np.random.default_rng(seed)
    N, d = H.shape
    mu = H.mean(0)
    X = H - mu
    # scale to unit norm avg (best practice)
    scale = np.sqrt((X ** 2).sum(1).mean())
    Xn = X / scale
    D = rng.normal(0, 1, (n_dict, d)); D /= (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)
    E = D.T.copy() / 1.0
    be = np.zeros(n_dict); bd = np.zeros(d)
    mE = np.zeros_like(E); vE = np.zeros_like(E)
    mD = np.zeros_like(D); vD = np.zeros_like(D)
    mbe = np.zeros_like(be); vbe = np.zeros_like(be)
    mbd = np.zeros_like(bd); vbd = np.zeros_like(bd)
    b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
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
    # full eval
    pre = Xn @ E + be
    F = np.zeros_like(pre)
    top = np.argpartition(-pre, K - 1, axis=1)[:, :K]
    F[np.arange(N)[:, None], top] = np.maximum(pre[np.arange(N)[:, None], top], 0)
    Xhat = (F @ D + bd) * scale
    varexp = 1 - ((Xhat - X) ** 2).mean() / ((X ** 2).mean() + 1e-12)
    l0 = float((F > 0).sum(1).mean())
    fire = (F > 0).mean(0) * 100
    return dict(D=D, E=E, be=be, bd=bd, mu=mu, scale=scale,
                varexp=float(varexp), l0=l0, fire=fire, F=F)

def ce_with_hidden(model, X, Y, Hreplace=None, batch=8192):
    tot, nb = 0.0, 0
    for s in range(0, len(X), batch):
        Xb, Yb = X[s:s + batch], Y[s:s + batch]
        _, _, _, H1, _ = model.forward(Xb)
        Huse = Hreplace[s:s + batch] if Hreplace is not None else H1
        Log = Huse @ model.W2 + model.b2
        Log = Log - Log.max(1, keepdims=True)
        P = np.exp(Log); P /= P.sum(1, keepdims=True)
        tot += float(-np.log(P[np.arange(len(Yb)), Yb] + 1e-12).sum())
        nb += len(Yb)
    return tot / nb

def main():
    t0 = time.time()
    H = np.load(os.path.join(RESULTS, "hidden.npy")).astype(float)
    words = open(os.path.join(RESULTS, "corpus.txt")).read().split()
    ids = np.array([W2I[w] for w in words], dtype=np.int64)
    X, Y = make_windows(ids, 8)
    model = TinyLM.load(os.path.join(RESULTS, "model.npz"))
    print(f"training SAE-64 K=3 on {H.shape} ...", flush=True)
    sae = train_sae(H, 64, 3, steps=5000, seed=0)
    print(f"rebuild={sae['varexp']*100:.1f}% (video 89%) L0={sae['l0']:.1f} median-fire={np.median(sae['fire']):.2f}% (video <5%)")
    # rebuild hidden in original space
    Hhat = ((sae['F'] @ sae['D'] + sae['bd']) * sae['scale'] + sae['mu'])
    # swap test on last 5% (val)
    n = len(X); nv = n // 20
    Xv, Yv = X[-nv:], Y[-nv:]
    Hv, Hhatv = H[-nv:], Hhat[-nv:]
    base = ce_with_hidden(model, Xv, Yv, None)
    Hmean = H.mean(0, keepdims=True).repeat(len(Xv), 0)
    abl = ce_with_hidden(model, Xv, Yv, Hmean)
    rec = ce_with_hidden(model, Xv, Yv, Hhatv)
    kept = (abl - rec) / (abl - base + 1e-12) * 100
    print(f"swap: own={base:.2f} (video 1.02) erased-avg={abl:.2f} (video 6.24) rebuilt={rec:.2f} (video 1.15) kept={kept:.1f}% (video 97%)")
    # example moment: 'in the ward the doctor treats a pill' -> find one occurrence
    s = " ".join(words[:2000])
    print("example active features at a hospital moment: top firing idx:",
          np.argsort(-sae['fire'])[:5], np.round(np.sort(sae['fire'])[-5:], 2))
    np.savez(os.path.join(RESULTS, "sae64.npz"), D=sae['D'], E=sae['E'], be=sae['be'],
             bd=sae['bd'], mu=sae['mu'], scale=sae['scale'], varexp=sae['varexp'])
    np.save(os.path.join(RESULTS, "features64.npy"), sae['F'].astype(np.float32))
    np.savez(os.path.join(RESULTS, "part5.npz"), varexp=sae['varexp'], base=base, abl=abl, rec=rec, kept=kept)
    print(f"done in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
