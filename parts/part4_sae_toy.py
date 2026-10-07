"""Part 4. Sparse autoencoder on the toy (ground truth known).
Dots = 2 toy neurons at 20k moments. Dict of 8 directions. Dense vs TopK(K=1).
Usage: python3 parts/part4_sae_toy.py
"""
import numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS

def gen_dots(Wtrue, btrue, p=0.1, n=20000, seed=5):
    rng = np.random.default_rng(seed)
    mask = (rng.random((n, 5)) < p).astype(float)
    vals = rng.uniform(0.5, 1.5, (n, 5)) * mask
    H = vals @ Wtrue.T  # (n,2) dots
    return H, vals

def train_sae(H, n_dict=8, K=None, steps=3000, seed=0):
    rng = np.random.default_rng(seed)
    N, d = H.shape
    D = rng.normal(0, 1, (n_dict, d)); D /= (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)
    E = D.T.copy()  # tied init (best practice, Gao et al 2024)
    be = np.zeros(n_dict); bd = np.zeros(d)
    mE = np.zeros_like(E); vE = np.zeros_like(E)
    mD = np.zeros_like(D); vD = np.zeros_like(D)
    mbe = np.zeros_like(be); vbe = np.zeros_like(be)
    mbd = np.zeros_like(bd); vbd = np.zeros_like(bd)
    b1, b2, eps, lr, t = 0.9, 0.999, 1e-8, 0.01, 0
    for s in range(steps):
        t += 1
        idx = rng.integers(0, N, 512)
        Xb = H[idx] - H.mean(0)  # center like paper best-practice
        pre = (Xb - bd) @ E + be
        if K is None:
            F = np.maximum(pre, 0)
        else:
            F = np.zeros_like(pre)
            top = np.argpartition(-pre, K - 1, axis=1)[:, :K]
            F[np.arange(len(Xb))[:, None], top] = np.maximum(pre[np.arange(len(Xb))[:, None], top], 0)
        Xhat = F @ D + bd
        err = Xhat - Xb
        loss = (err ** 2).mean()
        dXhat = 2 * err / err.size
        dD = F.T @ dXhat + 0.0
        dbd = dXhat.sum(0)
        dF = dXhat @ D.T
        dpre = dF * ((F > 0).astype(float) if K is None else (pre >= np.sort(pre, 1)[:, [-K]]).astype(float))
        dE = (Xb - bd).T @ dpre if False else (Xb[:, :, None] * dpre[:, None, :]).sum(0) / 1  # (d, dict)
        # simpler: dE[j,k] = sum_b (Xb[b,j]-bd[j]) * dpre[b,k] / B? use mean-scale
        dE = ((Xb - bd).T @ dpre) / len(Xb) * 2
        dbe = dpre.sum(0) / len(Xb)
        for P, G, m, v in [(E, dE, mE, vE), (D, dD * 0.5, mD, vD), (be, dbe, mbe, vbe), (bd, dbd / len(Xb), mbd, vbd)]:
            m[:] = b1 * m + (1 - b1) * G
            v[:] = b2 * v + (1 - b2) * (G * G)
            P -= lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
        D /= (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)
    # eval full
    Xc = H - H.mean(0)
    pre = (Xc - bd) @ E + be
    if K is None:
        F = np.maximum(pre, 0)
    else:
        F = np.zeros_like(pre)
        top = np.argpartition(-pre, K - 1, axis=1)[:, :K]
        F[np.arange(N)[:, None], top] = np.maximum(pre[np.arange(N)[:, None], top], 0)
    Xhat = F @ D + bd
    varexp = 1 - ((Xhat - Xc) ** 2).mean() / ((Xc ** 2).mean() + 1e-12)
    l0 = float((F > 0).sum(1).mean())
    return D, E, varexp, l0, F

def main():
    rng = np.random.default_rng(0)
    # true arrows: regular pentagon (from part3 finding)
    angs = np.deg2rad(np.array([0, 72, 144, 216, 288]))
    Wtrue = np.stack([np.cos(angs), np.sin(angs)], 0)  # (2,5)
    H, _ = gen_dots(Wtrue, None)
    # dense SAE
    Dd, _, vd, l0d, _ = train_sae(H, 8, K=None, steps=2500, seed=1)
    print(f"dense SAE: rebuild={vd*100:.1f}% L0={l0d:.1f} (video: 100%, L0~3)")
    # TopK K=1
    Dk, _, vk, l0k, Fk = train_sae(H, 8, K=1, steps=2500, seed=1)
    print(f"TopK K=1 : rebuild={vk*100:.1f}% L0={l0k:.1f} (video: 96.9%, L0=1)")
    # match: cosine of each true arrow to nearest dict atom
    def match(Wt, D):
        Wt_n = Wt / (np.linalg.norm(Wt, axis=0, keepdims=True) + 1e-12)
        Dn = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)
        sims = np.abs(Wt_n.T @ Dn.T).max(1)
        return sims
    print(" dense match:", np.round(match(Wtrue, Dd), 3))
    print(" TopK  match:", np.round(match(Wtrue, Dk), 3), "(video >=0.998)")
    # count how many dict atoms used
    use = (Fk > 0).sum(0)
    print(" TopK usage per atom:", use, "dead:", int((use == 0).sum()))
    np.savez(os.path.join(RESULTS, "part4.npz"), Dtopk=Dk, Ddense=Dd,
             varexp_dense=vd, varexp_topk=vk)
    print("saved part4.npz")

if __name__ == "__main__":
    main()
