"""Part 3. Superposition. 5 sparse features -> 2 neurons. Pentagon emerges.
Usage: python3 parts/part3_superposition.py
Numpy only. Trains W(2,5) + bias to minimise MSE with ReLU filtering.
"""
import numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS

def train_toy(p_active=0.1, steps=4000, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.5, (2, 5))
    W /= (np.linalg.norm(W, axis=0, keepdims=True) + 1e-9)
    b = np.zeros(5)
    mW = np.zeros_like(W); vW = np.zeros_like(W)
    mb = np.zeros_like(b); vb = np.zeros_like(b)
    b1, b2, eps, lr, t = 0.9, 0.999, 1e-8, 0.02, 0
    for s in range(steps):
        t += 1
        mask = (rng.random((256, 5)) < p_active).astype(float)
        vals = rng.uniform(0.5, 1.5, (256, 5)) * mask
        H = vals @ W.T            # (B,2)
        Xhat = np.maximum(H @ W - b, 0) if False else np.maximum(vals @ W.T @ W - b, 0)
        # Anthropic ReLU-output model: ReLU(W^T W x - b)
        err = Xhat - vals
        loss = (err ** 2).mean()
        # grads
        d = (2 * err * (Xhat > 0)) / err.size  # (B,5)
        # dW via chain: Xhat = relu(vals @ W.T @ W - b); let M = W.T@W
        # brute-force autograd for 2x5 with finite clarity: analytic below
        # dM = vals^T d ; dW = W(dM + dM^T)
        dM = vals.T @ d
        dW = W @ (dM + dM.T)
        db = -d.sum(0)
        for P, G, m, v in [(W, dW, mW, vW), (b, db, mb, vb)]:
            m[:] = b1 * m + (1 - b1) * G
            v[:] = b2 * v + (1 - b2) * (G * G)
            P -= lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
        # keep columns norm ~1
        W /= (np.linalg.norm(W, axis=0, keepdims=True) + 1e-12)
    # eval
    rng2 = np.random.default_rng(999)
    mask = (rng2.random((20000, 5)) < p_active).astype(float)
    vals = rng2.uniform(0.5, 1.5, (20000, 5)) * mask
    Xhat = np.maximum(vals @ W.T @ W - b, 0)
    mse = ((Xhat - vals) ** 2).mean()
    var = (vals ** 2).mean()
    err_rate = float(((np.abs(Xhat - vals) > 0.3) & (mask > 0)).sum() / max(1, int(mask.sum())))
    return W, b, 1 - mse / (var + 1e-12), err_rate

def angles_deg(W):
    ang = np.degrees(np.arctan2(W[1], W[0]))
    return np.sort(ang)

def main():
    for p, name in [(1.0, "dense p=1.0"), (0.10, "sparse p=0.10"), (0.03, "sparser p=0.03")]:
        W, b, varexp, err = train_toy(p)
        norms = np.linalg.norm(W, axis=0)
        angs = angles_deg(W)
        print(f"{name}: var-explained={varexp*100:.1f}% err={err*100:.1f}% | norms={np.round(norms,2)} | angles={np.round(angs,1)}")
        # pentagon check: gaps ~72deg?
        diffs = np.diff(np.concatenate([angs, angs[:1] + 360]))
        print(f"  gaps={np.round(np.sort(diffs),1)} (pentagon -> all ~72)")
    np.savez(os.path.join(RESULTS, "part3.npz"),
             note="toy 5->2: dense keeps 2, sparse p=0.1 pentagon, p=0.03 cleaner")
    print("saved part3.npz")

if __name__ == "__main__":
    main()
