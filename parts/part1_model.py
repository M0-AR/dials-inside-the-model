"""Part 1. The model. Build tiny LM (188 words, 12 topics), train, report surprise.
Usage: python3 parts/part1_model.py  (writes results/part1.npz + corpus.txt)
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import gen_corpus, make_windows, TinyLM, W2I, I2W, WORDS, RESULTS, TOPICS

def sample_sentence(model, w2i, i2w, seed_words, steps=12, rng=None):
    rng = rng or np.random.default_rng(0)
    ctx = list(seed_words)
    out = list(seed_words)
    for _ in range(steps):
        window = ctx[-8:]
        while len(window) < 8:
            window = ["the"] + window
        ids = np.array([[w2i[w] for w in window]])
        _, _, _, H1, Log = model.forward(ids)
        Log = Log - Log.max()
        P = np.exp(Log)[0]; P /= P.sum()
        nxt = int(rng.choice(len(P), p=P))
        out.append(i2w[nxt])
        ctx.append(i2w[nxt])
        if i2w[nxt] == ".":
            break
    return " ".join(out)

def main():
    t0 = time.time()
    words, topics = gen_corpus(270_000)
    with open(os.path.join(RESULTS, "corpus.txt"), "w") as f:
        f.write(" ".join(words))
    np.save(os.path.join(RESULTS, "corpus_topics.npy"), np.array(topics))
    ids = np.array([W2I[w] for w in words], dtype=np.int64)
    X, Y = make_windows(ids, 8)
    # train/val split
    ntr = int(len(X) * 0.95)
    model = TinyLM()
    print(f"params={model.params()} (expect 9284)", flush=True)
    blind = float(np.log(len(WORDS)))
    print(f"blind surprise={blind:.2f}", flush=True)
    model.train(X[:ntr], Y[:ntr], epochs=10, bs=1024, lr=0.01)
    _, P = model.ce_loss(model.forward(X[ntr:])[4], Y[ntr:])
    # proper val loss
    _, _, _, _, Logv = model.forward(X[ntr:])
    lossv, _ = TinyLM.ce_loss(Logv, Y[ntr:])
    print(f"val surprise={lossv:.2f} (target ~1.03)", flush=True)
    # validity: % generated sentences following one of 5 patterns (approx: end with .)
    rng = np.random.default_rng(1)
    valid = 0; trials = 200
    samples = []
    for i in range(trials):
        s = sample_sentence(model, W2I, I2W, ["the", "teacher", "teaches", "the"] if i % 2 == 0 else ["in", "the", "ward", "the", "doctor", "treats", "a", "pill"], rng=rng)
        samples.append(s)
        if s.strip().endswith("."):
            valid += 1
    print(f"valid sentences: {valid/trials*100:.1f}% (target ~95%)", flush=True)
    for s in samples[:3]:
        print(" sample:", s)
    # demo prediction: 'The teacher teaches the __'
    demo = ["the", "teacher", "teaches", "the", "teacher", "teaches", "the", "teacher"]
    # use school context
    ctx_ids = np.array([[W2I[w] for w in ["the", "teacher", "teaches", "the", "test", "and", "the", "teacher"]]])
    _, _, _, _, Log = model.forward(ctx_ids)
    P = np.exp(Log - Log.max()); P /= P.sum()
    top = np.argsort(-P[0])[:4]
    print("P(next | school context):", [(I2W[i], round(float(P[0][i]), 3)) for i in top])
    model.save(os.path.join(RESULTS, "model.npz"))
    np.savez(os.path.join(RESULTS, "part1.npz"), val_loss=lossv, blind=blind,
             valid_rate=valid / trials, seconds=time.time() - t0)
    print(f"done in {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
