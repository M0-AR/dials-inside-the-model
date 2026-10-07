"""Part 2. The neurons. Show 24 neurons are polysemantic (busy, flat topic bars).
Usage: python3 parts/part2_neurons.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import TinyLM, make_windows, W2I, RESULTS, TOPICS, TOPIC_OF, I2W

def main():
    words = open(os.path.join(RESULTS, "corpus.txt")).read().split()
    topics = np.load(os.path.join(RESULTS, "corpus_topics.npy"), allow_pickle=True)
    ids = np.array([W2I[w] for w in words], dtype=np.int64)
    X, Y = make_windows(ids, 8)
    model = TinyLM.load(os.path.join(RESULTS, "model.npz"))
    # batched forward to get H1 for all moments
    Hs = []
    for s in range(0, len(X), 8192):
        _, _, _, H1, _ = model.forward(X[s:s + 8192])
        Hs.append(H1)
    H = np.concatenate(Hs)  # (N, 24)
    N = len(H)
    on = (H > 0)
    per_neuron_on = on.mean(0) * 100
    print(f"typical neuron on {np.median(per_neuron_on):.1f}% of moments (video: 63%)")
    print(f"mean neurons on at once: {on.sum(1).mean():.1f} / 24 (video: 15)")
    # topic attribution: sum activity per neuron per topic
    # topic per moment = topic of *next* word? use current context topic = topics[ctx_end]
    ctx_topic = topics[8:8 + N]
    t2i = {t: i for i, t in enumerate(TOPICS)}
    attr = np.zeros((24, 12))
    for i, t in enumerate(TOPICS):
        m = (ctx_topic == t)
        attr[:, i] = H[m].sum(0)
    frac = attr / (attr.sum(1, keepdims=True) + 1e-12)
    best = frac.max(1)
    best_t = [TOPICS[int(np.argmax(frac[n]))] for n in range(24)]
    order = np.argsort(-best)
    print(f"best neuron: #{order[0]} {best[order[0]]*100:.1f}% in {best_t[order[0]]} (video: n11 18% snow)")
    for n in order[:5]:
        print(f" neuron {n}: max {best[n]*100:.1f}% ({best_t[n]})")
    cnt = int((best > 0.5).sum())
    print(f"neurons with >50% activity in one topic: {cnt} / 24 (video: 0)")
    # save for later parts
    np.save(os.path.join(RESULTS, "hidden.npy"), H.astype(np.float32))
    np.save(os.path.join(RESULTS, "part2_attr.npy"), frac)
    # also save per-moment next-word info for part6
    np.save(os.path.join(RESULTS, "ctx_topic.npy"), np.array(ctx_topic))
    print("saved hidden.npy", H.shape)

if __name__ == "__main__":
    main()
