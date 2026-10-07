"""Part 7. Steering. Clamp one feature, generate, measure topic share + validity.
Usage: python3 parts/part7_steer.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import TinyLM, W2I, I2W, RESULTS, TOPICS, LEX, KIND

def topic_words(t):
    d = LEX[t]
    return set(d["people"] + d["things"] + d["places"] + d["actions"])

def generate(model, sae, clamp_idx=None, clamp_val=None, n_words=2000, seed=0, ctx_init=None):
    rng = np.random.default_rng(seed)
    z = np.load(os.path.join(RESULTS, "sae64.npz"))
    D, E, be, bd, mu, scale = z["D"], z["E"], z["be"], z["bd"], z["mu"], float(z["scale"])
    K = 3
    ctx = list(ctx_init or ["the", "teacher", "teaches", "the", "test", "and", "the", "school"])
    out = []
    for _ in range(n_words):
        ids = np.array([[W2I[w] for w in ctx[-8:]]])
        _, _, _, H1, _ = model.forward(ids)
        if clamp_idx is not None:
            Xc = (H1 - mu) / scale
            pre = Xc @ E + be
            # TopK current value for this feature
            thr = np.sort(pre, 1)[:, [-K]]
            active = (pre >= thr) & (pre > 0)
            fcur = float(pre[0, clamp_idx]) if active[0, clamp_idx] else 0.0
            if fcur < 0:
                fcur = 0.0
            H1 = H1 + (clamp_val - fcur) * D[clamp_idx][None, :] * scale
        Log = H1 @ model.W2 + model.b2
        Log = Log - Log.max()
        P = np.exp(Log)[0]; P /= P.sum()
        nxt = int(rng.choice(len(P), p=P))
        w = I2W[nxt]
        out.append(w)
        ctx.append(w)
    return out

def score_topic(words, t):
    tw = topic_words(t)
    toks = [w for w in words if KIND.get(w, ("?",))[0] in ("person", "thing", "place", "action")]
    if not toks:
        return 0.0
    return sum(1 for w in toks if w in tw) / len(toks) * 100

def validity(words):
    # strict: split on '.', each sentence must match one of the 5 templates
    # templates (without trailing .):
    # 0: the DESC person action the DESC thing (8 tokens)
    # 1: the person action the thing (5)
    # 2: a person and a person liking the place (8)
    # 3: in the place the person action a thing (8)
    # 4: a person liking a DESC thing near the place (9)
    from common import LEX, DESCRIBING, LIKING, SMALL, TOPICS
    people = set(sum([LEX[t]["people"] for t in TOPICS], []))
    things = set(sum([LEX[t]["things"] for t in TOPICS], []))
    places = set(sum([LEX[t]["places"] for t in TOPICS], []))
    acts = set(sum([LEX[t]["actions"] for t in TOPICS], []))
    desc = set(DESCRIBING); lik = set(LIKING)
    sents = []
    cur = []
    for w in words:
        cur.append(w)
        if w == ".":
            sents.append(cur); cur = []
    if not sents:
        return 0.0
    def ok(s):
        b = s[:-1]
        if len(b) == 8 and b[0] == "the" and b[1] in desc and b[2] in people and b[3] in acts and b[4] == "the" and b[5] in desc and b[6] in things:
            return True
        if len(b) == 5 and b[0] == "the" and b[1] in people and b[2] in acts and b[3] == "the" and b[4] in things:
            return True
        if len(b) == 8 and b[0] == "a" and b[1] in people and b[2] == "and" and b[3] == "a" and b[4] in people and b[5] in lik and b[6] == "the" and b[7] in places:
            return True
        if len(b) == 8 and b[0] == "in" and b[1] == "the" and b[2] in places and b[3] == "the" and b[4] in people and b[5] in acts and b[6] == "a" and b[7] in things:
            return True
        if len(b) == 9 and b[0] == "a" and b[1] in people and b[2] in lik and b[3] == "a" and b[4] in desc and b[5] in things and b[6] == "near" and b[7] == "the" and b[8] in places:
            return True
        return False
    return sum(1 for s in sents if ok(s)) / len(sents) * 100

def main():
    model = TinyLM.load(os.path.join(RESULTS, "model.npz"))
    p6 = np.load(os.path.join(RESULTS, "part6.npz"), allow_pickle=True)
    topic_of = [str(x) for x in p6["topic_of"]]
    # pick snow feature: highest topic concentration among snow
    snow_idx = int(np.argmax([(p6["topic_readable"][i] if topic_of[i] == "snow" else -1) for i in range(64)]))
    print(f"snow feature = f{snow_idx} ({p6['topic_readable'][snow_idx]*100:.1f}% snow)")
    z = np.load(os.path.join(RESULTS, "sae64.npz"))
    F = np.load(os.path.join(RESULTS, "features64.npy")).astype(float)
    usual = float(F[:, snow_idx][F[:, snow_idx] > 0].mean()) if (F[:, snow_idx] > 0).any() else 1.0
    print(f"usual strength ~ {usual:.2f}")
    base = generate(model, None, None, None, 2000, seed=0)
    print(f"baseline: snow-share={score_topic(base,'snow'):.1f}% (video <20% any topic) valid={validity(base):.1f}%")
    for mult, label in [(0.5, "half"), (1.0, "full"), (2.0, "2x"), (5.0, "5x")]:
        w = generate(model, None, snow_idx, usual * mult, 2000, seed=0)
        print(f" snow {label}: snow-share={score_topic(w,'snow'):.1f}% valid={validity(w):.1f}% e.g. {' '.join(w[:14])}")
    # all 12 topics: use best feature per topic (even if <75%)
    print("--- all topics @1x ---")
    ok_count = 0
    for t in TOPICS:
        cands = [i for i in range(64) if topic_of[i] == t]
        if not cands:
            print(f" {t}: NO feature (fail)"); continue
        k = max(cands, key=lambda i: p6["topic_readable"][i])
        u = float(F[:, k][F[:, k] > 0].mean()) if (F[:, k] > 0).any() else 1.0
        w = generate(model, None, k, u, 1500, seed=1)
        s = score_topic(w, t)
        v = validity(w)
        ok = s >= 75
        ok_count += ok
        print(f" {t:8s} f{k}: share={s:.0f}% valid={v:.0f}% {'OK' if ok else 'SHORT'}")
    print(f"topics reached(>=75%): {ok_count}/12 (video 8/12, music fails)")
    # neurons as handles
    print("--- neurons as handles @1x ---")
    H = np.load(os.path.join(RESULTS, "hidden.npy")).astype(float)
    umean = float(H[H > 0].mean())
    reached = 0
    for n in range(24):
        # clamp neuron n: H1[:,n] = umean
        rng = np.random.default_rng(2)
        ctx = ["the", "teacher", "teaches", "the", "test", "and", "the", "school"]
        out = []
        for _ in range(1500):
            ids = np.array([[W2I[w] for w in ctx[-8:]]])
            _, _, _, H1, _ = model.forward(ids)
            H1[0, n] = umean
            Log = H1 @ model.W2 + model.b2
            Log -= Log.max(); P = np.exp(Log)[0]; P /= P.sum()
            w = I2W[int(rng.choice(len(P), p=P))]
            out.append(w); ctx.append(w)
        best = max(score_topic(out, t) for t in TOPICS)
        if best >= 75:
            reached += 1
    print(f"neuron handles reaching a topic @1x: {reached}/12 (video 0/12)")
    np.savez(os.path.join(RESULTS, "part7.npz"), snow_idx=snow_idx, usual=usual)
    print("saved part7.npz")

if __name__ == "__main__":
    main()
