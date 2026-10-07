"""Shared core: vocab (188 words, 12 topics), synthetic corpus, tiny MLP LM.
Plain Python + numpy only. Deterministic (seeded) for reproducibility.
"""
import os
import numpy as np

SEED = 1234
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, "results")
os.makedirs(RESULTS, exist_ok=True)

TOPICS = ["sea", "farm", "city", "kitchen", "space", "school",
          "forest", "music", "hospital", "castle", "sport", "snow"]

# Per-topic lexicon: 3 people, 4 things, 3 places, 4 actions = 14 x 12 = 168
LEX = {
 "sea":      dict(people=["captain", "sailor", "diver"], things=["boat", "net", "anchor", "whale"],
                   places=["harbor", "island", "beach"], actions=["rows", "hauls", "sails", "dives"]),
 "farm":     dict(people=["farmer", "shepherd", "milker"], things=["tractor", "wheat", "barn", "hay"],
                   places=["pasture", "meadow", "silo"], actions=["plows", "sows", "reaps", "feeds"]),
 "city":     dict(people=["driver", "clerk", "mayor"], things=["taxi", "bridge", "tram", "bus"],
                   places=["street", "plaza", "alley"], actions=["drives", "builds", "paints", "sweeps"]),
 "kitchen":  dict(people=["chef", "baker", "waiter"], things=["bread", "knife", "pan", "oven"],
                   places=["cafe", "bakery", "pantry"], actions=["bakes", "slices", "stirs", "serves"]),
 "space":    dict(people=["pilot", "alien", "engineer"], things=["rocket", "planet", "star", "comet"],
                   places=["orbit", "station", "crater"], actions=["launches", "orbits", "lands", "scans"]),
 "school":   dict(people=["teacher", "student", "pupil"], things=["pencil", "test", "book", "lesson"],
                   places=["school", "library", "class"], actions=["teaches", "grades", "reads", "writes"]),
 "forest":   dict(people=["ranger", "hunter", "hiker"], things=["tree", "river", "maple", "trail"],
                   places=["grove", "clearing", "camp"], actions=["climbs", "tracks", "camps", "hikes"]),
 "music":    dict(people=["singer", "drummer", "band"], things=["guitar", "piano", "song", "drum"],
                   places=["stage", "hall", "studio"], actions=["plays", "sings", "tunes", "records"]),
 "hospital": dict(people=["doctor", "nurse", "patient"], things=["pill", "bed", "mask", "chart"],
                   places=["ward", "clinic", "lab"], actions=["treats", "checks", "cures", "visits"]),
 "castle":   dict(people=["king", "queen", "knight"], things=["crown", "sword", "shield", "throne"],
                   places=["tower", "dungeon", "gate"], actions=["rules", "guards", "crowns", "feasts"]),
 "sport":    dict(people=["coach", "player", "referee"], things=["ball", "goal", "cup", "field"],
                   places=["stadium", "court", "track"], actions=["kicks", "scores", "trains", "wins"]),
 "snow":     dict(people=["skier", "penguin", "climber"], things=["sled", "glacier", "mountain", "snowman"],
                   places=["slope", "cabin", "peak"], actions=["slides", "freezes", "shivers", "melts"]),
}
DESCRIBING = ["big", "quiet", "happy", "tired", "small", "bright"]
LIKING = ["sees", "wants", "likes", "loves", "holds"]
SMALL = ["the", "a", "and", "in", "near", "with", "to", "of"]
PERIOD = ["."]

def build_vocab():
    words = []
    for t in TOPICS:
        d = LEX[t]
        words += d["people"] + d["things"] + d["places"] + d["actions"]
    words += DESCRIBING + LIKING + SMALL + PERIOD
    assert len(words) == 188, f"vocab={len(words)}"
    assert len(set(words)) == 188, "duplicate words!"
    w2i = {w: i for i, w in enumerate(words)}
    i2w = {i: w for w, i in w2i.items()}
    # word kinds for analysis
    kind = {}
    for t in TOPICS:
        d = LEX[t]
        for w in d["people"]:
            kind[w] = ("person", t)
        for w in d["things"]:
            kind[w] = ("thing", t)
        for w in d["places"]:
            kind[w] = ("place", t)
        for w in d["actions"]:
            kind[w] = ("action", t)
    for w in DESCRIBING:
        kind[w] = ("describing", "shared")
    for w in LIKING:
        kind[w] = ("liking", "shared")
    for w in SMALL:
        kind[w] = ("small", "shared")
    kind["."] = ("end", "shared")
    topic_of = {}
    for w in words:
        topic_of[w] = kind[w][1]
    return words, w2i, i2w, kind, topic_of

WORDS, W2I, I2W, KIND, TOPIC_OF = build_vocab()
V = len(WORDS)
N_TOPICS = len(TOPICS)

def gen_corpus(n_words=270_000, seed=SEED):
    """Stories stay on one topic for 3-6 sentences then switch. 5 patterns."""
    rng = np.random.default_rng(seed)
    out = []
    topics_seq = []  # topic per word (for analysis)
    cur = rng.integers(0, N_TOPICS)
    sent_left = 0
    while len(out) < n_words:
        if sent_left <= 0:
            # new story chunk: 3-6 sentences same topic, then maybe switch
            if rng.random() < 0.35 or sent_left == 0 and len(out) > 0 and rng.random() < 0.5:
                cur = rng.integers(0, N_TOPICS)
            sent_left = rng.integers(3, 7)
            # occasionally switch mid-chunk
            if rng.random() < 0.15:
                cur = rng.integers(0, N_TOPICS)
        t = TOPICS[cur]
        d = LEX[t]
        pat = rng.integers(0, 5)
        desc = rng.choice(DESCRIBING, size=2, replace=False)
        sent = []
        if pat == 0:  # The big band plays the quiet guitar .
            sent = ["the", desc[0], rng.choice(d["people"]), rng.choice(d["actions"]),
                    "the", desc[1], rng.choice(d["things"]), "."]
        elif pat == 1:  # The teacher grades the test .
            sent = ["the", rng.choice(d["people"]), rng.choice(d["actions"]),
                    "the", rng.choice(d["things"]), "."]
        elif pat == 2:  # A student and a pupil likes the school .
            sent = ["a", rng.choice(d["people"]), "and", "a", rng.choice(d["people"]),
                    rng.choice(LIKING), "the", rng.choice(d["places"]), "."]
        elif pat == 3:  # In the ward the doctor treats a pill .
            sent = ["in", "the", rng.choice(d["places"]), "the", rng.choice(d["people"]),
                    rng.choice(d["actions"]), "a", rng.choice(d["things"]), "."]
        else:  # A king sees a happy crown near the tower .
            sent = ["a", rng.choice(d["people"]), rng.choice(LIKING), "a", desc[0],
                    rng.choice(d["things"]), "near", "the", rng.choice(d["places"]), "."]
        out.extend(sent)
        topics_seq.extend([t] * len(sent))
        sent_left -= 1
    out = out[:n_words]
    topics_seq = topics_seq[:n_words]
    return out, topics_seq

def make_windows(ids, ctx=8):
    """X: (N, ctx) token ids, Y: (N,) next id."""
    N = len(ids) - ctx
    X = np.lib.stride_tricks.sliding_window_view(np.asarray(ids, dtype=np.int64), ctx)[:N].copy()
    Y = np.asarray(ids[ctx:ctx + N], dtype=np.int64)
    return X, Y

class TinyLM:
    """Embed(12) -> 96 -> FC(24, ReLU) -> FC(188). 9284 params."""
    def __init__(self, vocab=188, emb=12, ctx=8, hid=24, seed=7):
        rng = np.random.default_rng(seed)
        self.emb, self.ctx, self.hid, self.vocab = emb, ctx, hid, vocab
        self.E = rng.normal(0, 0.5, (vocab, emb))
        self.W1 = rng.normal(0, np.sqrt(2 / (ctx * emb)), (ctx * emb, hid))
        self.b1 = np.zeros(hid)
        self.W2 = rng.normal(0, np.sqrt(1 / hid), (hid, vocab))
        self.b2 = np.zeros(vocab)

    def params(self):
        return self.E.size + self.W1.size + self.b1.size + self.W2.size + self.b2.size

    def forward(self, Xb):
        Eb = self.E[Xb]                      # (B, ctx, emb)
        H0 = Eb.reshape(len(Xb), -1)         # (B, 96)
        Z1 = H0 @ self.W1 + self.b1
        H1 = np.maximum(Z1, 0)                # (B, hid)
        Log = H1 @ self.W2 + self.b2
        return Eb, H0, Z1, H1, Log

    @staticmethod
    def ce_loss(Log, Yb):
        Log = Log - Log.max(1, keepdims=True)
        exp = np.exp(Log)
        P = exp / exp.sum(1, keepdims=True)
        return float(-np.log(P[np.arange(len(Yb)), Yb] + 1e-12).mean()), P

    def train(self, X, Y, epochs=10, bs=1024, lr=0.01, seed=0, verbose=True):
        rng = np.random.default_rng(seed)
        N = len(X)
        # Adam state per parameter
        state = {}
        for name, p in [("E", self.E), ("W1", self.W1), ("b1", self.b1),
                        ("W2", self.W2), ("b2", self.b2)]:
            state[name] = (np.zeros_like(p), np.zeros_like(p))
        b1, b2, eps, step = 0.9, 0.999, 1e-8, 0

        def adam_update(param, grad, ms, vs):
            nonlocal step
            ms[:] = b1 * ms + (1 - b1) * grad
            vs[:] = b2 * vs + (1 - b2) * (grad * grad)
            mh = ms / (1 - b1 ** step)
            vh = vs / (1 - b2 ** step)
            param -= lr * mh / (np.sqrt(vh) + eps)

        for ep in range(epochs):
            perm = rng.permutation(N)
            tot, nb = 0.0, 0
            for s in range(0, N, bs):
                step += 1
                idx = perm[s:s + bs]
                Xb, Yb = X[idx], Y[idx]
                Eb, H0, Z1, H1, Log = self.forward(Xb)
                loss, P = self.ce_loss(Log, Yb)
                tot += loss * len(idx); nb += len(idx)
                dLog = P.copy()
                dLog[np.arange(len(Yb)), Yb] -= 1
                dLog /= len(Yb)
                dW2 = H1.T @ dLog; db2 = dLog.sum(0)
                dH1 = dLog @ self.W2.T
                dZ1 = dH1 * (Z1 > 0)
                dW1 = H0.T @ dZ1; db1 = dZ1.sum(0)
                dH0 = dZ1 @ self.W1.T
                dEb = dH0.reshape(len(Xb), self.ctx, self.emb)
                dE = np.zeros_like(self.E)
                np.add.at(dE, Xb, dEb)
                for name, param, grad in [("W2", self.W2, dW2), ("b2", self.b2, db2),
                                          ("W1", self.W1, dW1), ("b1", self.b1, db1),
                                          ("E", self.E, dE)]:
                    ms, vs = state[name]
                    adam_update(param, grad, ms, vs)
            if verbose:
                print(f"epoch {ep+1}/{epochs} loss={tot/nb:.4f}", flush=True)
        return self

    def save(self, path):
        np.savez(path, E=self.E, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        m = cls()
        m.E, m.W1, m.b1, m.W2, m.b2 = z["E"], z["W1"], z["b1"], z["W2"], z["b2"]
        return m
