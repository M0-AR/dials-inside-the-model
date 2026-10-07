"""Part 6. Reading the features. Topic concentration + next-word-kind.
Usage: python3 parts/part6_read.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, TOPICS, KIND, W2I, I2W

def main():
    F = np.load(os.path.join(RESULTS, "features64.npy")).astype(float)  # (N,64)
    ctx_topic = np.load(os.path.join(RESULTS, "ctx_topic.npy"), allow_pickle=True)
    words = open(os.path.join(RESULTS, "corpus.txt")).read().split()
    N = len(F)
    # next word + kind per moment: next id = words[8+i]
    next_words = words[8:8 + N]
    kinds = [KIND[w][0] for w in next_words]  # person/thing/place/action/describing/liking/small/end
    t2i = {t: i for i, t in enumerate(TOPICS)}
    # topic concentration per feature: share of activation mass per topic
    topic_readable, topic_of = [], []
    for k in range(64):
        a = F[:, k]
        tot = a.sum() + 1e-12
        per = np.array([(a[ctx_topic == t]).sum() / tot for t in TOPICS])
        j = int(np.argmax(per))
        topic_readable.append(float(per[j]))
        topic_of.append(TOPICS[j])
    # next-kind concentration
    uniq_kinds = sorted(set(kinds))
    kind_readable, kind_of = [], []
    for k in range(64):
        a = F[:, k]
        tot = a.sum() + 1e-12
        arr = np.array([a[np.array(kinds) == kk].sum() / tot for kk in uniq_kinds])
        j = int(np.argmax(arr))
        kind_readable.append(float(arr[j]))
        kind_of.append(uniq_kinds[j])
    # readable = >=75% in one topic (video rule) ; plus next-word family
    is_topic = np.array(topic_readable) >= 0.75
    print(f"topic-readable(>=75%): {is_topic.sum()} (video 16)")
    for k in np.where(is_topic)[0]:
        print(f"  f{k}: {topic_of[k]} {topic_readable[k]*100:.1f}% fire={100*(F[:,k]>0).mean():.2f}%")
    # next-word family: not topic-readable, but >=60% single next-kind and kind in person/thing/place/describing/liking/small/end
    is_next = (~is_topic) & (np.array(kind_readable) >= 0.60)
    print(f"next-word-readable(>=60%): {is_next.sum()} (video 24)")
    for k in np.where(is_next)[0][:24]:
        print(f"  f{k}: next={kind_of[k]} {kind_readable[k]*100:.1f}% (topic-best {topic_of[k]} {topic_readable[k]*100:.0f}%)")
    unclear = 64 - int(is_topic.sum()) - int(is_next.sum())
    print(f"unclear: {unclear} (video 24)")
    # spotlight: best sea feature + best place-next feature
    sea_cands = [k for k in range(64) if topic_of[k] == "sea"]
    if sea_cands:
        k = max(sea_cands, key=lambda k: topic_readable[k])
        print(f"sea spotlight: f{k} {topic_readable[k]*100:.1f}%")
    # save mapping
    np.savez(os.path.join(RESULTS, "part6.npz"),
             topic_readable=np.array(topic_readable), topic_of=np.array(topic_of),
             kind_readable=np.array(kind_readable), kind_of=np.array(kind_of),
             is_topic=is_topic, is_next=is_next)
    print("saved part6.npz")

if __name__ == "__main__":
    main()
