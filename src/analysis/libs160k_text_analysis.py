# /// script
# requires-python = ">=3.11"
# dependencies = ["scikit-learn", "scipy", "matplotlib", "pandas"]
# ///
"""Word, cluster, and pattern analysis of LIBS-160K's caption templates.

Only 39 distinct caption strings exist in the whole dataset (context/libs160k.md), 13
regions times 3 caption types, so this analyzes the 39 templates themselves, not per
image free text, there is no per image free text to analyze.
"""
import json
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score

LIBS = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\libs160k-imaging-raw\LIBS-160K-EN")
OUT_FIGURE = Path(__file__).resolve().parents[2] / "result" / "figures" / "libs160k_text_dendrogram.png"

plt.rcParams["font.family"] = "Times New Roman"

REGION_PHRASES = {
    "right chest": "chest_R", "left chest": "chest_L",
    "right shoulder joint": "shoulder_R", "left shoulder joint": "shoulder_L",
    "right knee joint": "knee_R", "left knee joint": "knee_L",
    "right elbow joint": "elbow_R", "left elbow joint": "elbow_L",
    "right ankle joint": "ankle_R", "left ankle joint": "ankle_L",
    "pelvis": "pelvis", "vertebrae": "vertebrae", "head": "head",
}

records = [json.loads(l) for l in (LIBS / "train" / "train_texts.jsonl").read_text(encoding="utf-8").splitlines()]
texts = [r["text"] for r in records]
counts = [len(r["image_ids"]) for r in records]
n = len(texts)
print(f"{n} caption templates, {sum(counts)} total image assignments\n")

# --- word / vocabulary analysis ---
all_words = re.findall(r"[a-z']+", " ".join(texts).lower())
word_freq = Counter(all_words)
print("=== word frequency ===")
print(f"{len(all_words)} word tokens, {len(word_freq)} unique words")
for word, c in word_freq.most_common(15):
    print(f"  {word:12s} {c}")

bigrams = Counter()
for t in texts:
    tokens = re.findall(r"[a-z']+", t.lower())
    bigrams.update(zip(tokens, tokens[1:]))
print("\n=== most common bigrams ===")
for bg, c in bigrams.most_common(10):
    print(f"  {' '.join(bg):25s} {c}")

# --- caption type classification, checked against the text itself ---
def classify(t: str) -> str:
    tl = t.lower()
    if tl.startswith("this is an image of"):
        return "description"
    if "does not show" in tl:
        return "normal"
    return "abnormal"

types = [classify(t) for t in texts]
print("\n=== caption type counts ===")
print(Counter(types))

# --- region detection, checked, not assumed one match per caption ---
def find_region(t: str) -> str:
    tl = t.lower()
    matches = [code for phrase, code in REGION_PHRASES.items() if phrase in tl]
    if len(matches) != 1:
        return f"AMBIGUOUS:{matches}"
    return matches[0]

regions = [find_region(t) for t in texts]
print("\n=== region detection ===")
bad = [t for t, r in zip(texts, regions) if r.startswith("AMBIGUOUS")]
print(f"{len(bad)} captions with 0 or 2+ region matches" + (f": {bad}" if bad else ", every caption matched exactly one region"))

# --- pattern / skeleton extraction: mask the region phrase, see what is left ---
def skeleton(t: str, region_code: str) -> str:
    for phrase, code in REGION_PHRASES.items():
        if code == region_code:
            t = re.sub(re.escape(phrase), "REGION", t, flags=re.IGNORECASE)
            break
    t = re.sub(r"\b(a|the) patient's\b", "_ patient's", t, flags=re.IGNORECASE)
    return t

skeletons = [skeleton(t, r) for t, r in zip(texts, regions)]
print("\n=== unique sentence skeletons per caption type ===")
for typ in ["description", "abnormal", "normal"]:
    uniq = sorted(set(s for s, ty in zip(skeletons, types) if ty == typ))
    print(f"\n{typ}: {len(uniq)} distinct skeleton(s)")
    for u in uniq:
        which = [r for s, r, ty in zip(skeletons, regions, types) if ty == typ and s == u]
        print(f"  ({len(which)}x, e.g. {which[0]}) {u}")

# --- unsupervised clustering on the raw text, no region/type labels used ---
vec = TfidfVectorizer()
X = vec.fit_transform(texts)

print("\n=== KMeans clustering, does unsupervised grouping recover type or region ===")
for k, truth, truth_name in [(3, types, "caption type"), (13, regions, "region")]:
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    ari = adjusted_rand_score(truth, km.labels_)
    print(f"k={k:2d} vs {truth_name:12s} ground truth: adjusted rand index = {ari:.3f} (1.0 = perfect recovery, 0.0 = no better than random)")

# --- hierarchical clustering, visualized ---
Z = linkage(X.toarray(), method="average", metric="cosine")
labels = [f"{t[:45]}..." for t in texts]
fig, ax = plt.subplots(figsize=(10, 12))
dendrogram(Z, labels=labels, orientation="left", ax=ax, leaf_font_size=7)
ax.set_xlabel("cosine distance")
fig.tight_layout()
OUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_FIGURE, dpi=200)
print(f"\nsaved {OUT_FIGURE}")
