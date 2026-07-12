# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "pillow"]
# ///
"""5 paired anterior/posterior whole body samples in one figure."""
import random
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

RAW = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\bs80k-imaging-raw")
ANT_DIR = RAW / "wholeBodyANT"
POST_DIR = RAW / "wholeBodyPOST"
OUT = Path(__file__).resolve().parents[2] / "result" / "dataset_samples" / "ant_post_samples.png"
N = 5

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.weight"] = "normal"

ant_ids = {int(p.stem) for p in ANT_DIR.glob("*.jpg")}
post_ids = {int(p.stem) for p in POST_DIR.glob("*.jpg")}
ids = random.Random(0).sample(sorted(ant_ids & post_ids), N)

w, h = Image.open(ANT_DIR / f"{ids[0]}.jpg").size
scale = 5 / (2 * h)
fig, axes = plt.subplots(2, N, figsize=(N * w * scale, 2 * h * scale))

for col, i in enumerate(ids):
    axes[0, col].imshow(Image.open(ANT_DIR / f"{i}.jpg"), cmap="gray")
    axes[1, col].imshow(Image.open(POST_DIR / f"{i}.jpg"), cmap="gray")

for ax in axes.flat:
    ax.axis("off")

fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight", pad_inches=0)
print(f"saved {OUT} for ids {ids}")
