# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "pillow"]
# ///
"""5 paired anterior/posterior samples per bone region, one figure per region."""
import random
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

RAW = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\bs80k-imaging-raw")
OUT_DIR = Path(__file__).resolve().parents[2] / "result" / "analysis"
N = 5

REGIONS = [
    "ankleL", "ankleR", "chestL", "chestR", "elbowL", "elbowR",
    "head", "kneeL", "kneeR", "pelvis", "shoL", "shoR", "vertbra",
]

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.weight"] = "normal"


def make_figure(name: str) -> None:
    ant_dir = RAW / f"{name}ANT"
    post_dir = RAW / f"{name}POST"
    ant_ids = {int(p.stem) for p in ant_dir.glob("*.jpg")}
    post_ids = {int(p.stem) for p in post_dir.glob("*.jpg")}
    ids = random.Random(0).sample(sorted(ant_ids & post_ids), N)

    w, h = Image.open(ant_dir / f"{ids[0]}.jpg").size
    scale = 5 / (2 * h)
    fig, axes = plt.subplots(2, N, figsize=(N * w * scale, 2 * h * scale))

    for col, i in enumerate(ids):
        axes[0, col].imshow(Image.open(ant_dir / f"{i}.jpg"), cmap="gray")
        axes[1, col].imshow(Image.open(post_dir / f"{i}.jpg"), cmap="gray")

    for ax in axes.flat:
        ax.axis("off")

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02, hspace=0.02)
    out = OUT_DIR / f"{name}_samples.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"saved {out} for ids {ids}")


OUT_DIR.mkdir(parents=True, exist_ok=True)
for region in REGIONS:
    make_figure(region)
