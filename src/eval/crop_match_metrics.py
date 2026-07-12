# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scikit-image"]
# ///
"""Similarity and error metrics for scoring a predicted crop window against the real crop.

No independent (x, y) ground truth exists for a region crop's location in its whole body
source (see context/dataset.md), so a predicted box can only be checked by cutting a same
size window out of the whole body image at the predicted (x, y) and comparing it, pixel by
pixel, against the real crop file already on disk. Any single number here can be fooled, so
`compare` reports several angles at once, see the per-function docstrings for what each one
catches that the others can miss.
"""
import numpy as np
from skimage.metrics import structural_similarity


def near_exact_fraction(real: np.ndarray, pred: np.ndarray, tol: int = 2) -> float:
    """Share of pixels within `tol` gray levels of each other.

    Exact/near-exact pixel agreement. Directly catches gross misalignment or wrong content,
    but real photon count noise means even a correct match will not hit fraction 1.0 at
    tol=0, hence the small tolerance rather than a literal equality check.
    """
    return float(np.mean(np.abs(real.astype(np.int16) - pred.astype(np.int16)) <= tol))


def mean_abs_error(real: np.ndarray, pred: np.ndarray) -> float:
    """Mean absolute pixel difference, magnitude of error.

    Typical per-pixel deviation, and unlike root_mean_sq_error below it is not dominated by
    one outlier pixel or hot spot, so it stays a fair "how far off on average" number.
    """
    return float(np.mean(np.abs(real.astype(np.float64) - pred.astype(np.float64))))


def root_mean_sq_error(real: np.ndarray, pred: np.ndarray) -> float:
    """Root mean squared pixel difference, magnitude of error.

    Squares differences before averaging, so one bright misaligned hot spot pulls this up
    far more than it pulls mean_abs_error up. A wide rmse/mae gap is itself a signal that a
    small region, not a uniform offset, is driving the mismatch.
    """
    return float(np.sqrt(np.mean((real.astype(np.float64) - pred.astype(np.float64)) ** 2)))


def ssim(real: np.ndarray, pred: np.ndarray) -> float:
    """Structural similarity index, structural/perceptual.

    Compares local luminance, contrast, and structure window by window rather than as one
    global average, so it catches local structural distortion, for example a partial overlap
    or blur, that a single correlation or histogram number can still look fine under.
    """
    data_range = max(real.max(), pred.max()) - min(real.min(), pred.min())
    if data_range == 0:
        data_range = 1
    return float(structural_similarity(real, pred, data_range=data_range))


def pearson_corr(real: np.ndarray, pred: np.ndarray) -> float:
    """Pearson correlation between flattened pixel values, correlation based.

    Invariant to a uniform brightness or contrast offset between the two windows, so it can
    confirm the spatial pattern lines up even when such an offset inflates mean_abs_error or
    root_mean_sq_error. That same invariance is its blind spot, a wrong crop with a rescaled
    version of a similar pattern could still score high here, hence pairing it with the error
    metrics above rather than using it alone.
    """
    r, p = real.astype(np.float64).flatten(), pred.astype(np.float64).flatten()
    if r.std() == 0 or p.std() == 0:
        return 1.0 if np.array_equal(r, p) else 0.0
    return float(np.corrcoef(r, p)[0, 1])


def hist_intersection(real: np.ndarray, pred: np.ndarray, bins: int = 256) -> float:
    """Normalized histogram intersection, distribution based.

    Ignores pixel position entirely and compares only the mix of gray values present, so it
    checks whether the two windows contain the same kind of content at all. High
    intersection alongside poor pixel-wise metrics points to a small spatial offset rather
    than a wrong region, low intersection points to a wrong region regardless of alignment.
    """
    lo = min(real.min(), pred.min())
    hi = max(real.max(), pred.max())
    if hi == lo:
        hi = lo + 1
    h_real, _ = np.histogram(real, bins=bins, range=(lo, hi))
    h_pred, _ = np.histogram(pred, bins=bins, range=(lo, hi))
    return float(np.minimum(h_real, h_pred).sum() / h_real.sum())


def compare(real: np.ndarray, pred: np.ndarray, tol: int = 2, mask: np.ndarray | None = None) -> dict[str, float]:
    """Run all metrics above on one real crop vs predicted window pair, same shape required.

    mask, if given, restricts every metric to the masked-in pixels. Some crops are padded
    with a black region the source pipeline added, not real anatomy, comparing that padding
    against real whole body content there would penalize an otherwise correct match for
    content the crop was never meant to carry in the first place.
    """
    assert real.shape == pred.shape, f"shape mismatch {real.shape} vs {pred.shape}"
    if mask is None:
        real_flat, pred_flat, real_2d, pred_2d = real, pred, real, pred
    else:
        keep = mask.astype(bool)
        real_flat, pred_flat = real[keep], pred[keep]
        real_2d, pred_2d = np.where(keep, real, 0), np.where(keep, pred, 0)
    return {
        "near_exact_fraction": near_exact_fraction(real_flat, pred_flat, tol),
        "mae": mean_abs_error(real_flat, pred_flat),
        "rmse": root_mean_sq_error(real_flat, pred_flat),
        "ssim": ssim(real_2d, pred_2d),
        "pearson_corr": pearson_corr(real_flat, pred_flat),
        "hist_intersection": hist_intersection(real_flat, pred_flat),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    base = np.zeros((40, 40), dtype=np.uint8)
    base[10:30, 10:30] = 180
    base[15:25, 15:25] = 90
    base = np.clip(base.astype(np.int16) + rng.integers(0, 10, base.shape), 0, 255).astype(np.uint8)

    identical = base.copy()
    shifted = np.roll(base, shift=(3, 3), axis=(0, 1))
    noisy = np.clip(base.astype(np.int16) + rng.integers(-15, 15, base.shape), 0, 255).astype(np.uint8)
    rescaled = np.clip(base.astype(np.float64) * 0.5 + 40, 0, 255).astype(np.uint8)
    hot_spot = base.copy()
    hot_spot[5, 5] = 255
    unrelated = rng.integers(0, 255, base.shape, dtype=np.uint8)

    cases = {
        "identical": identical,
        "shifted 3px": shifted,
        "noisy": noisy,
        "rescaled brightness": rescaled,
        "one hot spot pixel": hot_spot,
        "unrelated random": unrelated,
    }

    header = list(compare(base, identical))
    print(f"{'case':<22}" + "".join(f"{k:>13}" for k in header))
    for name, arr in cases.items():
        m = compare(base, arr)
        print(f"{name:<22}" + "".join(f"{m[k]:13.4f}" for k in header))

    perfect = compare(base, identical)
    assert perfect["near_exact_fraction"] == 1.0
    assert perfect["mae"] == 0.0 and perfect["rmse"] == 0.0
    assert abs(perfect["ssim"] - 1.0) < 1e-9
    assert abs(perfect["pearson_corr"] - 1.0) < 1e-9
    assert perfect["hist_intersection"] == 1.0

    shifted_m = compare(base, shifted)
    assert shifted_m["near_exact_fraction"] < perfect["near_exact_fraction"]
    assert shifted_m["mae"] > perfect["mae"]
    assert shifted_m["ssim"] < perfect["ssim"]

    noisy_m = compare(base, noisy)
    assert noisy_m["near_exact_fraction"] < perfect["near_exact_fraction"]
    assert 0.85 < noisy_m["pearson_corr"] < 1.0, "small IID noise should stay highly correlated"

    rescaled_m = compare(base, rescaled)
    assert rescaled_m["pearson_corr"] > 0.99, "correlation should be near invariant to a linear rescale"
    assert rescaled_m["mae"] > 20, "mae should catch the brightness/contrast shift correlation misses"

    hot_spot_m = compare(base, hot_spot)
    assert hot_spot_m["mae"] < 1.0, "one pixel out of 1600 should barely move mae"
    assert hot_spot_m["rmse"] > 5 * hot_spot_m["mae"], "rmse should be pulled far more than mae by one hot spot"

    # a crop with a padded-out region should not be penalized there if masked
    padded = base.copy()
    padded[:20, :] = 0
    real_content = rng.integers(50, 200, base.shape, dtype=np.uint8)
    real_content[20:, :] = base[20:, :]
    padding_mask = np.zeros(base.shape, dtype=np.uint8)
    padding_mask[20:, :] = 255
    unmasked_m = compare(padded, real_content)
    masked_m = compare(padded, real_content, mask=padding_mask)
    assert masked_m["near_exact_fraction"] > unmasked_m["near_exact_fraction"]
    assert abs(masked_m["ssim"] - 1.0) < 1e-9, "identical content in the unmasked region should score perfectly"

    unrelated_m = compare(base, unrelated)
    assert unrelated_m["ssim"] < 0.3
    assert unrelated_m["pearson_corr"] < 0.3

    print("all self test assertions passed")
