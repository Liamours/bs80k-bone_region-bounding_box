"""Shared matchTemplate helpers for locating a crop inside a whole body image."""
import cv2
import numpy as np


def second_peak_score(result: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    """Best score elsewhere in the correlation surface, best peak's own footprint suppressed.

    A plain second-highest value would just be a pixel next to the best peak, correlation
    surfaces are smooth. Suppressing a window the size of the template around the best match
    removes that smooth peak, leaving the next genuinely different candidate location.
    """
    suppressed = result.copy()
    ry, rx = h // 2, w // 2
    suppressed[max(0, y - ry):y + ry + 1, max(0, x - rx):x + rx + 1] = -np.inf
    return float(suppressed.max()) if np.isfinite(suppressed).any() else float("nan")


def locate(crop: np.ndarray, search_region: np.ndarray, y_offset: int = 0, mask: np.ndarray | None = None) -> dict:
    """Find crop inside search_region, x/y returned in the full image's own coordinates.

    mask, if given, excludes crop pixels from the correlation, for a crop that is mostly
    near-zero background this stops that background from drowning out the real signal.
    """
    h, w = crop.shape[:2]
    result = cv2.matchTemplate(search_region, crop, cv2.TM_CCOEFF_NORMED, mask=mask)
    if mask is not None:
        # a masked search window that lands on a fully blank region divides by zero,
        # producing nan/inf entries that would otherwise look like a perfect match
        result = np.where(np.isfinite(result), result, -np.inf)
    _, score, _, (x, y_local) = cv2.minMaxLoc(result)
    margin = score - second_peak_score(result, x, y_local, w, h)
    return {"x": x, "y": y_local + y_offset, "score": score, "peak_margin": margin, "h": h, "w": w}


def background_mask(crop: np.ndarray, threshold: int = 2) -> np.ndarray:
    """255 where crop is above threshold, 0 where it is near-zero background."""
    return ((crop > threshold).astype(np.uint8)) * 255


def percentile_mask(crop: np.ndarray, percentile: float) -> np.ndarray:
    """255 where crop is above its own given percentile, relative to that one crop.

    A fixed threshold can end up with no pixels above it at all for a dim crop, see
    context/method.md. A percentile of the crop's own pixels always keeps some fraction of
    it, regardless of how bright or dim that particular crop is overall.
    """
    cutoff = np.percentile(crop, percentile)
    return ((crop > cutoff).astype(np.uint8)) * 255
