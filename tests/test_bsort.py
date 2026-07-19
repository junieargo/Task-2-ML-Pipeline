import numpy as np

from bsort.preprocessing import get_color_class
from bsort.utils import class_id_to_label


def _solid_color_roi(hue: int, sat: int, val: int, size: int = 40) -> np.ndarray:
    """Build a small solid-color HSV image converted to BGR, for testing."""
    import cv2

    hsv = np.full((size, size, 3), (hue, sat, val), dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def test_light_blue_classified_as_class_0():
    roi = _solid_color_roi(hue=100, sat=200, val=200)
    assert get_color_class(roi) == 0


def test_dark_blue_classified_as_class_1():
    roi = _solid_color_roi(hue=100, sat=200, val=80)
    assert get_color_class(roi) == 1


def test_non_blue_classified_as_other():
    roi = _solid_color_roi(hue=30, sat=200, val=200)
    assert get_color_class(roi) == 2


def test_class_id_to_label_known_and_unknown():
    assert class_id_to_label(0) == "Light Blue"
    assert class_id_to_label(1) == "Dark Blue"
    assert class_id_to_label(2) == "Other"
    assert class_id_to_label(99) == "Unknown"
