"""Helper functions and constants shared across the bsort pipeline."""

# Maps the integer class IDs returned by preprocessing.get_color_class()
# to human-readable bottle cap color labels.
COLOR_CLASS_MAP = {
    0: "Light Blue",
    1: "Dark Blue",
    2: "Other",
}


def class_id_to_label(class_id: int) -> str:
    """Return the human-readable label for a given color class ID."""
    return COLOR_CLASS_MAP.get(class_id, "Unknown")
