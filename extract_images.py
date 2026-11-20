"""DEPRECATED — replaced by the refactored pipeline.

The old one-shot Arrow-to-JPG script has been replaced by a memory-efficient,
annotation-preserving pipeline.

Use instead::

    python scripts/convert_acne04.py

Which lives in:  src/acnevision/data/convert_arrow_to_jpg.py

This shim is kept only so old notes/commands do not silently do nothing.
"""

import sys

MESSAGE = (
    "extract_images.py is deprecated.\n"
    "Run the new pipeline instead:\n\n"
    "    python scripts/convert_acne04.py\n"
)

if __name__ == "__main__":
    sys.stderr.write(MESSAGE)
    raise SystemExit(1)
