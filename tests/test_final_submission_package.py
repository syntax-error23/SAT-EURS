import os
import io
import zipfile
import hashlib
import numpy as np
from PIL import Image

def verify_submission():
    zip_path = "deploy/eurs_track1_sota_submission.zip"
    print("=" * 70)
    print("VERIFYING SUBMISSION PACKAGE:", zip_path)
    print("=" * 70)

    assert os.path.exists(zip_path), f"Archive {zip_path} not found!"
    size_mb = os.path.getsize(zip_path) / (1024 ** 2)
    print(f"Archive Size: {size_mb:.2f} MB")
    assert size_mb < 1024, f"File exceeds 1024 MB limit ({size_mb:.2f} MB)"

    h_sha = hashlib.sha256()
    with open(zip_path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h_sha.update(chunk)
    sha256 = h_sha.hexdigest()
    print(f"SHA256: {sha256}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        print(f"Total entries in archive: {len(names)}")
        assert len(names) == 2400, f"Expected exactly 2400 entries, found {len(names)}"

        restored_files = [n for n in names if n.startswith("restored/") and n.endswith(".jpg")]
        mask_files = [n for n in names if n.startswith("seg_masks/") and n.endswith(".png")]

        print(f"Restored JPEG count: {len(restored_files)} (expected 1200)")
        print(f"Seg Mask PNG count:   {len(mask_files)} (expected 1200)")
        assert len(restored_files) == 1200, f"Expected 1200 restored JPEGs, got {len(restored_files)}"
        assert len(mask_files) == 1200, f"Expected 1200 mask PNGs, got {len(mask_files)}"

        r_stems = {os.path.splitext(os.path.basename(n))[0] for n in restored_files}
        m_stems = {os.path.splitext(os.path.basename(n))[0] for n in mask_files}
        assert r_stems == m_stems, "Restored and mask stems do not match 1:1!"
        print("Restored and mask stems match 1:1: PASS")

        # Sample check 100 images
        sample_stems = sorted(list(r_stems))[::12]  # 100 images sampled evenly
        unique_classes = set()
        dim_mismatches = 0

        for stem in sample_stems:
            r_bytes = zf.read(f"restored/{stem}.jpg")
            m_bytes = zf.read(f"seg_masks/{stem}.png")

            r_img = Image.open(io.BytesIO(r_bytes))
            m_img = Image.open(io.BytesIO(m_bytes))

            if r_img.size != m_img.size:
                dim_mismatches += 1

            m_arr = np.array(m_img)
            assert m_arr.dtype == np.uint8, f"Mask dtype is {m_arr.dtype}, expected uint8"
            unique_classes.update(np.unique(m_arr).tolist())

        assert dim_mismatches == 0, f"Found {dim_mismatches} dimension mismatches!"
        print(f"Dimension match across 100 sample pairs: PASS")
        print(f"Classes observed across sample masks: {sorted(list(unique_classes))}")
        assert set(unique_classes).issubset({0, 1, 2, 3, 4}), f"Invalid classes found: {unique_classes}"
        assert set(unique_classes) == {0, 1, 2, 3, 4}, f"Not all classes detected: {unique_classes}"

    print("=" * 70)
    print("ALL SUBMISSION PACKAGE CONTRACT TESTS PASSED 100%!")
    print("=" * 70)

if __name__ == "__main__":
    verify_submission()
