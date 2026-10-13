"""Focused delivery/completeness checks; run only in coordinated compute."""
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
from PIL import Image, PngImagePlugin

from feature_extractors import require_compute
from image_pixel_worker import extract_family, pack_ids
from run_image_feature_pilot import PasswordNoticeError, build_pixel_pack, exact_members, private_password, select_images


class ImageFeaturePilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_compute()

    def fixture(self, directory):
        # Unequal spatial axes and explicit orientation metadata make a silent
        # EXIF transpose distinguishable from the registered stored raster.
        pixels = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        original = Image.fromarray(pixels)
        info = PngImagePlugin.PngInfo()
        info.add_text("provider_name", "forbidden-source-name")
        exif = Image.Exif()
        exif[274] = 6
        stream = io.BytesIO()
        original.save(stream, format="PNG", pnginfo=info, exif=exif)
        archive = directory/"source.zip"
        with zipfile.ZipFile(archive, "w") as target:
            target.writestr("root/concept_b/picture.png", stream.getvalue())
            target.writestr("root/concept_a/picture.png", stream.getvalue())
        rows = [{"opaque_image_id": "f"*32, "canonical_image": "concept_a/picture.png"},
                {"opaque_image_id": "0"*32, "canonical_image": "concept_b/picture.png"}]
        events = [{"participant": "dev", "canonical_image": row["canonical_image"],
                   "archive_member": "root/"+row["canonical_image"]} for row in rows]
        return pixels, archive, rows, events

    def test_exact_join_strips_metadata_preserves_uuid_order_and_stored_pixels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pixels, archive, rows, events = self.fixture(root)
            selected = select_images(rows, count=2, expected_total=2)
            members = exact_members(events, selected, expected_events=2, expected_unique=2)
            pack = root/"pixels.npz"
            result = build_pixel_pack(archive, members, pack, lambda: None)
            self.assertEqual(result["decoded_RGB_bytes"], 36)
            self.assertEqual(pack_ids(pack, 2), ["0"*32, "f"*32])
            self.assertNotIn(b"forbidden-source-name", pack.read_bytes())
            self.assertNotIn(b"concept_", pack.read_bytes())
            with np.load(pack, allow_pickle=False) as data:
                for key in data.files:
                    np.testing.assert_array_equal(data[key], pixels)
                    self.assertEqual(type(data[key]), np.ndarray)

    def test_ambiguous_join_invalid_uuid_and_pixel_overflow_refuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, archive, rows, events = self.fixture(root)
            with self.assertRaises(ValueError):
                select_images([rows[0], rows[0]], 2, 2)
            with self.assertRaises(ValueError):
                select_images([{**rows[0], "opaque_image_id": "named-concept"}], 1, 1)
            selected = select_images(rows, 2, 2)
            conflicting = events + [{**events[0], "archive_member": "other/"+events[0]["canonical_image"]}]
            with self.assertRaises(ValueError):
                exact_members(conflicting, selected, 3, 2)
            members = exact_members(events, selected, 2, 2)
            with self.assertRaises(ValueError):
                build_pixel_pack(archive, members, root/"too_small.npz", lambda: None, byte_cap=20)

    def test_all_images_required_and_pixel_generated_details_retained(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, archive, rows, events = self.fixture(root)
            selected = select_images(rows, 2, 2)
            pack = root/"pixels.npz"
            build_pixel_pack(archive, exact_members(events, selected, 2, 2), pack, lambda: None)

            class PixelCaption:
                def __init__(self, family, assets, device):
                    self.family = family

                def encode(self, ids, arrays):
                    self_ids.append(ids[0])
                    # Known answer depends only on the supplied raster.
                    return np.full((1, 512), arrays[0].mean(), dtype=np.float32), {
                        "captions": ["pixel-derived text"], "length_cap_without_SEP": [True],
                        "text": {"truncated": [False]}}

            self_ids = []
            report = extract_family("blip_caption_clip_text", pack, {}, root/"complete", lambda: None,
                                    extractor_factory=PixelCaption, expected_count=2)
            self.assertEqual(self_ids, ["0"*32, "f"*32, "0"*32])
            self.assertEqual(report["images"], 2)
            self.assertEqual(report["inferences"], 3)
            self.assertEqual(report["singleton_repeatability"]["maximum_absolute_error"], 0.)
            np.testing.assert_array_equal(np.load(root/"complete/features.npy"), np.full((2, 512), 8.5, dtype=np.float32))
            details = json.loads((root/"complete/image_details.json").read_text())
            self.assertEqual(details[1]["details"]["captions"], ["pixel-derived text"])
            self.assertTrue(details[0]["details"]["length_cap_without_SEP"][0])

            class Broken(PixelCaption):
                def encode(self, ids, arrays):
                    if ids[0] == "f"*32:
                        raise RuntimeError("Stop; no image exclusion or retry")
                    return super().encode(ids, arrays)

            with self.assertRaises(RuntimeError):
                extract_family("blip_caption_clip_text", pack, {}, root/"failed", lambda: None,
                               extractor_factory=Broken, expected_count=2)
            self.assertFalse((root/"failed/report.json").exists())

            class Drifting(PixelCaption):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.calls = 0

                def encode(self, ids, arrays):
                    self.calls += 1
                    values, detail = super().encode(ids, arrays)
                    if self.calls == 3:
                        values[0, 0] += 1.
                    return values, detail

            with self.assertRaises(ValueError):
                extract_family("blip_caption_clip_text", pack, {}, root/"drifting", lambda: None,
                               extractor_factory=Drifting, expected_count=2)
            self.assertFalse((root/"drifting/report.json").exists())

            class ChangingCaption(Drifting):
                def encode(self, ids, arrays):
                    self.calls += 1
                    values, detail = PixelCaption.encode(self, ids, arrays)
                    if self.calls == 3:
                        detail["captions"] = ["changed pixel-derived text"]
                    return values, detail

            with self.assertRaises(ValueError):
                extract_family("blip_caption_clip_text", pack, {}, root/"caption_drift", lambda: None,
                               extractor_factory=ChangingCaption, expected_count=2)
            self.assertFalse((root/"caption_drift/report.json").exists())

    def test_incomplete_nonfinite_feature_output_never_completes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, archive, rows, events = self.fixture(root)
            selected = select_images(rows, 2, 2)
            pack = root/"pixels.npz"
            build_pixel_pack(archive, exact_members(events, selected, 2, 2), pack, lambda: None)

            class Nonfinite:
                def __init__(self, *args, **kwargs):
                    pass

                def encode(self, ids, arrays):
                    return np.full((1, 851), np.nan, dtype=np.float32), {}

            with self.assertRaises(ValueError):
                extract_family("low_level_851", pack, {}, root/"nonfinite", lambda: None,
                               extractor_factory=Nonfinite, expected_count=2)
            self.assertFalse((root/"nonfinite/report.json").exists())

    def test_private_password_notice_never_guesses_prose_or_ambiguous_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            notice = Path(temporary)/"notice.txt"
            notice.write_text("Source terms remain in force.\nZIP password: fixture-only-42\n")
            self.assertEqual(private_password(notice), b"fixture-only-42")
            notice.write_text("Source terms remain in force.\nThe archive password for these original research images: fixture-only-42\nRetain attribution.\n")
            self.assertEqual(private_password(notice), b"fixture-only-42")
            notice.write_text("fixture-only-42\n")
            self.assertEqual(private_password(notice), b"fixture-only-42")
            for text in ("Read these source terms before using images.", "Password: one\nPassword: two\n",
                         "Password for one archive: same\nPassword for another archive: same\n",
                         "The password for these images: several words\n",
                         "Password for images: one\nPassword for another archive: several words\n"):
                notice.write_text(text)
                with self.assertRaises(PasswordNoticeError):
                    private_password(notice)


if __name__ == "__main__":
    unittest.main()
