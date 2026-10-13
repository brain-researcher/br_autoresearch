"""Focused checks for the observed flattened-test/unmapped-catch CSV schema."""
import unittest

from freeze_metadata_views import source_image_identity


class SourceIdentityTests(unittest.TestCase):
    def test_experimental_test_and_catch_identity_semantics(self):
        metadata = ["alligator/alligator_01b.jpg", "alligator/alligator_14n.jpg"]
        experiment = {"image_path": "images_fmri/alligator/alligator_01b.jpg", "condition": "exp",
                      "things_image_nr": "1", "things_category_nr": "13"}
        test = dict(experiment, image_path="images_test_fmri/alligator_14n.jpg", condition="test", things_image_nr="2")
        catch = dict(experiment, image_path="images_catch_fmri/catch001.jpg", condition="catch",
                     things_image_nr="", things_category_nr="")
        self.assertEqual(source_image_identity(experiment, metadata), (metadata[0], "alligator"))
        self.assertEqual(source_image_identity(test, metadata), (metadata[1], "alligator"))
        self.assertEqual(source_image_identity(catch, metadata), ("images_catch_fmri/catch001.jpg", None))

    def test_index_filename_or_partial_mapping_conflicts_are_rejected(self):
        metadata = ["alligator/alligator_01b.jpg", "alligator/alligator_14n.jpg"]
        test = {"image_path": "images_test_fmri/alligator_14n.jpg", "condition": "test",
                "things_image_nr": "1", "things_category_nr": "13"}
        with self.assertRaises(ValueError):
            source_image_identity(test, metadata)
        with self.assertRaises(ValueError):
            source_image_identity(dict(test, things_image_nr=""), metadata)


if __name__ == "__main__":
    unittest.main()
