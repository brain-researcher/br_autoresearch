"""One targeted metadata-label fixture; execute only in coordinated compute."""
import copy
import unittest

from repeated_identity_metadata import check_bound_identity
from test_repeated_image_sequences import bind, fixture


class TestRepeatedIdentityMetadata(unittest.TestCase):
    def test_bijection_and_ambiguous_presentation_numbering_remain_separate(self):
        metadata, source, frozen, channels, taps = fixture()
        # Neither global numbering nor local presentation semantics are assumed.
        for index, row in enumerate(source):
            row["presentationnumber"] = str(10000 + index)
            row["withinsequencenumber"] = "n/a"
        bound = bind((metadata, source, frozen, channels, taps))
        result = check_bound_identity(bound, source, budget_check=lambda: None)
        self.assertTrue(result["stable_bijection"])
        self.assertEqual(result["image_by_teststimnumber"], list(bound.image_ids))
        self.assertEqual(len(result["sequences"]), 12)
        self.assertFalse(result["presentation_numbering_used_as_admission_rule"])
        self.assertFalse(result["EEG_amplitudes_read"])

    def test_per_sequence_label_permutation_passes_old_binding_but_is_detected(self):
        metadata, source, frozen, channels, taps = fixture()
        changed = copy.deepcopy(source)
        # Keep 0..199 exactly once, all filenames, times and E3/E1 joins intact;
        # alter only which image the separate source index denotes in sequence2.
        changed[200]["teststimnumber"], changed[201]["teststimnumber"] = (
            changed[201]["teststimnumber"], changed[200]["teststimnumber"])
        bound = bind((metadata, changed, frozen, channels, taps))
        with self.assertRaisesRegex(ValueError, "unstable teststimnumber/image mapping"):
            check_bound_identity(bound, changed, budget_check=lambda: None)

    def test_missing_or_outside_documented_index_is_not_repaired(self):
        values = fixture()
        bound = bind(values)
        for index in ("n/a", "200", "-1"):
            changed = copy.deepcopy(values[1])
            changed[0]["teststimnumber"] = index
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "teststimnumber"):
                check_bound_identity(bound, changed, budget_check=lambda: None)


if __name__ == "__main__":
    unittest.main()
