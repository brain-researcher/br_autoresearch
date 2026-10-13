"""Source-free fixtures for adopted eligibility and closed role boundaries."""

import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import qualify_source_roles as stage
import reconcile_released_events as core


def table(fields, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def taxonomy_payloads():
    categories = [{"uniqueID": "concept_a", "Word": "a", "animal": "1", "tool": "1"},
                  {"uniqueID": "concept_b", "Word": "b", "animal": "0", "tool": "0"}]
    concepts = [{"Word": "a", "uniqueID": "concept_a", "unused_rating": "CLOSED_UNUSED_RATING_A"},
                {"Word": "b", "uniqueID": "concept_b", "unused_rating": "CLOSED_UNUSED_RATING_B"}]
    return categories, concepts


def parsed_taxonomy():
    cats, concepts = taxonomy_payloads()
    return stage.parse_taxonomy(table(list(cats[0]), cats), table(list(concepts[0]), concepts), 2, 2)


def joined_image(name="image_a.jpg", cid=1, counts=(2, 2, 2, 2)):
    rows = []
    concept_name = {1: "concept_a", 2: "concept_b"}.get(cid, f"concept_{cid}")
    for subject, count in zip(stage.SUBJECTS, counts):
        for repeat in range(count):
            rows.append({"participant": subject, "session": "1", "run": "1", "run_order": "0",
                         "trial_in_run_zero_based": str(repeat), "glmsingle_trial_index_zero_based": str(repeat),
                         "expected_provider_image_path": f"images_fmri/{concept_name}/{name}",
                         "source_image_basename": name, "image_path_suffix_matches_expected": "True",
                         "things_category_nr": str(cid), "provider_category_nr": str(cid), "provider_exemplar_nr": "1",
                         "onset_seconds": str(3 + repeat * 4.47), "duration_seconds": "2.98",
                         "onset_TR_after_two_volume_removal": str(core.provider_onset_tr(str(3 + repeat * 4.47))),
                         "source_trial_number": str(repeat + 1), "source_order": str(repeat),
                         "source_repetition": str(repeat + 1), "retained_image_presentation_number": str(repeat + 1),
                         "retained_previous_same_image_index_gap": "" if repeat == 0 else "1",
                         "atypical": "False", "not_for_memory": "False", "delay_days": "" if repeat == 0 else "0",
                         "delay_seconds": "" if repeat == 0 else "4.47", "trials_since_lastrep": "" if repeat == 0 else "1",
                         "identity_and_onset_match": "True", "concept_name": concept_name, "condition": "exp"})
    return rows


def matching_inputs():
    joined = joined_image(counts=(2, 0, 0, 0))
    originals = []
    events = []
    for row in joined:
        originals.append({**{field: row[field] for field in stage.MATCH_FIELDS},
                          "provider_image_path": row["expected_provider_image_path"],
                          "image_name": row["source_image_basename"], "concept_name": row["concept_name"],
                          "condition": row["condition"], "image_presentation_number": row["retained_image_presentation_number"],
                          "previous_same_image_index_gap": row["retained_previous_same_image_index_gap"]})
        events.append({field: row[field] for field in core.OUTPUT_FIELDS})
    runs = [{"participant": "sub-01", "session": "1", "run": "1", "run_order": "0",
             "presentations": "2", "first_glmsingle_trial_index_zero_based": "0"}]
    return originals, events, runs


class EligibilityTests(unittest.TestCase):
    def test_minimum_two_every_person_and_one_image_per_concept(self):
        labels, taxonomy = parsed_taxonomy()
        rows = joined_image("fails.jpg", counts=(2, 2, 2, 1)) + joined_image("passes.jpg", counts=(2, 3, 4, 2))
        images, concepts, unknowns = stage.qualify(rows, taxonomy, labels)
        by_name = {row["image_name"]: row for row in images}
        self.assertFalse(by_name["fails.jpg"]["eligible_image"])
        self.assertIn("fewer_than_two_retained_presentations:sub-06", by_name["fails.jpg"]["failure_reasons"])
        self.assertTrue(by_name["passes.jpg"]["eligible_image"])
        self.assertEqual(concepts[0]["qualified_image_count"], 1)
        self.assertTrue(concepts[0]["eligible_concept"])
        self.assertEqual(unknowns, [])

    def test_missing_participant_is_explicit_and_does_not_create_common_image(self):
        labels, taxonomy = parsed_taxonomy()
        images, concepts, _ = stage.qualify(joined_image(counts=(3, 3, 0, 3)), taxonomy, labels)
        self.assertFalse(images[0]["common_to_four"])
        self.assertIn("absent:sub-03", images[0]["failure_reasons"])
        self.assertFalse(concepts[0]["eligible_concept"])

    def test_nonexperimental_row_is_changed_source_scope_before_taxonomy_lookup(self):
        labels, taxonomy = parsed_taxonomy()
        rows = joined_image()
        rows[0]["condition"] = "catch"
        rows[0]["things_category_nr"] = ""
        with self.assertRaises(core.Refusal) as result:
            stage.qualify(rows, taxonomy, labels)
        self.assertEqual(result.exception.detail["code"], "retained_source_condition_scope_changed")
        previous, events, runs = matching_inputs()
        previous[0]["condition"] = "test"
        with self.assertRaises(core.Refusal) as result:
            stage.check_presentations(previous, events, runs)
        self.assertEqual(result.exception.detail["code"], "retained_source_condition_scope_changed")

    def test_schedule_flags_and_ordinal_gaps_never_exclude(self):
        labels, taxonomy = parsed_taxonomy()
        rows = joined_image()
        for row in rows:
            row["source_repetition"] = "2" if row["source_order"] == "0" else "5"
            row["atypical"] = row["not_for_memory"] = "True"
        images, concepts, unknowns = stage.qualify(rows, taxonomy, labels)
        self.assertTrue(images[0]["eligible_image"])
        self.assertEqual(unknowns, [])
        with tempfile.TemporaryDirectory() as temporary:
            stage.publish(Path(temporary), rows, images, concepts, taxonomy, labels)
            with (Path(temporary) / "presentation_qualification.csv").open() as stream:
                projected = list(csv.DictReader(stream))
            self.assertEqual([r["source_repetition"] for r in projected], [r["source_repetition"] for r in rows])
            self.assertTrue(all(r["not_for_memory"] == "True" for r in projected))
            self.assertNotIn("repeat_half", projected[0])

    def test_nonunique_ordinal_reports_unknown_without_false_half_assignment(self):
        labels, taxonomy = parsed_taxonomy()
        rows = joined_image()
        rows[1]["source_repetition"] = rows[0]["source_repetition"]
        images, _, unknowns = stage.qualify(rows, taxonomy, labels)
        self.assertTrue(images[0]["eligible_image"])
        self.assertFalse(images[0]["source_ordinals_positive_unique_in_each_participant"])
        self.assertEqual(unknowns[0]["participant"], "sub-01")

    def test_all_coordinates_reconciled_and_wrong_image_or_missing_row_stops(self):
        previous, events, runs = matching_inputs()
        self.assertEqual(len(stage.check_presentations(previous, events, runs)), 2)
        bad = copy.deepcopy(events)
        bad[1]["things_category_nr"] = "2"
        with self.assertRaises(core.Refusal):
            stage.check_presentations(previous, bad, runs)
        with self.assertRaises(core.Refusal):
            stage.check_presentations(previous, events[:1], runs)

    def test_720_failure_cannot_publish_completion_or_roles(self):
        labels, taxonomy = parsed_taxonomy()
        with tempfile.TemporaryDirectory() as temporary, patch.object(stage, "load_sources", return_value=joined_image()), \
                patch.object(stage, "download_taxonomy", return_value=b"fixture"), \
                patch.object(stage, "parse_taxonomy", return_value=(labels, taxonomy)):
            out = Path(temporary)
            self.assertEqual(stage.run_attempt(out), 2)
            report = json.loads((out / "qualification_report.json").read_text())
            self.assertEqual(report["qualified_concept_count"], 1)
            self.assertEqual(report["errors"][0]["code"], "eligible_concept_count_not_720_no_rescue")
            for flag in ("eligibility_complete", "source_roles_complete", "roles_assigned",
                         "folds_created", "repeat_halves_assigned", "actual_acquisition_order_verified"):
                self.assertFalse(report[flag])
            self.assertTrue((out / "image_qualification.csv").exists())

    def test_partial_output_failure_never_completes(self):
        labels, taxonomy = parsed_taxonomy()
        with tempfile.TemporaryDirectory() as temporary, patch.object(stage, "load_sources", return_value=joined_image()), \
                patch.object(stage, "download_taxonomy", return_value=b"fixture"), \
                patch.object(stage, "parse_taxonomy", return_value=(labels, taxonomy)), \
                patch.object(stage, "publish", side_effect=OSError("do not expose filesystem or source detail")):
            out = Path(temporary)
            self.assertEqual(stage.run_attempt(out), 2)
            report = (out / "qualification_report.json").read_text()
            self.assertNotIn("do not expose", report)
            self.assertFalse(json.loads(report)["eligibility_complete"])

    def test_deadline_during_completion_clears_flags_and_exits_failure(self):
        labels, taxonomy = parsed_taxonomy()
        rows = joined_image()
        images, concepts, unknowns = stage.qualify(rows, taxonomy, labels)
        def interrupted_completion(report):
            report["eligibility_complete"] = True
            report["eligible_720_concept_universe_verified"] = True
            raise core.Refusal("attempt_deadline")
        with tempfile.TemporaryDirectory() as temporary, patch.object(stage, "load_sources", return_value=rows), \
                patch.object(stage, "download_taxonomy", return_value=b"fixture"), \
                patch.object(stage, "parse_taxonomy", return_value=(labels, taxonomy)), \
                patch.object(stage, "qualify", return_value=(images, concepts * 720, unknowns)), \
                patch.object(stage, "publish"), patch.object(stage, "complete_eligibility", side_effect=interrupted_completion):
            out = Path(temporary)
            self.assertEqual(stage.run_attempt(out), 2)
            report = json.loads((out / "qualification_report.json").read_text())
            self.assertEqual(report["status"], "stopped")
            self.assertFalse(report["eligibility_complete"])
            self.assertFalse(report["eligible_720_concept_universe_verified"])


class TaxonomyTests(unittest.TestCase):
    def test_pinned_annex_pointer_identity(self):
        for spec in stage.TAXONOMY:
            stage.verify_pinned_pointer(spec)
        changed = {**stage.TAXONOMY[0], "directory": "fK/XX"}
        with self.assertRaises(core.Refusal):
            stage.verify_pinned_pointer(changed)

    def test_one_based_bridge_multilabel_and_rating_projection(self):
        labels, taxonomy = parsed_taxonomy()
        self.assertEqual(taxonomy[1]["uniqueID"], "concept_a")
        self.assertEqual(taxonomy[2]["uniqueID"], "concept_b")
        self.assertEqual([taxonomy[1][label] for label in labels], [1, 1])
        self.assertEqual([taxonomy[2][label] for label in labels], [0, 0])
        self.assertNotIn("CLOSED_UNUSED", repr(taxonomy))
        images, concepts, _ = stage.qualify(joined_image(cid=2), taxonomy, labels)
        self.assertEqual(concepts[0]["uniqueID"], "concept_b")
        self.assertTrue(concepts[0]["taxonomy_unmapped"])
        self.assertTrue(images[0]["eligible_image"])

    def test_unknown_or_duplicate_id_and_graded_membership_refused(self):
        for variant in ("unknown", "duplicate_concept", "duplicate_category", "graded"):
            cats, concepts = taxonomy_payloads()
            if variant == "unknown":
                concepts[1]["uniqueID"] = "unknown_concept"
            elif variant == "duplicate_concept":
                concepts[1]["uniqueID"] = "concept_a"
            elif variant == "duplicate_category":
                cats[1]["uniqueID"] = "concept_a"
            else:
                cats[0]["animal"] = "0.5"
            with self.assertRaises(core.Refusal):
                stage.parse_taxonomy(table(list(cats[0]), cats), table(list(concepts[0]), concepts), 2, 2)

    def test_concept_table_permutations_within_and_outside_selected_names(self):
        for swap in ((0, 1), (2, 3)):
            cats, concepts = taxonomy_payloads()
            for name in ("outside_c", "outside_d"):
                cats.append({"uniqueID": name, "Word": name, "animal": "0", "tool": "1"})
                concepts.append({"Word": name, "uniqueID": name, "unused_rating": "CLOSED_UNUSED"})
            a, b = swap
            concepts[a], concepts[b] = concepts[b], concepts[a]
            labels, taxonomy = stage.parse_taxonomy(table(list(cats[0]), cats), table(list(concepts[0]), concepts), 4, 2)
            images, qualified_concepts, _ = stage.qualify(joined_image(), taxonomy, labels)
            self.assertTrue(images[0]["eligible_image"])
            self.assertEqual(qualified_concepts[0]["things_category_nr"], 1)
            self.assertEqual(qualified_concepts[0]["uniqueID"], "concept_a")

    def test_category_table_permutation_must_preserve_selected_numeric_anchor(self):
        cats, concepts = taxonomy_payloads()
        for name in ("outside_c", "outside_d"):
            cats.append({"uniqueID": name, "Word": name, "animal": "0", "tool": "1"})
            concepts.append({"Word": name, "uniqueID": name, "unused_rating": "CLOSED_UNUSED"})
        cats[2], cats[3] = cats[3], cats[2]
        labels, taxonomy = stage.parse_taxonomy(table(list(cats[0]), cats), table(list(concepts[0]), concepts), 4, 2)
        self.assertTrue(stage.qualify(joined_image(), taxonomy, labels)[0][0]["eligible_image"])
        cats[0], cats[1] = cats[1], cats[0]
        labels, taxonomy = stage.parse_taxonomy(table(list(cats[0]), cats), table(list(concepts[0]), concepts), 4, 2)
        with self.assertRaises(core.Refusal) as result:
            stage.qualify(joined_image(), taxonomy, labels)
        self.assertEqual(result.exception.detail["code"], "selected_source_concept_taxonomy_anchor_mismatch")

    def test_unknown_source_CID_or_mismatched_source_name_or_folder_refused(self):
        labels, taxonomy = parsed_taxonomy()
        for variant in ("unknown_CID", "wrong_name", "wrong_folder"):
            rows = joined_image()
            if variant == "unknown_CID":
                rows[0]["things_category_nr"] = "999"
            elif variant == "wrong_name":
                rows[0]["concept_name"] = "concept_b"
            else:
                rows[0]["expected_provider_image_path"] = "images_fmri/concept_b/image_a.jpg"
            with self.assertRaises(core.Refusal):
                stage.qualify(rows, taxonomy, labels)

    def test_oversized_local_input_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "metadata.csv"
            source.write_bytes(b"12345")
            with self.assertRaises(core.Refusal):
                stage.bounded_read(source, cap=4)

    def test_wrong_length_refuses_before_body_and_digest_refuses_after_bounded_read(self):
        class Response:
            status = 200
            def __init__(self, declared):
                self.headers = {"Content-Length": declared}
                self.read_sizes = []
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def read(self, limit):
                self.read_sizes.append(limit)
                return b"bad"
        spec = {"directory": "aa/BB", "bytes": 3, "md5": hashlib.md5(b"yes").hexdigest()}
        wrong_length = Response("4")
        with patch.object(stage, "build_opener") as opener:
            opener.return_value.open.return_value = wrong_length
            with self.assertRaises(core.Refusal):
                stage.download_taxonomy(spec, time.monotonic() + 20, {})
        self.assertEqual(wrong_length.read_sizes, [])
        wrong_digest = Response("3")
        with patch.object(stage, "build_opener") as opener:
            opener.return_value.open.return_value = wrong_digest
            with self.assertRaises(core.Refusal) as result:
                stage.download_taxonomy(spec, time.monotonic() + 20, {})
        self.assertEqual(result.exception.detail["code"], "taxonomy_annex_md5_mismatch")
        self.assertEqual(wrong_digest.read_sizes, [4])


if __name__ == "__main__":
    unittest.main()
