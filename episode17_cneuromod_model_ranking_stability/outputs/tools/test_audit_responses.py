"""Three complete audit-delivery checks, for coordinated compute only.

All606 image identities,120 concepts,twelve blocks,eight groups and B/C/D
are present. Only the previously qualified copied-HDF5 header reader is
instrumented; no private file, empirical value, model or audit is accessed.
"""
from collections import Counter
from contextlib import nullcontext
import unittest
from unittest.mock import patch

import numpy as np

import audit_responses as core
import audit_prediction as audit
import endpoint_evaluator as endpoint
from test_development_responses import Source


def fixture():
    concepts = tuple(item for item in range(120) for _ in range(6 if item < 6 else 5))
    design = audit.AuditDesign(tuple(f"images/audit-{item:03}/image-{index:04}.jpg"
        for index, item in enumerate(concepts)), tuple(f"audit-{item:03}" for item in concepts),
        ("sealed_audit",) * 606, tuple(f"block{item % 12 + 1:02}" for item in concepts))
    counts = {person: [3 if image == 0 else 2 + (image + p) % 2 for image in range(606)]
              for p, person in enumerate(endpoint.SUBJECTS)}
    manifest = tuple(dict(provider_image_path=image, concept_name=design.concept_ids[index],
        role="sealed_audit", outer_fold="", development_block="", audit_block=design.block_ids[index],
        eligible_image="True", common_to_four="True",
        **{"retained_count_" + person: str(counts[person][index]) for person in endpoint.SUBJECTS})
        for index, image in enumerate(design.image_ids))
    row_maps, values, chronological_rows = {}, {}, {}
    for person in endpoint.SUBJECTS:
        rows, base, ordered_images, rank = [], [], [], 0
        for index, image in enumerate(design.image_ids):
            count, image_rows = counts[person][index], []
            for repeat in reversed(range(count)):
                local = len(rows)
                image_rows.append((repeat, local))
                rows.append(dict(participant=person, local_row_index_zero_based=str(local),
                    glmsingle_trial_index_zero_based=str(local), expected_provider_image_path=image,
                    concept_name=design.concept_ids[index], role="sealed_audit", outer_fold="",
                    development_block="", audit_block=design.block_ids[index], eligible_image="True",
                    ordinal_status="valid", retained_acquisition_rank=str(rank + repeat),
                    retained_repeat_rank=str(repeat), repeat_half=str(1 + repeat % 2)))
                # Ordered float64 [large,-large,1] distinguishes the required
                # chronology from reversed physical rows or float32 averaging.
                base.append((2. ** 55, -(2. ** 55), 1.)[repeat] if index == 0
                            else index / 8. + repeat * repeat / 4.)
            ordered_images.append(tuple(local for _, local in sorted(image_rows)))
            rank += count
        local = len(rows)
        rows.append(dict(participant=person, local_row_index_zero_based=str(local),
            glmsingle_trial_index_zero_based=str(local), expected_provider_image_path="images/audit-000/ineligible.jpg",
            concept_name=design.concept_ids[0], role="sealed_audit", outer_fold="", development_block="",
            audit_block=design.block_ids[0], eligible_image="False"))
        base.append(np.nan)
        row_maps[person] = tuple(rows)
        values[person] = np.asarray(base, dtype=np.float32)
        chronological_rows[person] = tuple(ordered_images)
    columns = {(person, roi): np.array([1025, 3, 513] if roi == endpoint.ROIS[0]
                                      else [514, 3, 1026, 1], dtype=np.int64)
               for person, roi in endpoint.GROUPS}
    return design, manifest, row_maps, columns, values, chronological_rows


class AuditResponsesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.design, cls.manifest, cls.rows, cls.columns, cls.values, cls.ordered = fixture()

    @classmethod
    def tearDownClass(cls):
        for name in ("design", "manifest", "rows", "columns", "values", "ordered"):
            delattr(cls, name)

    def sources(self, *, bad_header=False, bad_value=False):
        headers, reads = [], []
        sources = {(person, stage): Source(person, stage, self.values[person], headers, reads,
                   bad=bad_value and person == endpoint.SUBJECTS[0] and stage == "B")
                   for person in endpoint.SUBJECTS for stage in endpoint.STAGES}

        def header(source, shape, dtype):
            self.assertEqual(shape, (len(self.rows[source.person]), 1, 1,
                                    endpoint.support.PARTICIPANT_COLUMNS[source.person]))
            self.assertEqual(dtype, "<f4")
            headers.append((source.person, source.stage))
            if bad_header and len(headers) == 12:
                raise ValueError("twelfth copied header refused")
            return source
        return sources, header, headers, reads

    def call(self, sources, check=lambda event: None, **changes):
        return core.build_audit_responses(changes.get("design", self.design),
            changes.get("image_manifest", self.manifest), changes.get("row_maps", self.rows),
            sources, changes.get("frozen_columns", self.columns), resource_check=check)

    def test_full_audit_identity_ordered_means_and_single_shared_stripe_reads(self):
        self.assertEqual((audit.AUDIT_IMAGES, len(set(self.design.concept_ids))), (606, 120))
        self.assertEqual(set(self.design.roles), {"sealed_audit"})
        self.assertEqual(Counter(dict(zip(self.design.concept_ids, self.design.block_ids)).values()),
                         {f"block{number:02}": 10 for number in range(1, 13)})
        sources, header, headers, reads = self.sources()
        with patch.object(core.copied, "output_dataset", new=header), \
                patch.object(audit, "apply_frozen_audit", side_effect=AssertionError("delivery cannot predict/score")), \
                patch.object(core.calibration, "qualify_variance", side_effect=AssertionError("no audit prescreen")):
            result = self.call(sources)
        self.assertEqual(set(result), set(endpoint.GROUPS))
        self.assertEqual(len(headers), 12)
        expected_reads = {(person, stage, start, start + 512)
                          for person in endpoint.SUBJECTS for stage in endpoint.STAGES for start in (0, 512, 1024)}
        self.assertEqual(set(reads), expected_reads)
        self.assertEqual(len(reads), len(expected_reads))
        for group in endpoint.GROUPS:
            person, _ = group
            panel = result[group]
            self.assertIsInstance(panel, audit.AuditResponsePanel)
            self.assertIs(panel.rows, self.design)
            np.testing.assert_array_equal(panel.column_ids, self.columns[group])
            self.assertEqual(set(panel.stages), set(endpoint.STAGES))
            for stage in endpoint.STAGES:
                expected = np.empty((606, len(self.columns[group])))
                for image, rows in enumerate(self.ordered[person]):
                    for position, column in enumerate(self.columns[group]):
                        offset = np.float32((int(column) % 17) / 8.
                            + endpoint.SUBJECTS.index(person) / 4. + endpoint.STAGES.index(stage) / 2.)
                        expected[image, position] = sum(float(np.float32(self.values[person][row] + offset))
                                                        for row in rows) / len(rows)
                actual = panel.stages[stage]
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(actual.dtype, np.dtype(np.float64))
                with self.assertRaises(ValueError):
                    actual.setflags(write=True)
            with self.assertRaises(ValueError):
                panel.column_ids.setflags(write=True)
            with self.assertRaises(TypeError):
                panel.stages["B"] = None
        self.assertEqual(result[endpoint.GROUPS[0]].stages["B"][0, 1], 1.375 / 3.)
        with self.assertRaises(TypeError):
            result[endpoint.GROUPS[0]] = None

    def test_role_repeat_row_axis_and_last_header_refuse_before_values(self):
        first, person = self.manifest[0], endpoint.SUBJECTS[0]
        variants = [dict(image_manifest=tuple(reversed(self.manifest))),
                    dict(image_manifest=self.manifest[:-1])]
        for change in ({"role": "development"}, {"role": "support_calibration"}, {"outer_fold": "fold01"},
                       {"development_block": "block01"}, {"audit_block": "block02"},
                       {"retained_count_sub-01": "2"}, {"eligible_image": "False"}):
            variants.append(dict(image_manifest=({**first, **change}, *self.manifest[1:])))
        for change in ({"role": "development"}, {"local_row_index_zero_based": "1"},
                       {"glmsingle_trial_index_zero_based": self.rows[person][1]["glmsingle_trial_index_zero_based"]},
                       {"audit_block": "block02"}, {"retained_repeat_rank": "0"}, {"repeat_half": "2"},
                       {"retained_acquisition_rank": self.rows[person][1]["retained_acquisition_rank"]}):
            variants.append(dict(row_maps={**self.rows, person: ({**self.rows[person][0], **change}, *self.rows[person][1:])}))
        variants.extend((dict(frozen_columns={**self.columns, endpoint.GROUPS[0]: np.array([3, 3])}),
                         dict(frozen_columns={**self.columns, endpoint.GROUPS[0]:
                             np.array([endpoint.support.PARTICIPANT_COLUMNS[person]])})))
        with patch.object(core.copied, "output_dataset", side_effect=AssertionError("header before complete metadata")):
            for index, change in enumerate(variants):
                with self.subTest(index=index), self.assertRaises(ValueError):
                    self.call(self.sources()[0], **change)
        sources, header, headers, reads = self.sources(bad_header=True)
        with patch.object(core.copied, "output_dataset", new=header), self.assertRaisesRegex(ValueError, "twelfth copied header"):
            self.call(sources)
        self.assertEqual(len(headers), 12)
        self.assertEqual(reads, [])

    def test_nonfinite_and_resource_refusal_prevent_partial_delivery(self):
        sources, header, _, reads = self.sources(bad_value=True)
        with patch.object(core.copied, "output_dataset", new=header), \
                self.assertRaisesRegex(ValueError, "nonfinite_frozen_response_repeat"):
            self.call(sources)
        self.assertEqual(reads, [(endpoint.SUBJECTS[0], "B", 0, 512)])
        for stop in ("begin", "allocate_image_means", "read_stripe", "average_repeats", "freeze_panel"):
            sources, header, _, reads = self.sources()

            def check(event):
                if event.operation == stop:
                    if stop == "read_stripe":
                        self.assertEqual(event.byte_count, len(self.rows[endpoint.SUBJECTS[0]]) * 512 * 4)
                    raise TimeoutError("fixed audit delivery cancellation")

            mean_guard = (patch.object(core.calibration, "center", side_effect=AssertionError("mean before admission"))
                          if stop == "average_repeats" else nullcontext())
            with self.subTest(stop=stop), patch.object(core.copied, "output_dataset", new=header), mean_guard, \
                    self.assertRaisesRegex(TimeoutError, "fixed audit delivery cancellation"):
                self.call(sources, check)
            self.assertEqual(len(reads), 3 if stop == "freeze_panel" else 1 if stop == "average_repeats" else 0)


if __name__ == "__main__":
    unittest.main()
