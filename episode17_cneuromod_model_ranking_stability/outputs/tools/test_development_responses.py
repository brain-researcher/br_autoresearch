"""Complete development delivery fixtures; run only in coordinated compute.

All2423 images,480 concepts, six folds,24 blocks, eight groups and B/C/D
are retained. Only the already-qualified copied-HDF5 header helper is replaced
by an instrumented source, avoiding enormous fabricated structural-axis files.
No actual source, fitting, variance qualification or audit execution occurs.
"""
import unittest
from unittest.mock import patch

import numpy as np

import development_responses as core
import endpoint_evaluator as endpoint
import nested_prediction as nested


def fixture():
    concepts = tuple(item for item in range(480) for _ in range(6 if item < 23 else 5))
    design = nested.DevelopmentDesign(tuple(f"images/concept-{item:03}/image-{index:04}.jpg"
        for index, item in enumerate(concepts)), tuple(f"concept-{item:03}" for item in concepts),
        ("development",) * 2423, tuple(f"fold{item // 80 + 1:02}" for item in concepts),
        tuple(f"block{item % 24 + 1:02}" for item in concepts))
    manifest, row_maps, values, chronological_rows = [], {}, {}, {}
    counts = {person: [3 if image == 0 else 2 + (image + person_index) % 2 for image in range(2423)]
              for person_index, person in enumerate(endpoint.SUBJECTS)}
    for index, image in enumerate(design.image_ids):
        manifest.append(dict(provider_image_path=image, concept_name=design.concept_ids[index],
            role="development", outer_fold=design.fold_ids[index], development_block=design.block_ids[index],
            audit_block="", eligible_image="True", common_to_four="True",
            **{"retained_count_" + person: str(counts[person][index]) for person in endpoint.SUBJECTS}))
    for person in endpoint.SUBJECTS:
        rows, base, ordered_images, rank = [], [], [], 0
        for index, image in enumerate(design.image_ids):
            count = counts[person][index]
            image_rows = []
            # Physical row order is reversed within each image, requiring the
            # frozen chronology join rather than a direct slice mean.
            for repeat in reversed(range(count)):
                local = len(rows)
                image_rows.append((repeat, local))
                rows.append(dict(participant=person, local_row_index_zero_based=str(local),
                    glmsingle_trial_index_zero_based=str(local), expected_provider_image_path=image,
                    concept_name=design.concept_ids[index], role="development", outer_fold=design.fold_ids[index],
                    development_block=design.block_ids[index], audit_block="", eligible_image="True",
                    ordinal_status="valid", retained_acquisition_rank=str(rank + repeat),
                    retained_repeat_rank=str(repeat), repeat_half=str(1 + repeat % 2)))
                # Chronological[huge,-huge,1] distinguishes float64 averaging
                # in the admitted order from physical-row or float32 means.
                base.append((2. ** 55, -(2. ** 55), 1.)[repeat] if index == 0
                            else index / 8. + repeat * repeat / 4.)
            ordered_images.append(tuple(local for _, local in sorted(image_rows)))
            rank += count
        local = len(rows)
        rows.append(dict(participant=person, local_row_index_zero_based=str(local),
            glmsingle_trial_index_zero_based=str(local), expected_provider_image_path="images/concept-000/ineligible.jpg",
            concept_name=design.concept_ids[0], role="development", outer_fold=design.fold_ids[0],
            development_block=design.block_ids[0], audit_block="", eligible_image="False"))
        base.append(np.nan)  # Metadata-ineligible data cannot enter an image mean.
        row_maps[person], values[person] = tuple(rows), np.asarray(base, dtype=np.float32)
        chronological_rows[person] = tuple(ordered_images)
    columns = {(person, roi): np.array([1025, 3, 513] if roi == endpoint.ROIS[0]
                                      else [514, 3, 1026, 1], dtype=np.int64)
               for person, roi in endpoint.GROUPS}
    return design, tuple(manifest), row_maps, columns, values, chronological_rows


class Source:
    def __init__(self, person, stage, values, headers, reads, bad=False):
        self.person, self.stage, self.values = person, stage, values
        self.headers, self.reads, self.bad = headers, reads, bad

    def __getitem__(self, key):
        if len(self.headers) != 12:
            raise AssertionError("value read before all twelve headers")
        rows, one, two, columns = key
        if rows != slice(None) or one != 0 or two != 0:
            raise AssertionError("unexpected physical stripe traversal")
        start, stop = columns.start, columns.stop
        self.reads.append((self.person, self.stage, start, stop))
        offset = (np.arange(start, stop, dtype=np.float32) % 17) / np.float32(8)
        offset += np.float32(endpoint.SUBJECTS.index(self.person) / 4. + endpoint.STAGES.index(self.stage) / 2.)
        block = self.values[:, None] + offset[None, :]
        if self.bad and start == 0:
            block[0, 3] = np.nan
        return block


class DevelopmentResponsesTests(unittest.TestCase):
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
        return core.build_development_responses(changes.get("design", self.design),
            changes.get("image_manifest", self.manifest), changes.get("row_maps", self.rows),
            sources, changes.get("frozen_columns", self.columns), resource_check=check)

    def test_all_images_native_columns_equal_repeat_means_and_shared_stripes(self):
        sources, header, headers, reads = self.sources()
        with patch.object(core.copied, "output_dataset", new=header), \
                patch.object(nested, "run_nested_prediction", side_effect=AssertionError("delivery cannot fit")), \
                patch.object(core.calibration, "qualify_variance", side_effect=AssertionError("no new qualification")):
            result = self.call(sources)
        self.assertEqual(set(result), set(endpoint.GROUPS))
        self.assertEqual(len(headers), 12)
        expected_reads = {(person, stage, start, start + 512)
                          for person in endpoint.SUBJECTS for stage in endpoint.STAGES for start in (0, 512, 1024)}
        self.assertEqual(set(reads), expected_reads)
        self.assertEqual(len(reads), len(expected_reads))
        for group in endpoint.GROUPS:
            person, _ = group
            self.assertIs(result[group].rows, self.design)
            np.testing.assert_array_equal(result[group].column_ids, self.columns[group])
            self.assertEqual(set(result[group].stages), set(endpoint.STAGES))
            for stage in endpoint.STAGES:
                expected = np.empty((2423, len(self.columns[group])))
                for image, rows in enumerate(self.ordered[person]):
                    for position, column in enumerate(self.columns[group]):
                        offset = np.float32((int(column) % 17) / 8.
                            + endpoint.SUBJECTS.index(person) / 4. + endpoint.STAGES.index(stage) / 2.)
                        # Independent ordered scalar reference, with the
                        # source's delivered float32 rounding retained.
                        expected[image, position] = sum(float(np.float32(self.values[person][row] + offset))
                                                        for row in rows) / len(rows)
                actual = result[group].stages[stage]
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(actual.dtype, np.dtype(np.float64))
                self.assertFalse(actual.flags.writeable)
                with self.assertRaises(ValueError):
                    actual.setflags(write=True)
            with self.assertRaises(ValueError):
                result[group].column_ids.setflags(write=True)
        with self.assertRaises(TypeError):
            result[endpoint.GROUPS[0]].stages["B"] = None
        # Image0's chronological sum differs from its physical-row sum.
        self.assertEqual(result[endpoint.GROUPS[0]].stages["B"][0, 1], float(np.float32(1.375)) / 3.)
        independent = result[endpoint.GROUPS[0]].stages["B"].copy()
        sources[endpoint.SUBJECTS[0], "B"].values = np.zeros_like(self.values[endpoint.SUBJECTS[0]])
        np.testing.assert_array_equal(result[endpoint.GROUPS[0]].stages["B"], independent)

    def test_metadata_counts_chronology_roles_and_original_axes_refuse_before_headers(self):
        variants = []
        first = self.manifest[0]
        variants.append(dict(image_manifest=({**first, "retained_count_sub-01": "2"}, *self.manifest[1:])))
        variants.append(dict(image_manifest=({**first, "eligible_image": "False"}, *self.manifest[1:])))
        variants.append(dict(image_manifest=tuple(reversed(self.manifest))))
        variants.append(dict(image_manifest=({**first, "outer_fold": 1}, *self.manifest[1:])))
        person = endpoint.SUBJECTS[0]
        for change in ({"role": "sealed_audit"}, {"local_row_index_zero_based": "1"},
                       {"retained_repeat_rank": "0"}, {"repeat_half": "2"},
                       {"retained_acquisition_rank": self.rows[person][1]["retained_acquisition_rank"]}):
            altered = ({**self.rows[person][0], **change}, *self.rows[person][1:])
            variants.append(dict(row_maps={**self.rows, person: altered}))
        variants.append(dict(frozen_columns={**self.columns, endpoint.GROUPS[0]: np.array([3, 3])}))
        variants.append(dict(frozen_columns={**self.columns, endpoint.GROUPS[0]:
            np.array([endpoint.support.PARTICIPANT_COLUMNS[person]])}))
        with patch.object(core.copied, "output_dataset", side_effect=AssertionError("header before complete metadata")):
            for index, change in enumerate(variants):
                with self.subTest(index=index), self.assertRaises(ValueError):
                    self.call(self.sources()[0], **change)

    def test_last_header_failure_prevents_all_value_reads(self):
        sources, header, headers, reads = self.sources(bad_header=True)
        with patch.object(core.copied, "output_dataset", new=header):
            with self.assertRaisesRegex(ValueError, "twelfth copied header refused"):
                self.call(sources)
        self.assertEqual(len(headers), 12)
        self.assertEqual(reads, [])

    def test_nonfinite_frozen_repeat_refuses_whole_delivery_without_pruning(self):
        sources, header, _, reads = self.sources(bad_value=True)
        with patch.object(core.copied, "output_dataset", new=header):
            with self.assertRaisesRegex(ValueError, "nonfinite_frozen_response_repeat"):
                self.call(sources)
        self.assertEqual(reads, [(endpoint.SUBJECTS[0], "B", 0, 512)])

    def test_resource_refusal_precedes_metadata_read_and_mean_work(self):
        def refuse_begin(event):
            if event.operation == "begin":
                raise RuntimeError("refused begin")
        with patch.object(core, "_bind_metadata", side_effect=AssertionError("metadata before admission")):
            with self.assertRaisesRegex(RuntimeError, "refused begin"):
                self.call(self.sources()[0], refuse_begin)
        sources, header, _, reads = self.sources()
        def refuse_read(event):
            if event.operation == "read_stripe":
                self.assertEqual(event.byte_count, len(self.rows[endpoint.SUBJECTS[0]]) * 512 * 4)
                raise RuntimeError("refused read")
        with patch.object(core.copied, "output_dataset", new=header):
            with self.assertRaisesRegex(RuntimeError, "refused read"):
                self.call(sources, refuse_read)
        self.assertEqual(reads, [])
        sources, header, _, reads = self.sources()
        def refuse_mean(event):
            if event.operation == "average_repeats":
                raise RuntimeError("refused mean")
        with patch.object(core.copied, "output_dataset", new=header), \
                patch.object(core.calibration, "center", side_effect=AssertionError("mean before admission")):
            with self.assertRaisesRegex(RuntimeError, "refused mean"):
                self.call(sources, refuse_mean)
        self.assertEqual(reads, [(endpoint.SUBJECTS[0], "B", 0, 512)])


if __name__ == "__main__":
    unittest.main()
