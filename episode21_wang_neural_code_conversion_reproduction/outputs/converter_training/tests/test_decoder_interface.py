"""Small source-free fixtures; run only in the coordinator's allocation."""
from contextlib import ExitStack
import io
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import decoder_interface as interface

assembly = interface.assembly


class DecoderInterfaceTests(unittest.TestCase):
    def axis(self):
        native = 1000 + 2 * np.arange(600, dtype=np.int64)
        mask = np.arange(600) < 551
        mask[1] = False
        normalization = SimpleNamespace(participant="sub-03", native_ids=native,
            eligible_ids=native[mask], eligible_mask=mask,
            mean=np.zeros(600, dtype=np.float32), std=mask.astype(np.float32))
        return interface.native_axis(normalization)

    def raw_block(self, axis, start=13, count=3, layer="conv5_1", channel=3):
        shape, _ = interface.active_shape(layer, channel)
        spatial = np.arange(start, start + count, dtype=np.int64)
        full_f = spatial if channel is None else channel + shape[0] * spatial
        full_c = (spatial if channel is None else channel * shape[1] * shape[2]
                  + (spatial % shape[1]) * shape[2] + spatial // shape[1])
        ids = np.asarray([np.roll(axis.eligible_native_ids[:500], index)
                          for index in range(count)])
        columns = np.searchsorted(axis.full_native_ids, ids)
        return assembly.SparseDecoderBlock(axis.participant, layer, full_f, full_c,
            ids, columns, np.ones((count, 500), dtype=np.float32),
            np.full(count, .25, dtype=np.float32), np.zeros((count, 500), dtype=np.float32),
            np.ones((count, 500), dtype=np.float32), np.arange(count, dtype=np.float32) + 10,
            np.full(count, 2, dtype=np.float32))

    def ref(self, layer, channel, start, stop, path="fixture"):
        return assembly.FragmentRef("outputs/decoder_training/" + path + "/channels.h5",
            "grouped_channels", f"{layer}/channel_{channel:03d}",
            "outputs/decoder_training/" + path + "/status.json",
            assembly.coverage.Span("sub-03", layer, channel, start, stop))

    def test_selective_full_source_load_never_opens_gram(self):
        values = np.arange(interface.IMAGES * 2, dtype=np.float32).reshape(interface.IMAGES, 2)
        encoded = io.BytesIO()
        np.save(encoded, values, allow_pickle=False)
        opened = []
        original = zipfile.ZipFile

        class RecordingZip(original):
            def open(self, name, *args, **kwargs):
                opened.append(name.filename if isinstance(name, zipfile.ZipInfo) else name)
                return super().open(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "standin.npz"
            with original(path, "w") as archive:
                archive.writestr("normalized_X.npy", encoded.getvalue())
                archive.writestr("penalized_Gram.npy", b"INVALID_AND_MUST_NOT_BE_READ")
            with mock.patch.object(interface.zipfile, "ZipFile", RecordingZip):
                actual = interface._read_normalized_X(path, 2, lambda: None)
            np.testing.assert_array_equal(actual, values)
            self.assertEqual(opened, ["normalized_X.npy", "normalized_X.npy"])
            self.assertFalse(actual.flags.writeable)
            with self.assertRaisesRegex(ValueError, "shape required|byte bound"):
                interface._read_normalized_X(path, 1, lambda: None)

    def test_exact_image_ids_reject_reordering_missing_or_duplicate_rows(self):
        ids = tuple(f"source_{index}" for index in range(interface.IMAGES))
        self.assertEqual(interface.require_image_order(ids, ids), ids)
        for changed in (ids[::-1], ids[:-1], ids[:-1] + (ids[0],)):
            with self.subTest(kind=len(changed)), self.assertRaisesRegex(ValueError, "identities and order"):
                interface.require_image_order(changed, ids)

    def test_eligible_axis_mapping_preserves_ranked_weights_and_rejects_excluded_ids(self):
        axis = self.axis()
        raw = self.raw_block(axis)
        block = interface.map_decoder_block(raw, axis, channel=3)
        self.assertEqual(len(axis.eligible_native_ids), 550)
        self.assertEqual(block.unit_start, 13)
        self.assertIs(block.weights, raw.weights)
        self.assertIs(block.bias, raw.bias)
        self.assertTrue(np.any(block.eligible_columns != raw.native_columns))
        np.testing.assert_array_equal(axis.eligible_native_ids[block.eligible_columns],
                                      raw.selected_native_voxel_ids)
        bad = self.raw_block(axis)
        bad.selected_native_voxel_ids[0, 0] = axis.full_native_ids[1]
        with self.assertRaisesRegex(ValueError, "absent from eligible axis"):
            interface.map_decoder_block(bad, axis, channel=3)
        wrong_axis = self.raw_block(axis)
        wrong_axis.native_columns[0, 0] += 1
        with self.assertRaisesRegex(ValueError, "coordinates disagree"):
            interface.map_decoder_block(wrong_axis, axis, channel=3)

    def test_source_feature_hyperslabs_are_exact_and_target_moments_are_applied_once(self):
        block = interface.map_decoder_block(self.raw_block(self.axis()), self.axis(), channel=3)

        class FeatureDataset:
            shape = (4, 512, 14, 14)
            def __init__(self):
                self.reads = []
            def __getitem__(self, key):
                self.reads.append(key)
                rows, channel, height, width = key
                self_test.assertEqual(rows, slice(None))
                self_test.assertEqual(channel, 3)
                return (1000 * np.arange(4, dtype=np.float32)[:, None]
                        + 100 * channel + 10 * np.arange(height.start, height.stop, dtype=np.float32)
                        + width)

        self_test = self
        data = FeatureDataset()
        flags = np.full((4, 512 * 14 * 14 // 8), 255, dtype=np.uint8)
        flags[:, 0] = 0  # An unrelated channel must not be accessed/required.
        actual = interface._read_feature_units(data, flags, block, lambda: None)
        raw = 1000 * np.arange(4, dtype=np.float32)[:, None] + np.asarray([430, 301, 311], dtype=np.float32)
        np.testing.assert_array_equal(actual, (raw - np.asarray([10, 11, 12], dtype=np.float32)) / 2)
        self.assertEqual(data.reads, [(slice(None), 3, slice(13, 14), 0),
                                     (slice(None), 3, slice(0, 2), 1)])
        index = int(block.full_layer_C_index[1])
        flags[0, index // 8] &= np.uint8(255 ^ (1 << (index % 8)))
        with self.assertRaisesRegex(ValueError, "source-feature flag"):
            interface._read_feature_units(data, flags, block, lambda: None)

    def test_channel_coverage_rejects_gap_overlap_and_missing_tail(self):
        a = self.ref("conv5_1", 3, 0, 100, "z_prefix")
        b = self.ref("conv5_1", 3, 100, 196, "a_later")
        other = self.ref("conv5_1", 4, 0, 196)
        pair = SimpleNamespace(target_axis=self.axis(),
            manifest=SimpleNamespace(fragments=(b, other, a)))
        self.assertEqual(interface._channel_refs(pair, "conv5_1", 3), [a, b])
        for bounds in ((101, 196), (99, 196), (100, 195)):
            pair.manifest.fragments = (a, self.ref("conv5_1", 3, *bounds))
            with self.subTest(bounds=bounds), self.assertRaisesRegex(ValueError, "coverage|tail"):
                interface._channel_refs(pair, "conv5_1", 3)
        self.assertEqual(interface.active_layer_units("conv5_1", 3), 196)
        self.assertEqual(interface.active_layer_units("fc8", None), 1000)
        for layer, channel in (("conv5_1", None), ("conv5_1", True), ("conv5_1", 512), ("fc8", 0)):
            with self.subTest(layer=layer, channel=channel), self.assertRaises(ValueError):
                interface.active_layer_units(layer, channel)

    def test_existing_sparse_iterator_filters_before_io_and_preserves_default(self):
        refs = (self.ref("conv5_1", 3, 0, 196), self.ref("conv5_1", 4, 0, 196))
        manifest = assembly.CompleteDecoderManifest(refs, (assembly.required_state("sub-03"),))
        budget = SimpleNamespace(check=lambda: None)
        with mock.patch.object(assembly, "_require_reader_context"), \
                mock.patch.object(assembly, "verify_catalog"), \
                mock.patch.object(assembly, "_admit_normalization"), \
                mock.patch.object(assembly, "_scan_fragments", return_value=iter(())) as scan:
            list(assembly.iter_sparse_blocks(manifest, participant="sub-03", layer="conv5_1", budget=budget))
            self.assertEqual(scan.call_args.args[1], list(refs))
            list(assembly.iter_sparse_blocks(manifest, participant="sub-03", layer="conv5_1", channel=4, budget=budget))
            self.assertEqual(scan.call_args.args[1], [refs[1]])
            scan.reset_mock()
            for invalid in (-1, 512, True, 1.5):
                with self.subTest(channel=invalid), self.assertRaisesRegex(ValueError, "in-range channel"):
                    list(assembly.iter_sparse_blocks(manifest, participant="sub-03", layer="conv5_1", channel=invalid, budget=budget))
            scan.assert_not_called()

    def test_decoder_iterator_sorts_spans_without_caching_whole_channel(self):
        axis = self.axis()
        refs = (self.ref("fc8", 0, 64, 1000, "a_later"), self.ref("fc8", 0, 0, 64, "z_prefix"))
        state = assembly.required_state("sub-03")
        pair = interface.ContentPair(assembly.CompleteDecoderManifest(refs, (state,)), None, state, None, axis)
        visited = []

        def scan(base, selected, state, normalization, check, block_units):
            self.assertEqual(len(selected), 1)
            span = selected[0].span
            visited.append(span.start)
            for start in range(span.start, span.stop, block_units):
                yield self.raw_block(axis, start, min(block_units, span.stop - start), "fc8", None)

        with mock.patch.object(assembly, "_require_reader_context"), \
                mock.patch.object(assembly, "verify_catalog") as verify, \
                mock.patch.object(assembly, "_scan_fragments", side_effect=scan):
            blocks = interface.iter_decoder_blocks(pair, layer="fc8", channel=None,
                budget=SimpleNamespace(check=lambda: None), block_units=64)
            first = next(blocks)
            self.assertEqual(visited, [0])  # No second-fragment read ahead.
            starts, count = [first.unit_start], len(first.bias)
            for block in blocks:
                starts.append(block.unit_start)
                self.assertLessEqual(len(block.bias), 64)
                count += len(block.bias)
            self.assertEqual(starts, list(range(0, 1000, 64)))
            self.assertEqual(visited, [0, 64])
            self.assertEqual(count, 1000)
            verify.assert_called_once_with(refs, (state,))

    def test_recovery_review_is_revalidated_before_ordered_scans_and_cannot_be_dropped(self):
        axis = self.axis()
        prefix = self.ref("conv5_1", 3, 0, 64, "z_reviewed_prefix")
        complement = self.ref("conv5_1", 3, 64, 196,
            "full_continuation_001/io_remainder_shards_001/work_0027")
        unrelated = self.ref("conv5_1", 4, 0, 196)
        refs = (complement, unrelated, prefix)
        state = assembly.required_state("sub-03")
        review = {"fixture": "exact retained review"}
        manifest = assembly.CompleteDecoderManifest(refs, (state,), review)
        pair = interface.ContentPair(manifest, None, state, None, axis)
        budget = SimpleNamespace(check=lambda: None)
        events = []

        def validate_catalog(actual_refs, actual_review):
            self.assertEqual(actual_refs, refs)  # Whole collection, not one ref.
            self.assertIs(actual_review, review)
            events.append("catalog")

        def revalidate(actual_review, check):
            self.assertIs(actual_review, review)
            self.assertIs(check, budget.check)
            events.append("review")

        def scan(base, selected, actual_state, normalization, check, block_units, **kwargs):
            self.assertEqual(events[:2], ["catalog", "review"])
            self.assertEqual(kwargs, {"recovery_review": review})
            self.assertIs(kwargs["recovery_review"], review)
            self.assertEqual(len(selected), 1)
            span = selected[0].span
            events.append(span.start)
            for start in range(span.start, span.stop, block_units):
                yield self.raw_block(axis, start, min(block_units, span.stop - start))

        with mock.patch.object(assembly, "_require_reader_context"), \
                mock.patch.object(assembly, "verify_catalog") as verify, \
                mock.patch.object(assembly, "_validate_recovery_catalog", side_effect=validate_catalog), \
                mock.patch.object(assembly, "_revalidate_recovery_review", side_effect=revalidate), \
                mock.patch.object(assembly, "_scan_fragments", side_effect=scan):
            blocks = interface.iter_decoder_blocks(pair, layer="conv5_1", channel=3,
                                                   budget=budget, block_units=64)
            first = next(blocks)
            self.assertEqual(events, ["catalog", "review", 0])
            starts = [first.unit_start] + [block.unit_start for block in blocks]
            self.assertEqual(starts, [0, 64, 128, 192])
            self.assertEqual(events, ["catalog", "review", 0, 64])
            verify.assert_called_once_with(refs, (state,))
            self.assertIs(pair.manifest, manifest)

        with mock.patch.object(assembly, "_require_reader_context"), \
                mock.patch.object(assembly, "verify_catalog"), \
                mock.patch.object(assembly, "_validate_recovery_catalog"), \
                mock.patch.object(assembly, "_revalidate_recovery_review",
                    side_effect=ValueError("recovery evidence changed")), \
                mock.patch.object(assembly, "_scan_fragments") as scan:
            with self.assertRaisesRegex(ValueError, "recovery evidence changed"):
                next(interface.iter_decoder_blocks(pair, layer="conv5_1", channel=3, budget=budget))
            scan.assert_not_called()

        # The real no-review catalog rule must refuse the same I/O remainder.
        dropped = interface.ContentPair(assembly.CompleteDecoderManifest(refs, (state,)),
                                         None, state, None, axis)
        with mock.patch.object(assembly, "_require_reader_context"), \
                mock.patch.object(assembly, "verify_catalog"), \
                mock.patch.object(assembly, "_scan_fragments") as scan:
            with self.assertRaisesRegex(ValueError, "explicit review binding"):
                next(interface.iter_decoder_blocks(dropped, layer="conv5_1", channel=3, budget=budget))
            scan.assert_not_called()

    def test_provider_binds_active_size_and_keeps_feature_rows(self):
        from content_loss import ActiveLayer
        import content_channel_cache as cache
        import h5py
        axis = self.axis()
        block = interface.map_decoder_block(self.raw_block(axis, 0, 2, "fc8", None), axis, channel=None)
        source_features = np.zeros((interface.IMAGES, 2), dtype=np.float32)
        state = assembly.required_state("sub-03")
        pair = interface.ContentPair(assembly.CompleteDecoderManifest((), (state,)),
                                     None, state, None, axis)
        budget = SimpleNamespace(allocation={"job_id": "fixture", "hostname": "compute"},
                                 check=lambda: None)
        with mock.patch.object(assembly, "_require_reader_context"), \
                mock.patch.object(assembly, "verify_catalog") as catalog, \
                mock.patch.object(interface, "_iter_content_blocks", return_value=iter([
                    interface.ContentBlock(block, source_features)])) as read:
            provider = interface.content_block_provider(pair, budget=budget)
            self.assertTrue(callable(provider.check_pass))
            result = list(provider(ActiveLayer("fc8", None, 1000)))
            self.assertEqual(result[0].unit_start, 0)
            self.assertIs(result[0].standardized_features, source_features)
            self.assertNotIn("reader_factory", read.call_args.kwargs)
            read.reset_mock()
            with self.assertRaisesRegex(ValueError, "active layer size"):
                list(provider(ActiveLayer("fc8", None, 999)))
            read.assert_not_called()

            # Composition must use the original provider's exact guard, not a
            # duplicate admission/guard, and select only the fixed new reader.
            catalog.reset_mock()
            wrapped = object()
            original_provider = interface.content_block_provider
            with mock.patch.object(interface, "content_block_provider", wraps=original_provider) as bind, \
                    mock.patch.object(cache, "TwoPassLossCache", return_value=wrapped) as retain:
                self.assertIs(cache.make_cached_provider(pair, budget=budget, block_units=2), wrapped)
                bind.assert_called_once_with(pair, budget=budget, block_units=2,
                                             reader_factory=cache.ContiguousChannelReader)
                bound_provider = retain.call_args.args[0]
                self.assertIs(retain.call_args.kwargs["check_pass"], bound_provider.check_pass)
                self.assertIs(retain.call_args.kwargs["check"], budget.check)
                bound_provider.check_pass()
            catalog.assert_called_once()

        # A caller-selected reader is constructed only after the same actual
        # HDF header admission; invalid headers cannot invoke its constructor.
        for invalid in (False, True):
            with self.subTest(invalid_header=invalid), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "outputs/steward_provisioning/regular_vgg19_sealed/sub-03"
                root.mkdir(parents=True)
                raw = np.arange(28, dtype=np.float32).reshape(4, 7)
                with h5py.File(root / "features.h5", "w") as handle:
                    handle.create_dataset("fc8", data=raw.astype(np.float64) if invalid else raw)
                with h5py.File(root / "finite_packed.h5", "w") as handle:
                    handle.create_dataset("fc8", data=np.full((4, 1), 127, dtype=np.uint8))
                supplied = interface.ContentPair(pair.manifest,
                    SimpleNamespace(axis=SimpleNamespace(participant="sub-03"), feature_root=root),
                    state, None, axis)
                created = []

                def reader_factory(*args, **kwargs):
                    reader = cache.ContiguousChannelReader(*args, **kwargs)
                    created.append(reader)
                    return reader

                with mock.patch.object(assembly, "BASE", Path(directory)), \
                        mock.patch.object(interface, "IMAGES", 4), \
                        mock.patch.object(interface, "active_shape", return_value=((7,), 7)):
                    stream = interface._iter_content_blocks(supplied, layer="fc8", channel=None,
                        decoder_blocks=iter((block,)), check=lambda: None, reader_factory=reader_factory)
                    if invalid:
                        with self.assertRaisesRegex(ValueError, "layout required"):
                            list(stream)
                        self.assertEqual(created, [])
                    else:
                        result = list(stream)
                        expected = raw[:, :2].copy()
                        expected -= block.target_mean
                        expected /= block.target_std
                        np.testing.assert_array_equal(result[0].standardized_features, expected)
                        self.assertEqual(len(created), 1)
                        self.assertEqual(created[0].retained_bytes, 0)
                        self.assertTrue(created[0]._closed)

    def test_bound_provider_replays_evidence_once_and_refuses_changed_bindings(self):
        from content_loss import ActiveLayer
        # The physical coefficient reader is stubbed; its real native mapping,
        # spatial coverage and provider pass lifecycle remain in this fixture.
        # Feature payloads are fabricated at full4712-row/bounded-block shape.
        for case in ("repeat", "between", "mid", "during_admission", "allocation", "missing", "direct_guard"):
            with self.subTest(case=case), ExitStack() as stack:
                axis = self.axis()
                refs = (self.ref("conv5_1", 3, 64, 196,
                    "full_continuation_001/io_remainder_shards_001/work_0027"),
                    self.ref("conv5_1", 3, 0, 64, "z_reviewed_prefix"))
                state = assembly.required_state("sub-03")
                review = {"fixture": {"fixed": True}}
                original_review = None if case == "missing" else review
                manifest = assembly.CompleteDecoderManifest(refs, (state,), original_review)
                pair = interface.ContentPair(manifest, None, state, None, axis)
                budget = SimpleNamespace(allocation={"job_id": "first", "hostname": "compute"},
                                         check=lambda: None)
                visited, copies = [], []
                actual_validate = assembly._validate_recovery_catalog

                def validate(actual_refs, actual_review):
                    self.assertEqual(actual_refs, refs)
                    if actual_review is None:
                        return actual_validate(actual_refs, actual_review)
                    self.assertEqual(actual_review, {"fixture": {"fixed": True}})
                    self.assertIsNot(actual_review, review)
                    self.assertIsNot(actual_review["fixture"], review["fixture"])

                def revalidate(actual_review, check):
                    copies.append(actual_review)
                    self.assertEqual(actual_review, {"fixture": {"fixed": True}})
                    if case == "during_admission":
                        # Python equality would consider True==1. The existing
                        # exact-JSON-type guard must still refuse this mutation.
                        review["fixture"]["fixed"] = 1
                    check()

                def scan(base, selected, actual_state, normalization, check, block_units, **kwargs):
                    self.assertIs(kwargs["recovery_review"], copies[0])
                    self.assertIs(kwargs["recovery_review"]["fixture"]["fixed"], True)
                    span = selected[0].span
                    visited.append(span.start)
                    for start in range(span.start, span.stop, block_units):
                        check()
                        yield self.raw_block(axis, start, min(block_units, span.stop - start))

                def features(bound_pair, *, layer, channel, decoder_blocks, check):
                    self.assertIsNot(bound_pair.manifest, manifest)
                    self.assertIs(bound_pair.manifest.fragments, refs)
                    for block in decoder_blocks:
                        check()
                        yield interface.ContentBlock(block,
                            np.zeros((interface.IMAGES, len(block.bias)), dtype=np.float32))

                context = stack.enter_context(mock.patch.object(assembly, "_require_reader_context"))
                catalog = stack.enter_context(mock.patch.object(assembly, "verify_catalog"))
                stack.enter_context(mock.patch.object(assembly, "_validate_recovery_catalog", side_effect=validate))
                evidence = stack.enter_context(mock.patch.object(assembly, "_revalidate_recovery_review", side_effect=revalidate))
                scan_call = stack.enter_context(mock.patch.object(assembly, "_scan_fragments", side_effect=scan))
                stack.enter_context(mock.patch.object(interface, "_iter_content_blocks", side_effect=features))
                if case in ("during_admission", "missing"):
                    pattern = "changed after provider" if case == "during_admission" else "explicit review binding"
                    with self.assertRaisesRegex(ValueError, pattern):
                        interface.content_block_provider(pair, budget=budget)
                    scan_call.assert_not_called()
                    self.assertEqual(evidence.call_count, 0 if case == "missing" else 1)
                    continue

                provider = interface.content_block_provider(pair, budget=budget)
                context.reset_mock()
                active = ActiveLayer("conv5_1", 3, 196)
                if case == "repeat":
                    for _ in range(3):
                        blocks = list(provider(active))
                        self.assertEqual([block.unit_start for block in blocks], [0, 64, 128, 192])
                        self.assertTrue(all(block.standardized_features.shape[0] == 4712 for block in blocks))
                    self.assertEqual(visited, [0, 64] * 3)
                    self.assertEqual(context.call_count, 6)  # Entry and exit of all three passes.
                elif case in ("between", "allocation", "direct_guard"):
                    list(provider(active))
                    before = scan_call.call_count
                    if case in ("between", "direct_guard"):
                        # Simulate mutation inside an entry budget callback.
                        budget.check = lambda: review.clear()
                        pattern = "changed after provider"
                    else:
                        budget.allocation = {"job_id": "second", "hostname": "compute"}
                        pattern = "another allocation"
                    with self.assertRaisesRegex(ValueError, pattern):
                        if case == "direct_guard":
                            provider.check_pass()
                        else:
                            list(provider(active))
                    self.assertEqual(scan_call.call_count, before)
                else:
                    stream = provider(active)
                    self.assertEqual(next(stream).unit_start, 0)
                    review["fixture"]["fixed"] = 1
                    # Mid-pass reads keep their private approved review; the
                    # final next() must refuse instead of completing the pass.
                    self.assertEqual([next(stream).unit_start for _ in range(3)], [64, 128, 192])
                    with self.assertRaisesRegex(ValueError, "changed after provider"):
                        next(stream)
                    self.assertEqual(visited, [0, 64])
                catalog.assert_called_once_with(refs, (state,))
                evidence.assert_called_once()


if __name__ == "__main__":
    unittest.main()
