"""Bounded truth reader with zero-sieve, read-only HDF access.

Only the file opener differs from the original reader. Its image/feature
binding, channel-bank bound, indexing, finite flags, and cleanup are inherited.
The draw caller supplies the unchanged admission callback.
"""
import math

import h5py

import finding_a_empirical_inputs as inputs
from truth_hdf_io import open_truth_readonly


class DirectTruthReader(inputs.TruthReader):
    def _open(self):
        if self._features is not None:
            return
        self._check()
        self._features = open_truth_readonly(inputs._path(self.root / "features.h5"))
        self._check()
        self._flags = open_truth_readonly(inputs._path(self.root / "finite_packed.h5"))
        inputs._require(set(self._features) == set(self._flags) == set(inputs.LAYERS),
                        "exact_truth_layer_vocabulary_required")
        for layer, shape in zip(inputs.LAYERS, inputs.SHAPES):
            self._check()
            inputs._require(isinstance(self._features.get(layer, getlink=True), h5py.HardLink)
                            and isinstance(self._flags.get(layer, getlink=True), h5py.HardLink),
                            "local_truth_dataset_links_required")
            data, flags = self._features[layer], self._flags[layer]
            inputs._require(isinstance(data, h5py.Dataset) and isinstance(flags, h5py.Dataset)
                            and data.shape == (inputs.TEST_IMAGES,) + tuple(shape) and data.dtype.str == "<f4"
                            and flags.shape == (inputs.TEST_IMAGES, (math.prod(shape) + 7) // 8)
                            and flags.dtype.str == "|u1"
                            and all(not value.is_virtual and value.external is None for value in (data, flags)),
                            "exact_truth_feature_and_finite_headers_required")
