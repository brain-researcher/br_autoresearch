"""One fixed public-model reconstruction runtime qualification, callable only.

The caller must admit the allocation, input scope, paths and original absolute
deadline before importing/calling this module. No empirical reader, CLI, image
writer, parameter search or retry is provided. The only download is the fixed
public generator archive; selected VGG members come from the existing cache.
The existing downloader and archive inspector have no internal deadline hook:
checks surround them, and the parent's process deadline remains mandatory.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path, PurePosixPath
import resource
import stat
import sys
import time
import zipfile


HERE = Path(__file__).resolve().parent
EPISODE = HERE.parents[1]
ARCHITECTURE = (EPISODE / "outputs/source_pipeline_binding/bdpy_0_25/plaintext"
                / "bdpy/dl/torch/models.py")
ARCHITECTURE_SHA256 = "d1f10981e3dd95c9196db3bb97b9431158192605a65cf0644c19eb8a43ab8d5d"
VGG_ARCHIVE_NAME = "VGG_ILSVRC_19_layers.zip"
VGG_ARCHIVE_BYTES = 533_710_127
# Only these two members are opened. Pins are the authenticated attempt017
# archive inventory; this is not a new inventory or lab-local equivalence claim.
VGG_MEMBERS = {
    "checkpoint": {
        "path": "VGG_ILSVRC_19_layers/VGG_ILSVRC_19_layers.pt",
        "uncompressed_bytes": 574_682_731,
        "sha256": "db5a4dbdbbfa55046d57a39663fa4a16d669ef6548051a2968b8df1c4fa057e8",
    },
    "feature_std": {
        "path": ("VGG_ILSVRC_19_layers/estimated_cnn_feat_std_"
                 "VGG_ILSVRC_19_layers_ImgSize_224x224_chwise_dof1.mat"),
        "uncompressed_bytes": 53_064,
        "sha256": "3a0dbe06ecbada569f711e56087a5d7cec26fc76fcc9bd02e535295802baf477",
    },
}
COPY_BYTES = 1024 * 1024
FIXTURE_OWNER = "protocol-neutral-fixture"


class ReconstructionQualificationError(RuntimeError):
    """Categorical failure only; the parent owns durable failure reporting."""

    def __init__(self, stage, failure_type):
        self.stage = stage
        self.failure_type = failure_type
        super().__init__(stage + ":" + failure_type)


def _stage_vgg(archive_path, directory, *, archive_tools, check):
    """Exclusive selected-member extraction with exact size and SHA binding."""
    check()
    descriptor, identity = archive_tools._open_regular(archive_path)
    with os.fdopen(descriptor, "rb", closefd=True) as raw:
        if identity[2] != VGG_ARCHIVE_BYTES:
            raise ValueError("cached_VGG_archive_size_differs_from_pin")
        with zipfile.ZipFile(raw) as archive:
            selected = {}
            for role, pin in VGG_MEMBERS.items():
                matches = [info for info in archive.infolist()
                           if info.filename == pin["path"]]
                if (len(matches) != 1 or matches[0].is_dir()
                        or not stat.S_ISREG(matches[0].external_attr >> 16)
                        or matches[0].file_size != pin["uncompressed_bytes"]):
                    raise ValueError("unique_pinned_regular_VGG_member_required")
                selected[role] = matches[0]
            directory.mkdir(mode=0o700, parents=False, exist_ok=False)
            staged = {}
            for role, info in selected.items():
                check()
                pin = VGG_MEMBERS[role]
                destination = directory / PurePosixPath(pin["path"]).name
                partial = destination.with_name(destination.name + ".partial")
                size, digest = 0, hashlib.sha256()
                with archive.open(info, "r") as source, partial.open("xb") as target:
                    while True:
                        check()
                        block = source.read(COPY_BYTES)
                        if not block:
                            break
                        if size + len(block) > pin["uncompressed_bytes"]:
                            raise ValueError("VGG_member_exceeds_immutable_size")
                        target.write(block)
                        digest.update(block)
                        size += len(block)
                    target.flush()
                    os.fsync(target.fileno())
                    os.fchmod(target.fileno(), 0o440)
                check()
                if size != pin["uncompressed_bytes"] or digest.hexdigest() != pin["sha256"]:
                    raise ValueError("VGG_member_identity_mismatch")
                os.link(partial, destination)
                partial.unlink()
                staged[role] = dict(pin, resolved_path=str(destination),
                                    extracted_bytes=size, identity_verified=True)
        if archive_tools._identity(os.fstat(raw.fileno())) != identity:
            raise ValueError("cached_VGG_archive_changed_during_extraction")
    check()
    return staged


def _architecture(check):
    check()
    if hashlib.sha256(ARCHITECTURE.read_bytes()).hexdigest() != ARCHITECTURE_SHA256:
        raise ValueError("pinned_architecture_source_mismatch")
    name = "ep21_reconstruction_qualification_pinned_architecture"
    spec = importlib.util.spec_from_file_location(name, ARCHITECTURE)
    if spec is None or spec.loader is None:
        raise ValueError("pinned_architecture_loader_unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    check()
    return module


def _load_model(constructor, checkpoint, *, torch, check):
    check()
    model = constructor().to(device="cpu", dtype=torch.float32)
    check()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model.eval()
    count, tensors = 0, 0
    for parameter in model.parameters():
        check()
        if (parameter.dtype != torch.float32 or parameter.device.type != "cpu"
                or not bool(torch.isfinite(parameter).all().item())):
            raise ValueError("finite_float32_CPU_model_parameters_required")
        count += parameter.numel()
        tensors += 1
    if not tensors:
        raise ValueError("model_has_no_parameters")
    check()
    return model, dict(strict_load=True, weights_only=True, parameter_tensors=tensors,
                       parameter_count=count, parameter_dtype="float32", device="cpu",
                       parameters_finite=True, evaluation_mode=True,
                       parameter_requires_grad_unchanged=True)


def qualify_reconstruction(*, budget, scratch, retained_archive_dir):
    """Run exactly one fixed 19-layer/200-step public fixture; return metadata.

    ``scratch`` is the caller-admitted existing real node-local directory.
    ``retained_archive_dir`` is a NEW direct child of the admitted durable run;
    its parent must exist. Child extraction paths also refuse reuse. The budget
    supplies check()/check_storage(); the original absolute deadline is never
    reset here. Generated images, features and losses are never serialized.
    """
    started = time.monotonic()
    stage = "admitted_budget_and_directories"
    try:
        if not callable(getattr(budget, "check", None)) or not callable(
                getattr(budget, "check_storage", None)):
            raise TypeError("explicit_runtime_and_storage_budget_required")

        def check():
            budget.check()
            budget.check_storage()

        check()
        scratch = Path(scratch)
        retained_archive_dir = Path(retained_archive_dir)
        if (not scratch.is_absolute() or not retained_archive_dir.is_absolute()
                or scratch.is_symlink() or not scratch.is_dir()):
            raise ValueError("admitted_absolute_real_directories_required")
        retained_archive_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
        if str(HERE) not in sys.path:
            sys.path.insert(0, str(HERE))
        import generator_assets
        archive_tools = generator_assets.archive_tools

        stage = "generator_archive_download"
        check()
        generator_pin = dict(filename=generator_assets.ARCHIVE_NAME,
                             url="https://ndownloader.figshare.com/files/52399790",
                             size_bytes=generator_assets.ARCHIVE_BYTES,
                             md5=generator_assets.ARCHIVE_MD5)
        download, _ = archive_tools.download_or_reuse(generator_pin, retained_archive_dir)
        check()
        stage = "generator_member_authentication_and_staging"
        generator_assets_record = generator_assets.extract_generator_assets(
            retained_archive_dir / generator_pin["filename"], scratch / "generator",
            budget_check=check)
        check()

        stage = "cached_VGG_selected_member_staging"
        cache_root = os.environ.get("SCRATCH", "")
        if not cache_root or not Path(cache_root).is_absolute():
            raise ValueError("existing_SCRATCH_cache_required")
        vgg_archive = (Path(cache_root) / "br_autoresearch" / EPISODE.name
                       / "public_model_assets" / VGG_ARCHIVE_NAME)
        vgg = _stage_vgg(vgg_archive, scratch / "vgg", archive_tools=archive_tools,
                        check=check)

        stage = "public_model_runtime_imports"
        check()
        import numpy as np
        from scipy.io import loadmat
        import torch
        import reconstruction_core as core
        if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
            raise ValueError("parent_must_configure_one_torch_thread")
        architecture = _architecture(check)
        stage = "VGG_strict_load"
        encoder, encoder_record = _load_model(
            architecture.VGG19, vgg["checkpoint"]["resolved_path"], torch=torch, check=check)
        stage = "generator_strict_load"
        generator, generator_record = _load_model(
            architecture.AlexNetGenerator,
            generator_assets_record["members"]["checkpoint"]["resolved_path"],
            torch=torch, check=check)

        stage = "public_statistics_and_latent_bounds"
        check()
        upper = np.loadtxt(generator_assets_record["members"]["latent_upper_bound"]
                           ["resolved_path"], delimiter=" ").reshape((4096,))
        if (upper.dtype not in (np.dtype("float32"), np.dtype("float64"))
                or not np.isfinite(upper).all() or not (upper >= 0).all()):
            raise ValueError("finite_nonnegative_4096_latent_bounds_required")
        upper_metadata = dict(shape=list(upper.shape), dtype=str(upper.dtype),
                              finite=True, nonnegative=True, element_count=int(upper.size))
        # Match the source's ordinary MATLAB dictionary and preserve its shape
        # and floating dtype. No h5py, squeeze, cast or substitute statistic.
        loaded = loadmat(vgg["feature_std"]["resolved_path"],
                         variable_names=list(core.LAYERS))
        public_std, std_metadata = {}, {}
        for layer in core.LAYERS:
            check()
            value = loaded.get(layer)
            if (not isinstance(value, np.ndarray) or value.size == 0
                    or value.dtype not in (np.dtype("float32"), np.dtype("float64"))
                    or not np.isfinite(value).all() or not (value >= 0).all()
                    or not np.isfinite(value.mean()) or not value.mean() > 0):
                raise ValueError("complete_finite_positive_mean_public_SD_required")
            public_std[layer] = value
            std_metadata[layer] = dict(shape=list(value.shape), dtype=str(value.dtype),
                                       element_count=int(value.size), finite=True,
                                       nonnegative=True, mean_positive=True)
        del loaded

        stage = "single_fixed_non_neural_fixture"
        features, means, fixture_metadata = {}, {}, {}
        for layer, shape in zip(core.LAYERS, core.SHAPES):
            check()
            # Signed integer arithmetic precedes float32 conversion. Every
            # layer uses the same fixed C-order formula without observations.
            indices = np.arange(int(np.prod(shape)), dtype=np.int32)
            values = ((indices % 257) - 128).astype(np.float32) / np.float32(128)
            del indices
            values = values.reshape((1,) + shape)
            features[layer] = values
            means[layer] = np.zeros_like(values)
            fixture_metadata[layer] = dict(shape=list(values.shape), dtype=str(values.dtype),
                                           element_count=int(values.size), finite=True,
                                           decoder_mean="all_zero_float32_fixture")
        check()
        stage = "one_full_200_update_reconstruction"
        reconstruction_started = time.monotonic()
        result = core.reconstruct(
            features, decoder_training_means=means, public_feature_std=public_std,
            encoder=encoder, generator=generator, latent_upper_bound=upper,
            method="within_source", source=FIXTURE_OWNER, target=None,
            decoder_owner=FIXTURE_OWNER, budget_check=check)
        reconstruction_seconds = time.monotonic() - reconstruction_started
        stage = "in_memory_output_metadata"
        check()
        if (result.raw_rgb.shape != (224, 224, 3)
                or result.raw_rgb.dtype not in (np.dtype("float32"), np.dtype("float64"))
                or not np.isfinite(result.raw_rgb).all()
                or result.image_uint8.shape != (224, 224, 3)
                or result.image_uint8.dtype != np.dtype("uint8")
                or result.operational_metadata["completed_updates"] != 200
                or result.operational_metadata["generator_forwards"] != 200):
            raise ValueError("complete_fixed_reconstruction_output_required")
        output_metadata = dict(
            raw_shape=list(result.raw_rgb.shape), raw_dtype=str(result.raw_rgb.dtype),
            raw_finite=True, final_shape=list(result.image_uint8.shape),
            final_dtype=str(result.image_uint8.dtype), final_finite=True,
            arrays_retained_or_serialized=False)
        operational = dict(result.operational_metadata)
        del result, features, means, public_std, upper, encoder, generator
        check()
        return dict(
            status="completed_public_runtime_qualification",
            qualification_scope="one_fixed_non_neural_fixture_no_scientific_inference",
            fixture_count=1, fixture_definition="float32((C_order_index_mod_257-128)/128)",
            fixture_decoder_mean_owner=FIXTURE_OWNER,
            generator_download=download, generator_assets=generator_assets_record,
            vgg_selected_members=vgg, architecture_sha256=ARCHITECTURE_SHA256,
            models=dict(encoder=encoder_record, generator=generator_record),
            feature_fixture=fixture_metadata, public_feature_SD=std_metadata,
            latent_upper_bound=upper_metadata,
            reconstruction=operational, output=output_metadata,
            reconstruction_elapsed_seconds=reconstruction_seconds,
            total_elapsed_seconds=time.monotonic() - started,
            process_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
            original_parent_deadline_preserved=True, image_or_loss_artifacts_written=False,
            neural_or_empirical_payload_accessed=False, empirical_access_authorized=False,
            protocol_lock_changed=False, scientific_success_claim=False,
            evaluator_lab_local_equivalence_established=False)
    except Exception as error:
        # Parent persists this categorical stage/type, never an exception's
        # potentially value-bearing message. No retry or partial success.
        raise ReconstructionQualificationError(stage, type(error).__name__) from None
