"""Materialize EP21's approved independent-localizer masks in one Slurm job."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def size(path):
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def grid(image):
    qform, qcode = image.get_qform(coded=True)
    sform, scode = image.get_sform(coded=True)
    return {
        "shape": list(map(int, image.shape)), "affine": image.affine.tolist(),
        "zooms": list(map(float, image.header.get_zooms())),
        "units": list(image.header.get_xyzt_units()),
        "qform_code": int(qcode), "sform_code": int(scode),
        "qform": None if qform is None else qform.tolist(),
        "sform": None if sform is None else sform.tolist(),
    }


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Mask materialization requires the approved Slurm allocation")
    root = Path(os.environ["OAK"]) / "users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction"
    contract = json.loads((root / "outputs/protocol_mask_amendment_2026-09-30.yaml").read_text())
    assert contract["status"] == "scientist_approved_before_neural_outcomes"
    assert not contract["neural_fitting_scoring_authorized"]
    assert not contract["ood_values_or_derivatives_authorized"]
    assert sum(map(len, contract["components"].values())) == 29
    caps = contract["resources"]
    source = Path(os.environ["OAK"]) / "users/zijiao" / contract["source_acquisition"] / "dataset/derivatives/rois"
    stage = Path(os.environ["SCRATCH"]) / "br_autoresearch" / root.name / ("independent-masks-" + os.environ["SLURM_JOB_ID"])
    destination = root / "outputs/steward_provisioning/independent_visual_masks"
    if stage.exists() or destination.exists():
        raise RuntimeError("Existing stage or published masks: do not duplicate or overwrite")
    inputs = []
    for subject in contract["participants"]:
        for family, labels in contract["components"].items():
            assert family != "laion"
            for label in labels:
                path = source / subject / family / f"{subject}_space-T1w_res-1pt8_label-{label}_mask.nii.gz"
                if path.is_symlink() or not path.is_file():
                    raise RuntimeError(f"Missing/nonregular approved mask: {path}")
                inputs.append((subject, family, label, path))
    assert len(inputs) == 145
    # Size checks precede staging; no closed-directory traversal or value reads.
    input_bytes = sum(p.stat().st_size for _, _, _, p in inputs)
    if input_bytes > caps["scratch_bytes"]:
        raise RuntimeError("Approved inputs exceed temporary storage allocation")
    stage.mkdir(parents=True, mode=0o700)
    out = stage / "outputs"
    out.mkdir()
    reports = []
    for subject in contract["participants"]:
        reference = None
        union = None
        components = []
        for sub, family, label, path in inputs:
            if sub != subject:
                continue
            local = stage / path.name
            shutil.copyfile(path, local)
            image = nib.load(local)
            if len(image.shape) != 3:
                raise RuntimeError(f"Non-3D component: {path}")
            geometry = grid(image)
            if reference is None:
                reference = image
                reference_grid = geometry
                union = np.zeros(image.shape, dtype=bool)
                # Bound the largest possible index array and masks before output.
                projected = int(np.prod(image.shape)) * 9 * len(contract["participants"])
                if projected + input_bytes > caps["scratch_bytes"] or projected > caps["durable_bytes"]:
                    raise RuntimeError("Native grid cannot fit approved storage limits")
            elif geometry != reference_grid:
                raise RuntimeError(f"Native component grid mismatch: {path}")
            values = np.asanyarray(image.dataobj)
            if not np.all(np.isfinite(values)) or not np.all((values == 0) | (values == 1)):
                raise RuntimeError(f"Nonfinite/nonbinary supplied component: {path}")
            present = values == 1
            count = int(np.count_nonzero(present))
            if count == 0:
                raise RuntimeError(f"Empty required component: {path}")
            union |= present
            components.append({"family": family, "label": label,
                               "source": str(path), "sha256": sha256(local),
                               "voxel_count": count,
                               "localizer_origin": contract["provenance"][family]})
            del values, present
            if size(stage) > caps["scratch_bytes"]:
                raise RuntimeError("Temporary storage cap exceeded")
        assert len(components) == 29
        indices = np.flatnonzero(union.ravel(order="C")).astype(np.int64)
        assert len(indices) == np.count_nonzero(union)
        assert np.all(np.diff(indices) > 0)
        # Retain the provider grid and coded transforms; change only mask payload.
        mask = nib.Nifti1Image(union.astype(np.uint8), None, reference.header.copy())
        mask.set_data_dtype(np.uint8)
        mask.header.set_slope_inter(1, 0)
        mask_path = out / f"{subject}_space-T1w_desc-independentVisualUnion_mask.nii.gz"
        index_path = out / f"{subject}_space-T1w_desc-independentVisualUnion_voxel_indices.npy"
        nib.save(mask, mask_path)
        np.save(index_path, indices, allow_pickle=False)
        # Focused real-output regression: exact supplied-grid union/index binding.
        saved = nib.load(mask_path)
        if grid(saved) != reference_grid or not np.array_equal(np.asanyarray(saved.dataobj), union):
            raise RuntimeError("Saved union/grid differs from actual component union")
        saved_indices = np.load(index_path, allow_pickle=False)
        if not np.array_equal(saved_indices, np.flatnonzero(np.asanyarray(saved.dataobj).ravel(order="C"))):
            raise RuntimeError("Saved canonical voxel indices differ from saved union")
        reports.append({"participant": subject, "component_count": 29,
                        "grid": reference_grid, "components": components,
                        "union_voxel_count": len(indices),
                        "summed_component_voxels": sum(x["voxel_count"] for x in components),
                        "overlap_deduplicated": True,
                        "mask_file": mask_path.name, "mask_sha256": sha256(mask_path),
                        "voxel_indices_file": index_path.name,
                        "voxel_indices_sha256": sha256(index_path),
                        "index_order": "ascending_full_grid_C_order",
                        "real_union_and_index_regression": "passed"})
        print(json.dumps({"participant": subject, "union_voxels": len(indices)}), flush=True)
        if size(stage) > caps["scratch_bytes"] or size(out) > caps["durable_bytes"]:
            raise RuntimeError("Approved storage cap exceeded")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.7, 3.8), layout="constrained")
    counts = [r["union_voxel_count"] for r in reports]
    ax.barh([r["participant"] for r in reports], counts, color="#346b8c", height=.58)
    ax.invert_yaxis()
    ax.set_xlim(0, max(counts) * 1.23)
    for row, count in enumerate(counts):
        ax.text(count + max(counts) * .02, row, f"{count:,}", va="center", fontsize=10)
    ax.set_xlabel("Voxels in supplied-grid union (overlap counted once)")
    ax.set_title("QC · Independent visual-mask support", loc="left", fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.text(.01, -.05, "29 fixed independent-localizer masks per participant; adapted visual support.\nVoxel counts do not measure neural conversion or whole-cortex coverage.", fontsize=9)
    stem = "independent_visual_mask_support_qc_2026-09-30"
    for ext in ("png", "svg"):
        fig.savefig(out / f"{stem}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    dump(out / f"{stem}.json", {"evidence_status": "QC", "source": "mask_report.json",
         "measure": "deduplicated_native_union_voxel_count", "participants": contract["participants"],
         "counts": counts, "contains_neural_outcomes": False})
    report = {"schema_version": "ep21.independent_native_mask_report.v1",
              "evidence_status": "QC", "scientist_amendment": "outputs/protocol_mask_amendment_2026-09-30.yaml",
              "provider_code_commit": contract["provider_code_commit"],
              "primary_roi": contract["primary_roi"], "subjects": reports,
              "coverage_claim": contract["coverage_claim"], "resources": caps,
              "job_id": os.environ["SLURM_JOB_ID"], "input_bytes": input_bytes,
              "numpy_version": np.__version__, "nibabel_version": nib.__version__,
              "mask_values_only": True, "beta_values_read": False,
              "ood_values_or_derivatives_read": False, "neural_fitting_scoring_run": False,
              "beta_grid_alignment": "not_yet_authenticated_no_beta_values_read",
              "construction": contract["construction"], "publication": "complete"}
    dump(out / "mask_report.json", report)
    # Include duplicate figure delivery in the durable allocation calculation.
    figure_files = list(out.glob(stem + ".*"))
    durable_bytes = size(out) + sum(p.stat().st_size for p in figure_files)
    if durable_bytes > caps["durable_bytes"] or size(stage) > caps["scratch_bytes"]:
        raise RuntimeError("Final publication exceeds approved storage")
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists():
        raise RuntimeError("An earlier partial publication requires explicit repair")
    shutil.copytree(out, partial)
    for path in partial.iterdir():
        if not path.is_file() or sha256(path) != sha256(out / path.name):
            raise RuntimeError("Publication copy differs from completed staged output")
        path.chmod(0o400)
    partial.chmod(0o500)
    partial.rename(destination)
    figures = root / "outputs/figures"
    figures.mkdir(exist_ok=True)
    for path in figure_files:
        target = figures / path.name
        if target.exists():
            raise RuntimeError("QC figure already exists: avoid overwriting history")
        shutil.copyfile(path, target)
        target.chmod(0o400)
    shutil.rmtree(stage)
    print(json.dumps({"result": "complete", "destination": str(destination),
                      "durable_bytes_including_figure_delivery": durable_bytes}), flush=True)


if __name__ == "__main__":
    main()
