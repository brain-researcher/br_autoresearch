#!/usr/bin/env python3
"""Join the fixed candidate-map list to producer metadata and subject outputs.

Only exact statistics-model/dataset-description JSON files are opened. Subject
effect-map paths are enumerated at two declared depths and statted, never read.
Output availability is not actual group inclusion: actual contributor IDs/N
remain null because this source interface has no fitted group-input manifest.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time

from census_group_map_metadata import camel_name
from qualify_source_metadata import require_compute, subject_token


MAP_NAME = re.compile(r"^(?P<prefix>.*?)contrast-(?P<contrast>.+)_stat-(?P<stat>[^_]+)_statmap\.nii\.gz$")


def map_identity(path):
    match = MAP_NAME.fullmatch(Path(path).name)
    if not match:
        return None
    entities = {}
    for token in match['prefix'].strip('_').split('_'):
        if '-' in token:
            key, value = token.split('-', 1)
            if key in entities:
                raise ValueError(f"Duplicate filename entity in {Path(path).name}")
            entities[key] = value
    return {"contrast": match['contrast'], "stat": match['stat'], "entities": entities}


def summarize_support(planned, rows):
    """Keep configured IDs, available outputs and actual inclusion distinct."""
    present = defaultdict(list)
    missing_payload = []
    for row in rows:
        subject = row['entities'].get('sub')
        if subject is None:
            continue
        if row['target_file_present']:
            present[subject_token(subject)].append(row)
        else:
            missing_payload.append(row['path'])
    configured = None if planned is None else sorted({subject_token(x) for x in planned})
    supported = sorted(present)
    return {
        "configured_subject_ids_original": planned,
        "configured_subject_ids_numeric_padding_normalized": configured,
        "configured_N": None if configured is None else len(configured),
        "source_output_supported_subject_ids": supported,
        "source_output_supported_subject_count": len(supported),
        "configured_without_source_output_support": None if configured is None else sorted(set(configured)-set(supported)),
        "source_output_subjects_outside_configured_input": None if configured is None else sorted(set(supported)-set(configured)),
        "subject_ids_with_multiple_matching_effect_paths": sorted(key for key, value in present.items() if len(value) != 1),
        "unavailable_annex_or_file_paths": missing_payload,
        "actual_group_contributor_ids": None,
        "actual_group_N": None,
        "actual_group_inclusion_status": "unknown_no_direct_fitted_group_input_record",
        "source_output_support_is_actual_inclusion": False,
    }


def producer_record(task_root):
    path = task_root/'dataset_description.json'
    if not path.is_file():
        return {"metadata_path": str(path), "status": "missing"}
    metadata = json.loads(path.read_text())
    pipeline = metadata.get('PipelineDescription', {})
    params = pipeline.get('Parameters', {})
    return {
        "metadata_path": str(path), "status": "provider_metadata_present",
        "name": pipeline.get('Name'), "version": pipeline.get('Version'),
        "code_url": pipeline.get('CodeURL', metadata.get('CodeURL')),
        "parameters": {key: params.get(key) for key in (
            'analysis_level', 'model', 'participant_label', 'derivatives',
            'space', 'smoothing', 'estimator', 'drop_missing')},
        "source_dataset_urls": metadata.get('SourceDatasetsURLs'),
        "license": metadata.get('License'),
        "copied_lock_is_executed_version": False,
    }


def subject_outputs(task_root, spec):
    rows, node_records = [], []
    for node in spec.get('Nodes', []):
        if str(node.get('Level', '')).lower() != 'subject':
            continue
        name = node['Name']
        if Path(name).name != name:
            raise ValueError('Invalid declared node name')
        roots = sorted({task_root/f'node-{name}', task_root/f'node-{camel_name(name)}'})
        node_records.append({"name": name, "group_by": node.get('GroupBy'),
                             "model": node.get('Model'),
                             "dummy_contrasts": node.get('DummyContrasts'),
                             "paths_checked": [str(root) for root in roots],
                             "existing_paths": [str(root) for root in roots if root.is_dir()]})
        for root in roots:
            if not root.is_dir():
                continue
            # No **, rglob, reports, run-node descent or arbitrary recursion.
            paths = set(root.glob('sub-*/*_stat-effect_statmap.nii.gz'))
            paths.update(root.glob('sub-*/ses-*/*_stat-effect_statmap.nii.gz'))
            for path in sorted(paths):
                identity = map_identity(path)
                if identity is None or identity['stat'] != 'effect':
                    continue
                rows.append({"path": str(path), "declared_node": name,
                             "target_file_present": path.is_file(), **identity})
    return rows, node_records


def process_task(task, analysis_root):
    task_root = Path(task['task_directory'])
    expected = analysis_root/task['dataset_id']/f"task-{task['task']}"
    if task_root != expected:
        raise ValueError('Manifest task route does not equal the selected analysis root')
    specs = task['existing_spec_paths']
    if len(specs) != 1:
        raise ValueError('Exactly one previously identified source model is required')
    spec_path = Path(specs[0])
    spec_root = analysis_root/task['dataset_id']/'code/openneuro_glmfitlins/statsmodel_specs'/task['dataset_id']
    if spec_path.parent != spec_root:
        raise ValueError('Source model outside the exact declared source location')
    spec = json.loads(spec_path.read_text())
    provider = producer_record(task_root)
    available, subject_nodes = subject_outputs(task_root, spec)
    nodes = {node['Name']: node for node in spec.get('Nodes', [])}
    definitions = defaultdict(list)
    for node in spec.get('Nodes', []):
        if str(node.get('Level', '')).lower() == 'run':
            for definition in node.get('Contrasts', []):
                definitions[definition.get('Name')].append(definition)
    comparisons = {row['contrast']: row for row in task['contrast_definition_comparison']}
    configured = spec.get('Input', {}).get('subject')
    if configured is not None and not isinstance(configured, list):
        raise ValueError('Configured subject field is not a list or null')
    maps = []
    for candidate in task['map_filename_candidates']:
        path = Path(candidate['path'])
        identity = map_identity(path)
        if identity is None or identity['stat'] != 'z' or identity['contrast'] != candidate['contrast']:
            raise ValueError('Previously selected candidate filename no longer matches its identity')
        group_node = nodes[candidate['declared_node_name']]
        group_roots = {task_root/f"node-{group_node['Name']}", task_root/f"node-{camel_name(group_node['Name'])}"}
        if not (path.parent in group_roots or
                (path.parent.name.startswith('ses-') and path.parent.parent in group_roots)):
            raise ValueError('Candidate map is outside the exact declared node depths')
        group_by = group_node.get('GroupBy', [])
        unknown_group_fields = sorted(set(group_by)-{'contrast', 'session'})
        session_identity_resolved = (('session' in group_by) == ('ses' in identity['entities']))
        matches = []
        for output in available:
            if not session_identity_resolved:
                continue
            if output['contrast'] != identity['contrast']:
                continue
            # The selected group map's session, when present, must match.
            if ('session' in group_by and
                    identity['entities'].get('ses') != output['entities'].get('ses')):
                continue
            matches.append(output)
        original = definitions.get(candidate['contrast'], [])
        comparison = comparisons.get(candidate['contrast'], {})
        sign_definition = original[0] if len(original) == 1 else None
        signed_t = bool(sign_definition and str(sign_definition.get('Test', '')).lower() == 't'
                        and comparison.get('exact_conditions_and_weights_match') is True
                        and sign_definition.get('ConditionList') == comparison.get('receipt_conditions')
                        and sign_definition.get('Weights') == comparison.get('receipt_weights'))
        group_mean_t = (str(group_node.get('Level', '')).lower() == 'dataset'
                        and group_node.get('Model', {}).get('X') == [1]
                        and str(group_node.get('Model', {}).get('Type', '')).lower() == 'glm'
                        and str(group_node.get('DummyContrasts', {}).get('Test', '')).lower() == 't'
                        and 'contrast' in group_by
                        and session_identity_resolved and not unknown_group_fields)
        subject_mean_t = (len(subject_nodes) == 1
                          and {'subject', 'contrast'}.issubset(set(subject_nodes[0].get('group_by') or []))
                          and not (set(subject_nodes[0].get('group_by') or [])-{'subject', 'contrast', 'session'})
                          and subject_nodes[0].get('model', {}).get('X') == [1]
                          and str(subject_nodes[0].get('model', {}).get('Type', '')).lower() == 'glm'
                          and str(subject_nodes[0].get('dummy_contrasts', {}).get('Test', '')).lower() == 't')
        simple_chain = ([str(node.get('Level', '')).lower() for node in spec.get('Nodes', [])]
                        == ['run', 'subject', 'dataset'] and not spec.get('Edges'))
        declared_signed_z = (signed_t and group_mean_t and subject_mean_t and simple_chain
                             and provider.get('name') == 'FitLins'
                             and provider.get('parameters', {}).get('estimator') == 'nilearn')
        support = summarize_support(configured, matches)
        maps.append({
            "dataset_id": task['dataset_id'], "task": task['task'],
            "contrast": candidate['contrast'], "session": identity['entities'].get('ses'),
            "group_map_path": str(path), "group_node": group_node['Name'],
            "group_by": group_by, "unresolved_group_by_fields": unknown_group_fields,
            "map_session_identity_matches_grouping": session_identity_resolved,
            "source_signed_definition": sign_definition,
            "source_signed_definition_matches_design_receipt": signed_t,
            "simple_positive_mean_subject_to_dataset_chain": bool(subject_mean_t and group_mean_t and simple_chain),
            "provider_declared_statistic_status": ('provider_declared_signed_group_z'
                if declared_signed_z else 'source_statistic_or_sign_route_unresolved'),
            "statistic_verified_from_voxels_or_nifti_intent": False,
            "threshold_status": ('provider_declared_raw_fitlins_statmap_not_thresholded_report_png'
                if declared_signed_z else 'unknown'),
            "canonical_polarity": None,
            "matched_subject_output_records": matches,
            **support,
            "biological_group_independence_authenticated": False,
            "primary_map_eligibility_authenticated": False,
        })
    return {"dataset_id": task['dataset_id'], "task": task['task'],
            "source_model_path": str(spec_path), "producer": provider,
            "subject_nodes": subject_nodes, "maps": maps}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group-map-manifest', required=True, type=Path)
    parser.add_argument('--analysis-root', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args(argv)
    require_compute()
    if args.output_dir.exists():
        raise FileExistsError('Preserve prior attempts; choose a fresh output directory')
    started = time.monotonic()
    manifest = json.loads(args.group_map_manifest.read_text())
    if manifest['explicit_dataset_task_pairs'] != 46 or manifest['candidate_group_Z_filenames'] != 297:
        raise ValueError('This selected step is bound to the already observed 297-map/46-route manifest')
    tasks, errors = [], []
    for task in manifest['tasks']:
        try:
            tasks.append(process_task(task, args.analysis_root))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append({"dataset_id": task['dataset_id'], "task": task['task'],
                           "error": f'{type(exc).__name__}: {exc}'})
    maps = [row for task in tasks for row in task['maps']]
    versions = Counter(task['producer'].get('version', 'unknown') for task in tasks)
    report = {
        "analysis_id": "source_handoff_qualification",
        "status": "source_output_support_join_completed_actual_group_inclusion_unresolved" if not errors else 'source_output_support_join_incomplete',
        "job_id": os.environ['SLURM_JOB_ID'],
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic()-started,
        "source_manifest": str(args.group_map_manifest),
        "task_routes_processed": len(tasks), "candidate_maps_processed": len(maps),
        "executed_provider_versions_by_route": dict(versions),
        "provider_declared_signed_group_z_count": sum(row['provider_declared_statistic_status'] == 'provider_declared_signed_group_z' for row in maps),
        "maps_with_source_output_support": sum(row['source_output_supported_subject_count'] > 0 for row in maps),
        "maps_with_configured_subject_output_gaps": sum(bool(row['configured_without_source_output_support']) for row in maps),
        "maps_with_multiple_effect_paths_per_subject": sum(bool(row['subject_ids_with_multiple_matching_effect_paths']) for row in maps),
        "actual_group_N_authenticated_count": 0,
        "actual_contributor_dependency": 'Direct fitted Dataset-node inclusion record keyed to map contrast/session, with subject IDs and derivative paths; configured subjects and existing subject effects alone do not establish inclusion under drop_missing.',
        "tasks": tasks, "errors": errors,
        "numeric_padding_rule": 'Within one accession, decimal subject labels are compared after removal of zero padding; original IDs and all exact paths are retained. No cross-accession identity inference.',
        "payload_or_png_values_opened": False,
        "group_map_payloads_or_headers_opened": False,
        "full_577_program_handoff_equivalence": False,
        "candidate_handoff_qualified": False,
        "audit_opened": False,
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'group_contributor_support.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: value for key, value in report.items() if key not in {'tasks'}}, indent=2), flush=True)
    return 2 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
