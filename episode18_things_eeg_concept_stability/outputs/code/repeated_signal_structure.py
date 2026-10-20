"""Model-free signal diagnostics for the unchanged development repeated EEG.

No reader, model fit, raw detrending, filter change or inference occurs here.
Every participant retains 12 physical sequences, 200 identities and 62 channels.
Diagnostics use the same fixed 61 sensor contrasts and scoring support as the
registered control. The fixed 37-image alignment shift is a descriptive
comparison, not a registered label/negative-lag control or a primary rescue.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import periodogram

from repeated_image_control import _roster
from training_transforms import CommonSensorSpace


SCHEMA = "ep18_repeated_signal_structure_v1"
BANDS = ("below_1Hz", "from_1_through_40Hz", "above_40Hz")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _sequence_metrics(contrasts, item, check):
    """Descriptive moments and Parseval-normalized power on fixed score rows."""
    check()
    mask = np.asarray(item.retained_mask)
    selected = np.flatnonzero(mask)
    times = np.asarray(item.sample_times_seconds)[selected]
    require(len(selected) >= 2 and np.all(np.diff(selected) == 1)
            and np.allclose(np.diff(times), .004, rtol=0., atol=1e-10),
            "spectral diagnostic requires the complete contiguous fixed 250-Hz score grid")
    values = np.asarray(contrasts)[selected]
    mean = values.mean(axis=0)
    centered = values - mean
    dc = float(mean @ mean)
    variance = float(np.mean(np.sum(centered * centered, axis=1)))
    total = float(np.mean(np.sum(values * values, axis=1)))
    frequencies, psd = periodogram(centered, fs=250., window="boxcar", nfft=len(values),
        detrend=False, return_onesided=True, scaling="density", axis=0)
    df = 250. / len(values)
    power = psd.sum(axis=1) * df
    masks = ((frequencies > 0) & (frequencies < 1),
             (frequencies >= 1) & (frequencies <= 40),
             (frequencies > 40) & (frequencies <= 125))
    sums = {name: float(power[keep].sum()) for name, keep in zip(BANDS, masks)}
    require(np.isfinite(power).all() and np.all(power >= 0), "nonfinite spectral power")
    check()
    return dict(sequence_id=item.sequence_id, ordinal=int(item.ordinal), retained_samples=len(values),
        sample_rate_hz=250, frequency_bin_width_hz=df,
        mean_square_dc_V2=dc, centered_variance_V2=variance, total_mean_square_V2=total,
        spectral_power_V2=sums,
        spectral_fraction_centered={key: value / variance if variance > 0 else None for key, value in sums.items()},
        spectral_dc_bin_power_V2=float(power[frequencies == 0].sum()),
        spectral_total_power_V2=float(power.sum()),
        spectral_parseval_difference_V2=float(power.sum()-variance),
        moment_decomposition_difference_V2=float(total-dc-variance))


def _event_templates(contrasts, item, check):
    """Exact acquisition-grid bin means, before across-image centering."""
    check()
    times = np.asarray(item.sample_times_seconds)
    onsets = np.asarray(item.event_onsets_seconds)
    ticks = np.rint(times * 1000).astype(np.int64)
    event_ticks = np.rint(onsets * 1000).astype(np.int64)
    require(np.all(np.abs(ticks / 1000. - times) < 1e-10)
            and np.all(np.abs(event_ticks / 1000. - onsets) < 1e-10)
            and np.all(np.diff(ticks) == 4), "unaltered 1000-Hz acquisition ticks and 250-Hz samples required")
    require(contrasts.shape == (len(times), 61) and np.isfinite(contrasts).all(),
            "complete finite 61-contrast sequence required")
    # For every event, bins are [0,20), ... [760,780), [780,800] ms.
    # Search only recorded sample locations: no event snapping/interpolation.
    edges = event_ticks[:, None] + np.arange(41, dtype=np.int64)[None, :] * 20
    lower = np.searchsorted(ticks, edges[:, :-1], side="left")
    upper = np.searchsorted(ticks, edges[:, 1:], side="left")
    upper[:, -1] = np.searchsorted(ticks, edges[:, -1], side="right")
    count = upper - lower
    require(np.all(count > 0) and np.all(lower >= 0) and np.all(upper <= len(ticks)),
            "every image and fixed lag bin requires observed sample support")
    retained_prefix = np.concatenate(([0], np.cumsum(np.asarray(item.retained_mask, dtype=np.int64))))
    require(np.array_equal(retained_prefix[upper]-retained_prefix[lower], count),
            "all image-bin samples must lie in the original complete filter/scoring support")
    prefix = np.vstack((np.zeros((1, 61)), np.cumsum(contrasts, axis=0)))
    in_event_order = (prefix[upper]-prefix[lower]) / count[:, :, None]
    templates = np.empty((200, 40, 61), dtype=np.float64)
    templates[np.asarray(item.image_indices)] = in_event_order
    require(np.isfinite(templates).all(), "nonfinite repeated-image templates")
    check()
    return templates, int(count.min()), int(count.max())


def _cosine(cross, first, second):
    denominator = float(np.sqrt(first) * np.sqrt(second))
    if denominator == 0:
        return None
    value = float(cross / denominator)
    require(np.isfinite(value) and abs(value) <= 1. + 1e-10, "template cosine is not numerically valid")
    return value


def _template_metrics(odd, even):
    require(odd.shape == even.shape == (200, 40, 61)
            and np.isfinite(odd).all() and np.isfinite(even).all(),
            "both complete finite image-aligned half templates are required")
    shifted = np.roll(even, -37, axis=0)
    # Sum sensor-contrast energy; average across images (then lag bins below).
    odd_lag = np.mean(np.sum(odd * odd, axis=2), axis=0)
    even_lag = np.mean(np.sum(even * even, axis=2), axis=0)
    matched_lag = np.mean(np.sum(odd * even, axis=2), axis=0)
    shifted_lag = np.mean(np.sum(odd * shifted, axis=2), axis=0)
    odd_energy, even_energy = float(odd_lag.mean()), float(even_lag.mean())
    matched, shifted_cross = float(matched_lag.mean()), float(shifted_lag.mean())
    cosine = _cosine(matched, odd_energy, even_energy)
    shifted_cosine = _cosine(shifted_cross, odd_energy, even_energy)
    lag_cosines = [_cosine(float(c), float(a), float(b)) for c, a, b in zip(matched_lag, odd_lag, even_lag)]
    lag_shifted = [_cosine(float(c), float(a), float(b)) for c, a, b in zip(shifted_lag, odd_lag, even_lag)]
    return dict(odd_energy_V2=odd_energy, even_energy_V2=even_energy,
        matched_cross_covariance_V2=matched, shift37_cross_covariance_V2=shifted_cross,
        matched_minus_shift37_cross_covariance_V2=matched-shifted_cross,
        matched_cosine=cosine, shift37_cosine=shifted_cosine,
        matched_minus_shift37_cosine=None if cosine is None else cosine-shifted_cosine,
        cosine_defined=cosine is not None,
        by_lag=dict(odd_energy_V2=odd_lag.tolist(), even_energy_V2=even_lag.tolist(),
            matched_cross_covariance_V2=matched_lag.tolist(), shift37_cross_covariance_V2=shifted_lag.tolist(),
            matched_minus_shift37_cross_covariance_V2=(matched_lag-shifted_lag).tolist(),
            matched_cosine=lag_cosines, shift37_cosine=lag_shifted,
            matched_minus_shift37_cosine=[None if a is None else a-b for a, b in zip(lag_cosines, lag_shifted)]))


def _order_metrics(sequences):
    odds, evens = sequences[::2], sequences[1::2]
    coincidence, correlation = [], []
    for first in odds:
        first_order = np.asarray(first.image_indices)
        first_positions = np.argsort(first_order).astype(float) - 99.5
        coincidence.append([])
        correlation.append([])
        for second in evens:
            second_order = np.asarray(second.image_indices)
            second_positions = np.argsort(second_order).astype(float) - 99.5
            coincidence[-1].append(float(np.mean(first_order == second_order)))
            correlation[-1].append(float(first_positions @ second_positions / (first_positions @ first_positions)))
    return dict(cross_parity_same_position_fraction=coincidence,
        cross_parity_position_rank_correlation=correlation,
        odd_sequence_ordinals=[int(row.ordinal) for row in odds],
        even_sequence_ordinals=[int(row.ordinal) for row in evens],
        role="order_coincidence_exposes_possible_temporal_position_confounding_not_an_exclusion_rule")


def measure_structure(person, *, budget_check):
    """Measure all fixed signal diagnostics; no participant selection/inference."""
    sequences = _roster(person, budget_check)
    require(tuple(person.image_ids) == tuple(sorted(person.image_ids)),
            "fixed shift37 requires canonical lexicographic image-ID order")
    sensor = CommonSensorSpace(tuple(sequences[0].channel_order))
    halves = [np.zeros((200, 40, 61)), np.zeros((200, 40, 61))]
    sequence_metrics = []
    bin_min, bin_max = [], []
    for item in sequences:
        budget_check()
        contrasts = sensor.apply(item.sensor_samples, channel_order=item.channel_order)
        sequence_metrics.append(_sequence_metrics(contrasts, item, budget_check))
        templates, minimum, maximum = _event_templates(contrasts, item, budget_check)
        # This removes a shared image mean in the diagnostic templates only.
        # It is not raw EEG detrending or prestimulus baseline correction.
        templates -= templates.mean(axis=0, keepdims=True)
        halves[item.ordinal % 2] += templates / 6.
        bin_min.append(minimum)
        bin_max.append(maximum)
    metrics = _template_metrics(*halves)
    budget_check()
    return dict(schema=SCHEMA, status="complete_repeated_signal_structure",
        participant_id=person.participant_id, physical_sequence_count=12, image_identity_count=200,
        recorded_channel_count=62, sensor_contrast_count=61,
        units=dict(energy="V_squared_summed_over_61_contrasts_then_mean_over_samples_or_images_and_lags",
            spectrum="one_sided_boxcar_periodogram_density_times_df_summed_over_contrasts",
            covariance="signed_cross_half_product_not_a_confidence_interval",
            cosine="normalized_cross_half_inner_product_no_Spearman_Brown_correction"),
        sequence_metrics=sequence_metrics, template_metrics=metrics, image_order=_order_metrics(sequences),
        template_definition=dict(lag_bin_edges_ms=list(range(0, 801, 20)), final_bin_includes_800ms=True,
            sample_bin_minimum_by_sequence=bin_min, sample_bin_maximum_by_sequence=bin_max,
            centering="across200_images_separately_within_each_sequence_lag_sensor_contrast",
            halves="six_chronological_human_odd_and_six_even_sequences",
            shifted_alignment="odd_image_i_with_even_image_(i+37)_mod200_in_canonical_ID_order",
            interpolation=False, event_snapping=False, prestimulus_correction=False),
        data_role="development_repeated_signal_recovery_diagnostic_only",
        inference_performed=False, model_fit_performed=False, raw_EEG_detrending=False,
        original_control_threshold_changed=False, primary_rescue=False, audit_access=False,
        concept_or_false_scores_computed=False,
        interpretation="Descriptive signal structure and alignment diagnostics; no absence-of-concept or registered-control pass claim.")
