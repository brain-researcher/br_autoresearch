"""EP06's frozen mean/median-ridge and electrode-PCA2 mapping branches.

This array-only primitive has no reader, profile scorer, classifier, subset
search, frequency combiner or opening authority. The caller admits the session,
native window and complete trial roles, supplies exactly 15 selected physical
electrodes, and matches support/rank/capacity across M1 and PMd. Those duties
and the EP07 freeze/opening prerequisite cannot be proved by metadata here.

PCA uses unresidualized training spikes, centered across equally weighted
whole-trial/native-bin rows. Loadings are orthonormal; each column's largest
absolute loading is assigned positive sign. SVD bases inside tied subspaces
need not be identical across numerical libraries. Numerical rank uses
eps64*max(rows,neurons)*largest_singular_value; a deficient requested rank
fails without changing rank or dropping a neuron. The explicit rank and
positive ridge penalty are candidate choices, not a selected search schedule.

Primary targets are (spikes - training_direction_bin_mean) @ PCA_loadings.
The PCA grand mean is NOT subtracted again. This training-template residual
retains evaluation-versus-training cell-mean shifts; it is not component E's
evaluation-centered residual. Prediction returns residual coordinates only.

Every trial/bin retains each of the nine native classes separately. Exactly 15
supplied electrodes are averaged or median-pooled for each class. Each resulting predictor is
centered/scaled using training rows (population SD, ddof0); zero SD has scale 1.
Blocks concatenate their declared bands. A block is a singleton, a contiguous
canonical-order group of power bands, or the full nine-class profile. LMP has
no frequency interval, so it is singleton or part of the full profile only.
Block declarations are prospective caller choices, never inferred from data.
Ridge minimizes .5*mean_rows||residual-X@B-intercept||² + .5*penalty||B||²,
using an unpenalized intercept and float64 augmented least squares.

The separate CCA branch requires population rank2 and all nine singleton
blocks. Within each band, standardize the15 electrodes by training mean and
ddof0 SD (zero SD becomes1), center again at numerical precision, and retain
the leading two electrode PCs with the same rank/sign convention as spike PCA.
The reviewed cca_mapping primitive fits one regularized canonical score to
both training-template residual coordinates. Its canonical loading scale is
unchanged. No lower-rank fallback, extra mode, normalization or band selection
is introduced; residual targets may have rank0/1 despite population rank2.
The PCA2/ridge successor retains that exact per-band electrode transform and
fits both electrode PCs directly to both residual population PCs with the ridge
objective above. It does not construct a canonical score or rescale either PC.
Both PCA2 branches require the full nine singleton blocks; a caller may later
select the declared low-frequency profile without refitting. All branches
share only their genuine frozen population/template context.

BandwiseLFP.design names NATIVE neuron axes as metadata only, without receiving
spike values. Every input binds session/region, whole trials, directions, native
bins, and native axes before value access. Prediction also binds the exact
electrode/band order. Nontraining roles must be disjoint from fitting IDs;
calibration/evaluation separation from each other remains the caller's duty.
"""
from dataclasses import dataclass, replace
import math
from numbers import Integral, Real

import numpy as np

from cca_mapping import FittedCCAMapping, fit_cca_mapping, predict_cca_mapping
from component_scoring import FrozenDesign, BoundActivity, _ids


BANDS = ((0., 0.), (.5, 4.), (4., 8.), (8., 12.), (12., 25.),
         (25., 50.), (50., 100.), (100., 200.), (200., 400.))


@dataclass(frozen=True)
class FeatureBlock:
    name: str
    bands: tuple


@dataclass(frozen=True)
class BandwiseRecipe:
    rank: int
    penalty: float
    blocks: tuple[FeatureBlock, ...]
    feature_family: str = "mean_ridge"
    covariance_regularization: float | None = None


@dataclass(frozen=True)
class BandwiseLFP:
    region: str
    design: FrozenDesign
    electrode_ids: tuple
    bands: tuple
    values: object                 # [trial,native_bin,15,9]


@dataclass(frozen=True)
class FittedBlock:
    block: FeatureBlock
    predictor_indices: tuple
    coefficients: np.ndarray       # [bands_in_block,rank]
    intercept: np.ndarray          # [rank]


@dataclass(frozen=True)
class FrozenPopulationContext:
    region: str
    training_design: FrozenDesign  # native neuron axes
    electrode_ids: tuple
    bands: tuple
    recipe: BandwiseRecipe
    population_axis_ids: tuple
    population_mean: np.ndarray   # [neuron], diagnostic origin for PCA only
    loadings: np.ndarray          # [neuron,rank]
    singular_values: np.ndarray
    numerical_rank: int
    rank_tolerance: float
    training_template: np.ndarray # [direction,native_bin,neuron]


@dataclass(frozen=True)
class FittedBandwise(FrozenPopulationContext):
    predictor_mean: np.ndarray    # [9], supplied band order
    predictor_scale: np.ndarray   # [9], supplied band order
    blocks: tuple[FittedBlock, ...]


@dataclass(frozen=True)
class FittedCCAFeature:
    block: FeatureBlock
    predictor_index: int
    electrode_mean: np.ndarray       # [15], training only
    electrode_scale: np.ndarray      # [15], ddof0 and zeroSD1
    pca_mean: np.ndarray             # [15], centered standardized electrodes
    loadings: np.ndarray             # [15,2]
    singular_values: np.ndarray
    numerical_rank: int
    rank_tolerance: float
    mapping: FittedCCAMapping        # genuine canonical score/full2D map


@dataclass(frozen=True)
class FittedCCABandwise(FrozenPopulationContext):
    blocks: tuple[FittedCCAFeature, ...]


@dataclass(frozen=True)
class FittedPCARidgeFeature:
    block: FeatureBlock
    predictor_index: int
    electrode_mean: np.ndarray       # [15], training only
    electrode_scale: np.ndarray      # [15], ddof0 and zeroSD1
    pca_mean: np.ndarray             # [15], centered standardized electrodes
    loadings: np.ndarray             # [15,2]
    singular_values: np.ndarray
    numerical_rank: int
    rank_tolerance: float
    coefficients: np.ndarray         # [2,2], both predictor and target PCs
    intercept: np.ndarray            # [2], unpenalized


@dataclass(frozen=True)
class FittedPCARidgeBandwise(FrozenPopulationContext):
    blocks: tuple[FittedPCARidgeFeature, ...]


@dataclass(frozen=True)
class BandwisePredictions:
    region: str
    design: FrozenDesign          # frozen population axes
    blocks: tuple[FeatureBlock, ...]
    predictions: tuple[BoundActivity, ...]


def _check(check):
    if not callable(check):
        raise TypeError("caller-owned cancellation/resource check required")
    check()


def _readonly(value):
    array = np.ascontiguousarray(value, dtype=np.float64)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _bands(value):
    if (not isinstance(value, tuple) or not value
            or any(not isinstance(band, tuple) or len(band) != 2
                   or any(isinstance(number, (bool, np.bool_)) or not isinstance(number, Real)
                          or not math.isfinite(float(number)) for number in band) for band in value)):
        raise ValueError("explicit ordered registered band identities are required")
    bands = tuple(tuple(float(number) for number in band) for band in value)
    if len(set(bands)) != len(bands) or not set(bands) <= set(BANDS):
        raise ValueError("bands must be unique registered classes, without invented edges")
    return bands


def _recipe(value):
    if (not isinstance(value, BandwiseRecipe)
            or isinstance(value.rank, (bool, np.bool_)) or not isinstance(value.rank, Integral)
            or value.rank < 1 or isinstance(value.penalty, (bool, np.bool_))
            or not isinstance(value.penalty, Real) or not math.isfinite(float(value.penalty))
            or value.penalty <= 0 or not isinstance(value.blocks, tuple) or not value.blocks):
        raise ValueError("explicit positive rank, positive ridge penalty and ordered blocks required")
    blocks = []
    for block in value.blocks:
        if not isinstance(block, FeatureBlock) or not isinstance(block.name, str) or not block.name:
            raise ValueError("each declared block requires an explicit name")
        bands = _bands(block.bands)
        start = BANDS.index(bands[0])
        if (len(bands) > 1 and bands != BANDS
                and (start == 0 or bands != BANDS[start:start + len(bands)])):
            raise ValueError("groups must be contiguous power bands; LMP is singleton or full-profile only")
        blocks.append(FeatureBlock(block.name, bands))
    if len({block.name for block in blocks}) != len(blocks):
        raise ValueError("feature block names must be unique")
    if value.feature_family not in ("mean_ridge", "median_ridge", "pca2_cca", "pca2_ridge"):
        raise ValueError("an explicit supported feature family is required")
    regularization = value.covariance_regularization
    if value.feature_family in ("pca2_cca", "pca2_ridge"):
        if (value.rank != 2 or len(blocks) != 9 or any(len(block.bands) != 1 for block in blocks)
                or {block.bands[0] for block in blocks} != set(BANDS)):
            raise ValueError("PCA2 mapping requires population rank2 and all nine singleton blocks")
    if value.feature_family == "pca2_cca":
        if (isinstance(regularization, (bool, np.bool_)) or not isinstance(regularization, Real)
                or not math.isfinite(float(regularization)) or regularization <= 0):
            raise ValueError("CCA covariance regularization must be explicit positive and finite")
        regularization = float(regularization)
    elif regularization is not None:
        raise ValueError("ridge recipes cannot carry unused CCA regularization")
    return BandwiseRecipe(int(value.rank), float(value.penalty), tuple(blocks),
                          value.feature_family, regularization)


def _design(value):
    if not isinstance(value, FrozenDesign) or len(value.directions) != 8:
        raise ValueError("an explicit native design with all eight reach directions is required")
    return value


def _lfp_metadata(value):
    if not isinstance(value, BandwiseLFP) or value.region not in ("M1", "PMd"):
        raise ValueError("one explicitly bound M1 or PMd LFP pack is required")
    design = _design(value.design)
    if not isinstance(value.electrode_ids, tuple):
        raise ValueError("electrode order must be an immutable tuple")
    electrodes = _ids(value.electrode_ids, "electrode_ids")
    bands = _bands(value.bands)
    if len(electrodes) != 15 or len(bands) != 9 or set(bands) != set(BANDS):
        raise ValueError("all nine bands and exactly 15 supplied physical electrodes required")
    return design, electrodes, bands


def _observed_metadata(value):
    if not isinstance(value, BoundActivity):
        raise TypeError("native observations need their explicit BoundActivity design")
    return _design(value.design)


def _array(value, shape, *, counts=False):
    raw = np.asarray(value)
    if (raw.dtype.kind not in "fiu" or raw.shape != shape or not np.isfinite(raw).all()
            or (counts and (np.any(raw < 0) or np.any(raw != np.floor(raw))))):
        raise ValueError("finite complete native arrays required; spikes must be nonnegative integer counts")
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.array(raw, dtype=np.float64, copy=True)
    return _finite(array)


def _finite(value):
    if not np.isfinite(value).all():
        raise FloatingPointError("nonfinite bandwise arithmetic; no band, neuron or trial was dropped")
    return value


def _lfp_values(lfp):
    shape = (len(lfp.design.trial_ids), len(lfp.design.time_ids), 15, 9)
    return _array(lfp.values, shape)


def _electrode_summary(values, family):
    with np.errstate(over="ignore", invalid="ignore"):
        if family == "mean_ridge":
            return _finite(np.mean(values, axis=2))
        if family == "median_ridge":
            return _finite(np.median(values, axis=2))
    raise ValueError("an explicit mean/median ridge family is required")


def _pca(values, rank, label):
    rows, columns = values.shape
    mean = _finite(np.mean(values, axis=0))
    centered = _finite(values - mean)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    _finite(singular_values)
    _finite(vt)
    tolerance = float(np.finfo(np.float64).eps * max(rows, columns) * singular_values[0])
    numerical_rank = int(np.count_nonzero(singular_values > tolerance))
    if numerical_rank < rank:
        raise ValueError(f"requested {label} rank {rank} exceeds numerical training rank {numerical_rank}")
    loadings = vt[:rank].T.copy()
    for column in range(rank):
        pivot = int(np.argmax(np.abs(loadings[:, column])))
        if loadings[pivot, column] < 0:
            loadings[:, column] *= -1
    return mean, loadings, singular_values, numerical_rank, tolerance


def _direction_rows(design):
    return tuple(np.asarray([index for index, label in enumerate(design.trial_directions)
                             if label == direction], dtype=np.intp) for direction in design.directions)


def _residual(values, design, template, loadings):
    result = np.empty((len(design.trial_ids), len(design.time_ids), loadings.shape[1]))
    with np.errstate(over="ignore", invalid="ignore"):
        for direction, rows in enumerate(_direction_rows(design)):
            # Residuals have a zero origin. PCA's unresidualized grand mean
            # belongs to fitting its loadings, not to this transformation.
            result[rows] = (values[rows] - template[direction]) @ loadings
    return _finite(result)


def _fit_pc_ridge(scores, target, penalty):
    """Both PCs, mean-row loss and unpenalized intercept; no score rescaling."""
    rows, predictors = scores.shape
    with np.errstate(over="ignore", invalid="ignore"):
        x_mean, y_mean = np.mean(scores, axis=0), np.mean(target, axis=0)
        a = np.vstack(((scores - x_mean) / math.sqrt(rows),
                       math.sqrt(penalty) * np.eye(predictors)))
        b = np.vstack(((target - y_mean) / math.sqrt(rows),
                       np.zeros((predictors, target.shape[1]))))
        _finite(a)
        _finite(b)
        coefficients = _finite(np.linalg.lstsq(a, b, rcond=None)[0])
        intercept = _finite(y_mean - x_mean @ coefficients)
    return coefficients, intercept


def fit_bandwise(lfp, observed, *, recipe, check):
    """Fit one independent session/region from its training whole trials only."""
    _check(check)
    recipe = _recipe(recipe)
    design, electrodes, bands = _lfp_metadata(lfp)
    observed_design = _observed_metadata(observed)
    if design.role != "training" or observed_design.role != "training":
        raise PermissionError("population axes, templates, scaling and mappings fit training trials only")
    if observed_design != design:
        raise ValueError("training LFP and native spikes must have identical scope, trials, bins and axes")
    rows, neurons = len(design.trial_ids) * len(design.time_ids), len(design.axis_ids)
    if recipe.rank > min(rows, neurons):
        raise ValueError("requested population rank exceeds supplied training dimensions")
    # Every metadata and role check above precedes either value array.
    lfp_values = _lfp_values(lfp)
    y = _array(observed.values, design.shape, counts=True)
    _check(check)
    with np.errstate(over="ignore", invalid="ignore"):
        population_mean, loadings, singular_values, numerical_rank, rank_tolerance = _pca(
            y.reshape(rows, neurons), recipe.rank, "population")
        template = _finite(np.stack([np.mean(y[indices], axis=0) for indices in _direction_rows(design)]))
        target = _residual(y, design, template, loadings).reshape(rows, recipe.rank)
    common = (lfp.region, design, electrodes, bands, recipe,
        tuple(f"{lfp.region}:training_pc_{index + 1:03d}" for index in range(recipe.rank)),
        _readonly(population_mean), _readonly(loadings), _readonly(singular_values), numerical_rank,
        rank_tolerance, _readonly(template))
    if recipe.feature_family in ("pca2_cca", "pca2_ridge"):
        blocks = []
        for block in recipe.blocks:
            _check(check)
            index = bands.index(block.bands[0])
            x = lfp_values[:, :, :, index].reshape(rows, 15)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                mean = _finite(np.mean(x, axis=0))
                scale = _finite(np.std(x, axis=0, ddof=0))
                scale[scale == 0] = 1.
                standardized = _finite((x - mean) / scale)
                pca_mean, x_loadings, x_singular_values, x_rank, x_tolerance = _pca(
                    standardized, 2, "electrode")
                scores = _finite((standardized - pca_mean) @ x_loadings)
            state = (block, index, _readonly(mean), _readonly(scale),
                _readonly(pca_mean), _readonly(x_loadings), _readonly(x_singular_values),
                x_rank, x_tolerance)
            if recipe.feature_family == "pca2_cca":
                mapping = fit_cca_mapping(scores, target,
                    covariance_regularization=recipe.covariance_regularization,
                    prediction_penalty=recipe.penalty, check=check)
                blocks.append(FittedCCAFeature(*state, mapping))
            else:
                _check(check)
                coefficients, intercept = _fit_pc_ridge(scores, target, recipe.penalty)
                blocks.append(FittedPCARidgeFeature(*state, _readonly(coefficients), _readonly(intercept)))
        _check(check)
        fitted_type = FittedCCABandwise if recipe.feature_family == "pca2_cca" else FittedPCARidgeBandwise
        return fitted_type(*common, tuple(blocks))
    with np.errstate(over="ignore", invalid="ignore"):
        predictors = _electrode_summary(lfp_values, recipe.feature_family).reshape(rows, 9)
        mean = _finite(np.mean(predictors, axis=0))
        scale = _finite(np.std(predictors, axis=0, ddof=0))
        scale[scale == 0] = 1.
        predictors = _finite((predictors - mean) / scale)
    blocks = []
    for block in recipe.blocks:
        _check(check)
        indices = tuple(bands.index(band) for band in block.bands)
        x = predictors[:, indices]
        with np.errstate(over="ignore", invalid="ignore"):
            x_mean, y_mean = np.mean(x, axis=0), np.mean(target, axis=0)
            a = np.vstack(((x - x_mean) / math.sqrt(rows), math.sqrt(recipe.penalty) * np.eye(len(indices))))
            b = np.vstack(((target - y_mean) / math.sqrt(rows), np.zeros((len(indices), recipe.rank))))
            _finite(a)
            _finite(b)
            coefficients = _finite(np.linalg.lstsq(a, b, rcond=None)[0])
            intercept = _finite(y_mean - x_mean @ coefficients)
        blocks.append(FittedBlock(block, indices, _readonly(coefficients), _readonly(intercept)))
    _check(check)
    return FittedBandwise(*common, _readonly(mean), _readonly(scale), tuple(blocks))


def _bind_fitted(fitted, design, region):
    if not isinstance(fitted, (FittedBandwise, FittedCCABandwise, FittedPCARidgeBandwise)):
        raise TypeError("one frozen bandwise training fit is required")
    recipe = _recipe(fitted.recipe)
    if ((isinstance(fitted, FittedCCABandwise)) != (recipe.feature_family == "pca2_cca")
            or (isinstance(fitted, FittedPCARidgeBandwise)) != (recipe.feature_family == "pca2_ridge")):
        raise ValueError("fitted feature state and declared recipe family disagree")
    if tuple(block.block for block in fitted.blocks) != recipe.blocks:
        raise ValueError("fitted feature blocks differ from the explicit recipe")
    design = _design(design)
    trained = fitted.training_design
    if region != fitted.region or design.session_id != trained.session_id:
        raise PermissionError("fitted state cannot cross session or region")
    for field in ("directions", "time_ids", "axis_ids"):
        if getattr(design, field) != getattr(trained, field):
            raise ValueError("direction/native-bin/native-neuron identities differ from the frozen fit")
    if (len(fitted.population_axis_ids) != fitted.recipe.rank
            or fitted.loadings.shape != (len(trained.axis_ids), fitted.recipe.rank)):
        raise ValueError("frozen population axis dimensions are inconsistent")
    if design.role == "training":
        if design != trained:
            raise ValueError("training predictions must retain the exact fitted whole-trial identities")
    elif set(design.trial_ids).intersection(trained.trial_ids):
        raise PermissionError("calibration/evaluation trials overlap fitted training trials")
    return replace(design, axis_ids=fitted.population_axis_ids)


def predict_bandwise(fitted, lfp, *, check):
    """Predict frozen primary residual coordinates from LFP and metadata only."""
    _check(check)
    design, electrodes, bands = _lfp_metadata(lfp)
    output_design = _bind_fitted(fitted, design, lfp.region)
    if electrodes != fitted.electrode_ids or bands != fitted.bands:
        raise ValueError("electrode and band order must match the frozen fitting inputs")
    lfp_values = _lfp_values(lfp)
    predictions = []
    if isinstance(fitted, (FittedCCABandwise, FittedPCARidgeBandwise)):
        rows = len(design.trial_ids) * len(design.time_ids)
        for block in fitted.blocks:
            _check(check)
            x = lfp_values[:, :, :, block.predictor_index].reshape(rows, 15)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                standardized = _finite((x - block.electrode_mean) / block.electrode_scale)
                scores = _finite((standardized - block.pca_mean) @ block.loadings)
            if isinstance(fitted, FittedCCABandwise):
                values = predict_cca_mapping(block.mapping, scores, check=check).reshape(output_design.shape)
            else:
                with np.errstate(over="ignore", invalid="ignore"):
                    values = _finite(scores @ block.coefficients + block.intercept).reshape(output_design.shape)
            predictions.append(BoundActivity(output_design, _readonly(values)))
        _check(check)
        return BandwisePredictions(lfp.region, output_design,
                                  tuple(block.block for block in fitted.blocks), tuple(predictions))
    predictors = _electrode_summary(lfp_values, fitted.recipe.feature_family)
    with np.errstate(over="ignore", invalid="ignore"):
        predictors = _finite((predictors - fitted.predictor_mean) / fitted.predictor_scale)
    for block in fitted.blocks:
        _check(check)
        with np.errstate(over="ignore", invalid="ignore"):
            values = _finite(predictors[:, :, block.predictor_indices] @ block.coefficients + block.intercept)
        predictions.append(BoundActivity(output_design, _readonly(values)))
    _check(check)
    return BandwisePredictions(lfp.region, output_design, tuple(block.block for block in fitted.blocks),
                              tuple(predictions))


def transform_observations(fitted, observed, *, region, check):
    """Apply the training template/loadings without fitting or scoring outcomes."""
    _check(check)
    design = _observed_metadata(observed)
    output_design = _bind_fitted(fitted, design, region)
    values = _array(observed.values, design.shape, counts=True)
    result = _residual(values, design, fitted.training_template, fitted.loadings)
    _check(check)
    return BoundActivity(output_design, _readonly(result))
