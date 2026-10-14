"""Shared Stage 1 setup: saved B2/2001, the retained attempt002 noise draw.

Only the existing development readers and recipe builders are called, with
the arguments of run_conditional_preflight_cost.py. No other world, final
model, Rat3, held-aside or outcome material is opened. No electrical state is
advanced: the array is assembled only to bind the noise ports, as in the
legacy preflight.
"""
import os
from pathlib import Path
import resource
import socket
import time
from types import SimpleNamespace


EPISODE_ROOT = Path(__file__).resolve().parents[2]
WORLD_PATH = EPISODE_ROOT / "outputs/runs/development_worlds_job_47085635/panel/development_worlds.h5"
NOISE_PATH = EPISODE_ROOT / "outputs/runs/model_specification/conditional_forward_cost_attempt002/conditional_noise_draw.npz"
COMPONENT_REPORT = (EPISODE_ROOT / "outputs/runs/model_specification/conditional_component_cost_attempt004"
                    / "component_cost_report.json")
RUN_ROOT = EPISODE_ROOT / "outputs/runs/fast_forward_stage1/planner_readout"
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")


def compute_context():
    """Refuse login nodes; record the allocation before any numerical import."""
    hostname = socket.gethostname()
    job = os.environ.get("SLURM_JOB_ID", "")
    cpus = os.environ.get("SLURM_CPUS_PER_TASK", "")
    if not job.isdecimal() or "-ln" in hostname or not hostname.startswith("sh"):
        raise RuntimeError("Stage 1 execution requires a Sherlock compute allocation")
    for key in THREAD_VARIABLES:
        if os.environ.get(key) != cpus:
            raise RuntimeError(f"{key} must equal SLURM_CPUS_PER_TASK before numerical imports")
    return dict(hostname=hostname, job_id=job, cpus_per_task=cpus,
                memory_mib=os.environ.get("SLURM_MEM_PER_NODE"))


def peak_rss_kib():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def current_rss_kib():
    try:
        with open("/proc/self/statm", encoding="ascii") as stream:
            pages = int(stream.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") // 1024
    except (OSError, ValueError, IndexError):
        return None


class PhaseClock:
    """Wall time per phase plus RSS observations at phase boundaries."""

    def __init__(self, deadline_seconds):
        self.started = time.monotonic()
        self.deadline = float(deadline_seconds)
        self.phase, self._phase_started = "boundary", self.started
        self.timings, self.rss = {}, {}

    def check(self):
        if time.monotonic() - self.started >= self.deadline:
            raise TimeoutError("internal deadline (90% of the Slurm limit) reached")

    def set(self, name):
        now = time.monotonic()
        self.timings[self.phase] = self.timings.get(self.phase, 0.) + now - self._phase_started
        self.rss[self.phase] = dict(peak_kib=peak_rss_kib(), current_kib=current_rss_kib())
        self.phase, self._phase_started = str(name), now
        self.check()

    def close(self):
        self.set("terminal")
        return dict(self.timings)

    @property
    def elapsed(self):
        return time.monotonic() - self.started


def build_case(clock):
    """Legacy setup exactly as in the October 9 preflight driver."""
    check = clock.check
    clock.set("scientific imports")
    from conditional_acquisition_recipe import (build_conditional_frontend, build_conditional_schedule,
        CYCLE_COUNT, CYCLE_TICKS, SLOT_TICKS, HALF_MICROSECOND_TICKS_PER_SECOND)
    from conditional_forward_recipe import (build_conditional_circuit_parameters, build_conditional_noise,
                                            expected_state_ids)
    from cpe_array import NeurocamCPEArray
    from development_simulation import load_saved_development_world
    from frozen_cubic_noise_binding import (FrozenOriginalCellClock, FrozenCubicNoiseBinding,
                                            bind_frozen_cubic_noise)
    from raw_world_time import RawWorldTime
    from streaming_array_acquisition import StreamingArrayLimits
    from surface_to_contact import build_contact_factors, registered_pad_geometry

    clock.set("loading exactly saved B2 seed 2001")
    world = load_saved_development_world(WORLD_PATH, condition_id="B2", replicate_seed=2001,
        max_elements=2000000, max_array_bytes=64 * 1024**2, max_metadata_bytes=1024**2, check=check)
    clock.set("contact factors and raw time")
    geometry = registered_pad_geometry("G0", short_axis="x", max_elements=2000000, check=check)
    contacts = build_contact_factors(world, geometry, max_elements=2000000, check=check)
    raw_time = RawWorldTime(world, max_elements=2000000, check=check)
    clock.set("legacy circuit parameters and validated 4-ms template")
    frontend = build_conditional_frontend()
    parameters = build_conditional_circuit_parameters(daq_frontend=frontend, check=check)
    schedule = build_conditional_schedule(frontend, state_ids=expected_state_ids(), check=check)
    if schedule.acquisition_window != (0., 16.):
        raise ValueError("the complete original physical record is required")
    clock.set("electrical assembly for noise-port binding (no advance)")
    array = NeurocamCPEArray(**dict(parameters))
    if array.state_ids != expected_state_ids() or len(array.state_ids) != 69848:
        raise ValueError("all 69,848 original circuit states are required")
    initial_state, initial_field = array.volts, array.previous_field_volts
    clock.set("original noise replay and owned spectral-clock binding")
    cell_clock = FrozenOriginalCellClock(CYCLE_COUNT, CYCLE_TICKS, SLOT_TICKS, HALF_MICROSECOND_TICKS_PER_SECOND)
    noise = build_conditional_noise(array=array, check=check, retained_draw_path=None,
        current_hold_tolerance_amp=1e-12, voltage_hold_tolerance_volts=1e-8,
        max_evaluations=200000000, max_term_evaluations=200000000000000,
        reuse_draw_path=NOISE_PATH, cubic_cell_for_key=cell_clock)
    noise = bind_frozen_cubic_noise(noise, clock=cell_clock, max_numeric_bytes=512 * 1024**2, check=check)
    if type(noise) is not FrozenCubicNoiseBinding:
        raise RuntimeError("owned cubic binding unavailable")
    if array.volts != initial_state or array.previous_field_volts != initial_field:
        raise RuntimeError("noise binding must not advance the array")
    limits = StreamingArrayLimits(max_windows=4000, max_input_bindings=24,
        max_physical_events=40000000, max_total_fragments=40000000,
        max_window_fragments=200000, max_split_intervals=40000000,
        max_window_split_intervals=200000, max_stored_window_numeric_bytes=128 * 1024**2,
        max_window_digest_bytes=4000 * 32, max_total_operator_products=1000000000)
    return SimpleNamespace(world=world, geometry=geometry, contacts=contacts, raw_time=raw_time,
                           frontend=frontend, parameters=parameters, schedule=schedule, array=array,
                           noise=noise, cell_clock=cell_clock, limits=limits,
                           initial_state=initial_state, initial_field=initial_field)


def build_plan(case, clock):
    """New O1 planner on the validated legacy template; returns plan and totals."""
    from fast_io_planner import FullRecordPlan, template_from_schedule
    clock.set("O1 template arrays")
    template = template_from_schedule(case.schedule, case.frontend)
    clock.set("O1 descriptor digests")
    plan = FullRecordPlan(template=template, raw_time=case.raw_time, contact_factors=case.contacts,
                          noise=case.noise, epsilon_input_volts=1e-8, max_intervals=40000000,
                          max_work=100000000, check=clock.check)
    clock.set("O1 full-record structural validation and closed-form holds")
    totals = plan.validate()
    return plan, totals


def assert_not_advanced(case):
    if case.array.volts != case.initial_state or case.array.previous_field_volts != case.initial_field:
        raise RuntimeError("planning must preserve the original electrical state and field")
