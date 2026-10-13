#!/usr/bin/env python3
"""Compute-only bubblewrap feasibility using coordinator-owned fake inputs.

No EEG/image path or external endpoint is opened. This is a runtime sentinel
test, not verification of a real data mount or authorization to compare EEG.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import errno
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time


def child():
    spec = json.loads(Path("/probe_spec.json").read_text())
    checks = {}
    checks["permitted_development_sentinel_readable"] = Path("/inputs/development/allowed.txt").read_text() == "fake-development-only\n"
    forbidden = [spec["host_audit_path"], spec["host_development_path"],
                 f"/proc/{spec['coordinator_pid']}/root{spec['host_audit_path']}",
                 f"/proc/{spec['coordinator_pid']}/fd/{spec['coordinator_audit_fd']}",
                 "/proc/1/root" + spec["host_audit_path"],
                 "/proc/self/root" + spec["host_audit_path"]]
    checks["host_audit_and_process_routes_denied"] = all(not Path(path).exists() for path in forbidden)
    checks["no_host_home_runtime_or_socket_mounts"] = all(not Path(path).exists() for path in ("/home", "/oak", "/scratch", "/run", "/var/run", "/sys"))
    checks["coordinator_environment_not_inherited"] = "EP18_COORDINATOR_SENTINEL" not in os.environ
    status = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in {"CapEff", "CapPrm", "CapBnd", "NoNewPrivs"}:
            status[key] = value.strip()
    checks["no_capabilities_and_no_new_privileges"] = all(int(status[name], 16) == 0 for name in ("CapEff", "CapPrm", "CapBnd")) and status["NoNewPrivs"] == "1"
    namespaces = {name: os.readlink(f"/proc/self/ns/{name}") for name in ("mnt", "pid", "net", "user")}
    checks["mount_pid_network_user_namespaces_differ"] = all(namespaces[name] != spec["coordinator_namespaces"][name] for name in namespaces)
    descriptors = {}
    for path in Path("/proc/self/fd").iterdir():
        try:
            descriptors[path.name] = os.readlink(path)
        except FileNotFoundError:
            continue
    checks["no_host_file_or_socket_descriptors"] = all(not value.startswith((spec["host_temporary_root"], "socket:")) for value in descriptors.values())
    try:
        with Path("/inputs/development/allowed.txt").open("a") as stream:
            stream.write("unexpected-child-write\n")
        write_denied = False
    except OSError as error:
        write_denied = error.errno in (errno.EROFS, errno.EACCES, errno.EPERM)
    checks["development_mount_readonly"] = write_denied
    libc = ctypes.CDLL(None, use_errno=True)
    ctypes.set_errno(0)
    nested_result = libc.unshare(ctypes.c_int(0x10000000))  # CLONE_NEWUSER
    nested_errno = ctypes.get_errno()
    checks["nested_user_namespace_denied"] = nested_result == -1 and nested_errno in (errno.EPERM, errno.EACCES, errno.ENOSPC)
    ctypes.set_errno(0)
    remount_result = libc.mount(None, b"/inputs/development", None,
                               ctypes.c_ulong(32 | 4096), None)  # MS_REMOUNT | MS_BIND
    remount_errno = ctypes.get_errno()
    checks["child_cannot_remount_development_input"] = remount_result == -1 and remount_errno == errno.EPERM
    network_errno = None
    with socket.socket() as connection:
        connection.settimeout(0.5)
        try:
            connection.connect(("127.0.0.1", spec["coordinator_loopback_port"]))
            network_denied = False
        except OSError as error:
            network_denied = True
            network_errno = error.errno
    checks["coordinator_live_loopback_listener_unreachable"] = network_denied
    print(json.dumps({"checks": checks, "all_checks_pass": all(checks.values()),
                      "child_namespaces": namespaces, "child_security_status": status,
                      "child_fd_targets": descriptors, "nested_userns_errno": nested_errno,
                      "remount_errno": remount_errno, "network_errno": network_errno}))
    return 0 if all(checks.values()) else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Coordinator-owned compute allocation required")
    runtime = shutil.which("bwrap")
    if runtime is None:
        raise RuntimeError("Use the existing system bubblewrap/0.11.2 module; do not install a runtime")
    prefixes = {Path(sys.base_prefix).resolve(), Path(sys.prefix).resolve()}
    software_root = Path("/share/software/user/open")
    scratch_root = args.scratch_dir.resolve()
    for prefix in prefixes:
        system_runtime = prefix == Path("/usr") or Path("/usr") in prefix.parents
        module_runtime = software_root in prefix.parents and len(prefix.relative_to(software_root).parts) >= 2
        overlaps_scratch = prefix == scratch_root or prefix in scratch_root.parents or scratch_root in prefix.parents
        if not (system_runtime or module_runtime) or overlaps_scratch:
            raise RuntimeError("Only the selected system or /share/software/user/open module Python prefix may be mounted; no group/home/Oak/Scratch environment")
    executable = Path(sys.executable).resolve()
    if not any(executable == prefix or prefix in executable.parents for prefix in prefixes):
        raise RuntimeError("Interpreter executable is outside its allowed runtime prefixes")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {"analysis_id": "input_role_preparation", "phase": "isolated_runtime_sentinel_feasibility",
              "slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
              "runtime": runtime, "all_inputs_are_harmless_sentinels": True,
              "scientific_payloads_opened": False, "external_network_requests": 0,
              "shared_infrastructure_modified": False, "audit_seal_verified": False,
              "EEG_candidate_comparison_authorized": False}
    try:
        report["runtime_version"] = subprocess.check_output([runtime, "--version"], text=True, timeout=5).strip()
        with tempfile.TemporaryDirectory(prefix="ep18-runtime-sentinel-", dir=args.scratch_dir) as temporary:
            base = Path(temporary)
            development, evaluator = base / "development", base / "evaluator"
            development.mkdir()
            evaluator.mkdir()
            allowed = development / "allowed.txt"
            audit = evaluator / "audit.txt"
            allowed.write_text("fake-development-only\n")
            audit.write_text("fake-audit-not-mounted\n")
            with audit.open() as audit_fd, socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                listener.listen(2)
                port = listener.getsockname()[1]
                # Confirm the positive endpoint in the coordinator's namespace.
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    accepted, _ = listener.accept()
                    accepted.close()
                spec = {"host_audit_path": str(audit), "host_development_path": str(allowed),
                        "host_temporary_root": str(base), "coordinator_pid": os.getpid(),
                        "coordinator_audit_fd": audit_fd.fileno(), "coordinator_loopback_port": port,
                        "coordinator_namespaces": {name: os.readlink(f"/proc/self/ns/{name}") for name in ("mnt", "pid", "net", "user")}}
                specification = base / "probe_spec.json"
                specification.write_text(json.dumps(spec))
                command = [runtime, "--unshare-all", "--unshare-user", "--disable-userns", "--assert-userns-disabled",
                           "--cap-drop", "ALL", "--new-session", "--die-with-parent", "--clearenv",
                           "--setenv", "LD_LIBRARY_PATH", str(Path(sys.base_prefix).resolve() / "lib")]
                roots = {"/usr", "/bin", "/lib", "/lib64", *(str(prefix) for prefix in prefixes)}
                for root in sorted(roots):
                    if Path(root).exists() and root != "/":
                        command += ["--ro-bind", root, root]
                command += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
                            "--ro-bind", str(development), "/inputs/development",
                            "--ro-bind", str(Path(__file__).resolve()), "/probe.py",
                            "--ro-bind", str(specification), "/probe_spec.json",
                            "--chdir", "/", "--", str(Path(sys.executable).resolve()), "/probe.py", "--child"]
                environment = dict(os.environ, EP18_COORDINATOR_SENTINEL="must-not-reach-child")
                completed = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                                           text=True, close_fds=True, env=environment, timeout=30)
                report.update(returncode=completed.returncode, stderr=completed.stderr,
                              command=command, coordinator_loopback_positive_control=True)
                if completed.stdout.strip():
                    report["child_report"] = json.loads(completed.stdout)
                report["sentinel_feasibility_pass"] = completed.returncode == 0 and report.get("child_report", {}).get("all_checks_pass", False)
                report["coordinator_development_sentinel_unchanged"] = allowed.read_text() == "fake-development-only\n"
                report["sentinel_feasibility_pass"] &= report["coordinator_development_sentinel_unchanged"]
    except Exception as error:
        report.update(sentinel_feasibility_pass=False, error=f"{type(error).__name__}: {error}")
    report.update(observed_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started,
                  claim_limit="Only the tested sentinel mount/runtime is qualified. A real data worker needs the same boundary and explicit allowed mount list, dependencies and controlled outputs before the actual seal can be claimed.")
    (args.output_dir / "runtime_feasibility.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["sentinel_feasibility_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(child() if sys.argv[1:] == ["--child"] else main())
