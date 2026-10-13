#!/usr/bin/env python3
"""Trusted EP17 launcher for its existing, fixed development image pilot.

Run only after coordinator selection with 1 CPU, 4 GiB, no GPU, one numerical
thread and an external 660-second timeout plus 10-second kill grace. A larger
shared parent may host an actual 1-CPU/4-GiB Slurm step; record the step's
allocation and node. The selected 15-minute step also accommodates focused
checks and ASTRA without changing the pilot's 600-second internal deadline.
Slurm supplies the memory limit. The callable only reports peak RSS; neither
it nor this launcher supplies an additional process-RSS guard. Use the
qualified encoding_model CPython 3.11 runtime and EP17 h5py package target.

Only the launcher reads this acquisition's approved private password notice.
It replaces its process with the existing pilot caller, inheriting a pipe
containing the bare password. No credential enters arguments, environment,
logs or persistent output. This is procedural same-account separation, not
an isolation mechanism. No EP18 archive, password or feature artifact is used.
No plaintext image extraction, neural access, reader release or retry occurs.
Select another attempt only with --attempt pilotNNN; omission keeps pilot001.
Existing attempts are refused, never replaced or advanced automatically.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import socket
import sys


NOTICE_BYTES = 65536
CHILD_OPTION = "--private-pilot-fd"


class DeliveryRefusal(ValueError):
    """Categorical delivery failure; never include private source text."""


def password_from_notice(value: bytes) -> bytes:
    """Accept one explicit field (including a wrapped archive list) or bare token.

    The bounded provider-notice format may have descriptive words between
    'password' and its colon. Ambiguous fields, prose without a field, blank
    values, quoted values and whitespace-bearing tokens refuse without guessing.
    A password label listing ZIP filenames may wrap once before its final
    filename and colon; that filename remains part of the label, not the secret.
    Python objects are not claimed to provide secure erasure.
    """
    if not isinstance(value, bytes) or not 0 < len(value) <= NOTICE_BYTES:
        raise DeliveryRefusal()
    try:
        text = value.decode("utf-8-sig")
    except UnicodeError:
        raise DeliveryRefusal() from None
    fields = re.findall(r"(?im)\bpassword[^\r\n:]{0,120}:[ \t]*([^\r\n]*)", text)
    fields += re.findall(
        r"(?im)^[ \t]*(?:password|zip password|archive password)[ \t]*=[ \t]*([^\r\n]*)",
        text,
    )
    wrapped = re.findall(
        r"(?im)^([ \t]*password\b[^\r\n:]{0,240}\.zip[^\r\n:]{0,80})\r?\n"
        r"[ \t]*[A-Za-z0-9_.-]+\.zip[ \t]*:[ \t]*([^\r\n]*)", text)
    fields += [value for _, value in wrapped]
    wrapped_headings = {label.strip() for label, _ in wrapped}
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    heading = re.compile(r"(?i)^[#*_ ]*password(?:[ \t]+[A-Za-z0-9_()./, -]+)?[*_ ]*$")
    headings = [index for index, line in enumerate(lines)
                if heading.fullmatch(line) and not line.rstrip("#*_ ").endswith(".")
                and line not in wrapped_headings]
    if fields:
        if len(fields) != 1 or headings:
            raise DeliveryRefusal()
        candidate = fields[0].strip()
    else:
        candidate = text.strip()
    if (re.fullmatch(r"[!-~]{1,256}", candidate) is None
            or candidate.startswith(("'", '"'))
            or candidate.endswith(("'", '"'))):
        raise DeliveryRefusal()
    return candidate.encode("ascii")


def _require_compute():
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise DeliveryRefusal()


def _deliver_and_exec(attempt="pilot001"):
    import private_stimulus_feature_pilot as pilot

    # Reuse the fixed metadata selector before reading the private notice.
    # No neural values, pixels or model parameters enter this preparation.
    private_output, report_path = pilot.pilot_paths(attempt)
    pilot._selection()
    if (private_output.exists() or private_output.is_symlink()
            or private_output.parent.exists() or report_path.parent.exists()):
        raise DeliveryRefusal()
    notice = pilot.archive_metadata.ACQUISITION / "terms/password_images.txt"
    with notice.open("rb") as source:
        password = password_from_notice(source.read(NOTICE_BYTES + 1))
    read_fd, write_fd = os.pipe()
    try:
        try:
            # At most 256 bytes fit in an empty pipe without a reader process.
            if os.write(write_fd, password) != len(password):
                raise DeliveryRefusal()
        finally:
            os.close(write_fd)
        del password
        os.set_inheritable(read_fd, True)
        # exec retains this PID, so the selected external timeout bounds both
        # notice delivery and pilot execution without an orphan child process.
        os.execv(sys.executable, [sys.executable, "-B", str(Path(__file__).resolve()),
                                 "--attempt", attempt, CHILD_OPTION, str(read_fd)])
    finally:
        os.close(read_fd)


def _run_pilot(read_fd: int, attempt="pilot001") -> int:
    import private_stimulus_feature_pilot as pilot

    try:
        private_output, report_path = pilot.pilot_paths(attempt)
        report = pilot.run_private_development_feature_pilot(
            password_fd=read_fd,
            private_output_dir=private_output,
            operational_report_path=report_path,
            random_checkpoint_path=pilot.RANDOM_CHECKPOINT,
            attempt=attempt,
        )
    finally:
        os.close(read_fd)
    return 0 if (report.get("status") == "completed_private_staging"
                 and report.get("pilot_feature_extraction_complete") is True) else 1


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else list(argv)
    phase = "private_delivery"
    try:
        _require_compute()
        attempt = "pilot001"
        if len(args) >= 2 and args[0] == "--attempt":
            attempt, args = args[1], args[2:]
        if re.fullmatch(r"pilot[0-9]{3,6}", attempt) is None:
            raise DeliveryRefusal()
        if not args:
            _deliver_and_exec(attempt)
            raise DeliveryRefusal()  # exec must replace this process.
        if (len(args) != 2 or args[0] != CHILD_OPTION
                or not args[1].isascii() or not args[1].isdecimal()
                or int(args[1]) < 3):
            raise DeliveryRefusal()
        phase = "fixed_pilot"
        code = _run_pilot(int(args[1]), attempt)
        print(json.dumps({"status": "completed" if code == 0 else "stopped",
                          "phase": phase}), flush=True)
        return code
    except BaseException:
        # Never emit exception text, traceback, private notice or password.
        print(json.dumps({"status": "stopped", "phase": phase}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
