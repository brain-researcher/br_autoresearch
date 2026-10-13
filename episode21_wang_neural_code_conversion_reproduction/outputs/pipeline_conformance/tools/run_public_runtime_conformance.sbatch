#!/usr/bin/env bash
# Retired standalone submission template. The check is now a bounded stage.
# Preserve this path for old references, but never submit it as a short job.
set -euo pipefail
printf '%s\n' 'Standalone public-runtime submission is retired. Run run_public_runtime_conformance.py --allocation-authorization <approved EP21 job metadata> inside the substantive allocation.' >&2
exit 2
