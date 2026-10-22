"""Third manually selected retry: preserve both earlier retries' bindings."""
from pathlib import Path
import importlib.util
import sys

_spec = importlib.util.spec_from_file_location(
    "_ep21_draw_retry003_adapter", Path(__file__).with_name("run_amended_draw_retry.py"))
adapter = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = adapter
_spec.loader.exec_module(adapter)

# This private instance reuses admission, composition, clock and publication.
# Original batch 27 completed and is intentionally absent from this selection.
adapter.ATTEMPT = "attempt003"
adapter.RETRY_ROOT = adapter.ROOT / "draw_retries" / adapter.ATTEMPT
adapter.SELECTION = adapter.RETRY_ROOT / "selection.json"
adapter.INDICES = (*range(16, 27), 28)

if __name__ == "__main__":
    raise SystemExit(adapter.caller.main())
