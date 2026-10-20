"""Second manually selected retry: preserve the first retry's live bindings."""
from pathlib import Path
import importlib.util
import sys

_spec = importlib.util.spec_from_file_location(
    "_ep21_draw_retry002_adapter", Path(__file__).with_name("run_amended_draw_retry.py"))
adapter = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = adapter
_spec.loader.exec_module(adapter)

# These globals belong to this private adapter instance, never attempt001.
# Its unchanged admission, composition, clock and publication code are reused.
adapter.ATTEMPT = "attempt002"
adapter.RETRY_ROOT = adapter.ROOT / "draw_retries" / adapter.ATTEMPT
adapter.SELECTION = adapter.RETRY_ROOT / "selection.json"
adapter.INDICES = tuple(range(8, 16))

if __name__ == "__main__":
    raise SystemExit(adapter.caller.main())
