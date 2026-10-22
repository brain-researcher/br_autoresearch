"""Fresh batch-37 retry after its diagnosed metadata JSON read failure."""
from pathlib import Path
import importlib.util
import json
import sys

_spec = importlib.util.spec_from_file_location(
    "_ep21_draw_retry004_adapter", Path(__file__).with_name("run_amended_draw_retry.py"))
adapter = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = adapter
_spec.loader.exec_module(adapter)

adapter.ATTEMPT = "attempt004"
adapter.RETRY_ROOT = adapter.ROOT / "draw_retries" / adapter.ATTEMPT
adapter.SELECTION = adapter.RETRY_ROOT / "selection.json"
adapter.INDICES = (37,)


def metadata_failure_exit_code(index):
    adapter.caller._require(index == 37, "batch37_metadata_retry_only")
    chosen = json.loads(adapter.caller._text(adapter.SELECTION))
    adapter.caller._metadata_read_failure(adapter.ROOT, adapter.ATTEMPT, index,
        chosen.get("failed_original_children", {}).get("37", {}))
    return "1"


# Only this private instance accepts exit 1 and requires its preserved diagnosis.
adapter.failure_exit_code = metadata_failure_exit_code

if __name__ == "__main__":
    raise SystemExit(adapter.caller.main())
