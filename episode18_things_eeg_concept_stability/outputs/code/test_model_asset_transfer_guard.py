"""Source-free regression for the corrected Slurm transfer route; run in Slurm."""
import unittest
from unittest import mock

import acquire_model_assets as transfer


class ModelAssetTransferGuardTests(unittest.TestCase):
    def environment(self):
        return {"SLURM_JOB_ID": "12345", "SLURM_CPUS_PER_TASK": "1",
                **{name: "1" for name in transfer.THREADS}}

    def test_exact_one_cpu_compute_context_is_allowed_without_credentials(self):
        with mock.patch.dict(transfer.os.environ, self.environment(), clear=True), \
                mock.patch.object(transfer.socket, "gethostname", return_value="sh04-01n01.int"):
            transfer.require_compute_transfer()

    def test_login_dtn_or_missing_allocation_refuses_before_transfer(self):
        for host, environment in (("sh04-ln04.stanford.edu", self.environment()),
                                  ("dtn.sherlock.stanford.edu", self.environment()),
                                  ("sh04-01n01.int", {})):
            with self.subTest(host=host, allocated=bool(environment)), \
                    mock.patch.dict(transfer.os.environ, environment, clear=True), \
                    mock.patch.object(transfer.socket, "gethostname", return_value=host), \
                    self.assertRaisesRegex(RuntimeError, "actual Slurm compute"):
                transfer.require_compute_transfer()

    def test_cpu_and_all_six_thread_limits_are_mandatory(self):
        for key in ("SLURM_CPUS_PER_TASK", *transfer.THREADS):
            with self.subTest(key=key), \
                    mock.patch.dict(transfer.os.environ, {**self.environment(), key: "2"}, clear=True), \
                    mock.patch.object(transfer.socket, "gethostname", return_value="sh04-01n01.int"), \
                    self.assertRaisesRegex(RuntimeError, "one allocated CPU"):
                transfer.require_compute_transfer()


if __name__ == "__main__":
    unittest.main()
