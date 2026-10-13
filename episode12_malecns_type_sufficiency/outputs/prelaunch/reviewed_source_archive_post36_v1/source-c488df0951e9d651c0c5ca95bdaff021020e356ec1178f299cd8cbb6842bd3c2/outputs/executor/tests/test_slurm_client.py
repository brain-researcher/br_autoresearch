from __future__ import annotations

import unittest

from ep12_executor.slurm_client import sanitized_slurm_client_environment


class SlurmClientEnvironmentTests(unittest.TestCase):
    def test_hostile_path_and_scheduler_overrides_are_removed(self) -> None:
        environment = sanitized_slurm_client_environment(
            {
                "PATH": "/tmp/hostile:/home/example/bin",
                "SBATCH_PARTITION": "poison",
                "SLURM_JOB_ID": "999",
                "EP12_UNSAFE": "poison",
                "PYTHONPATH": "/tmp/poison",
                "LANG": "C",
            }
        )
        self.assertEqual(environment, {"PATH": "/usr/bin:/bin", "LANG": "C"})

    def test_only_nul_free_locale_values_survive(self) -> None:
        environment = sanitized_slurm_client_environment(
            {
                "PATH": "/does/not/matter",
                "LC_ALL": "C.UTF-8",
                "LC_CTYPE": "bad\x00value",
                "LANGUAGE": "en_US:en",
            }
        )
        self.assertEqual(
            environment,
            {
                "PATH": "/usr/bin:/bin",
                "LANGUAGE": "en_US:en",
                "LC_ALL": "C.UTF-8",
            },
        )


if __name__ == "__main__":
    unittest.main()
