"""Unit tests for powermetrics log parsing."""

from __future__ import annotations

import plistlib
import unittest
from pathlib import Path

from powermetrics_log import parse_powermetrics_file, sample_energy_joules

FIXTURE = Path(__file__).parent / "fixtures" / "sample.powermetrics.plist.xml"


class PowerMetricsLogTests(unittest.TestCase):
    def test_parse_fixture(self) -> None:
        raw = FIXTURE.read_bytes()
        path = self.enterContext(self._temp_file(raw))
        samples = parse_powermetrics_file(path)
        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample.thermal_pressure, "Nominal")
        self.assertAlmostEqual(sample.package_mw, 5500.5)
        self.assertAlmostEqual(sample.temp_c, 52.5)
        self.assertAlmostEqual(sample.gpu_active_ratio, 0.25)
        self.assertAlmostEqual(sample_energy_joules(sample), 0.175)

    def _temp_file(self, raw: bytes):
        from tempfile import NamedTemporaryFile

        handle = NamedTemporaryFile(delete=False)
        handle.write(raw)
        handle.flush()
        handle.close()
        path = Path(handle.name)

        class _Cleanup:
            def __enter__(self_inner):
                return path

            def __exit__(self_inner, exc_type, exc, tb):
                path.unlink(missing_ok=True)

        return _Cleanup()

    def test_parse_nul_separated_blobs(self) -> None:
        blob = plistlib.dumps(
            {
                "elapsed_ns": 1_000_000_000,
                "thermal_pressure": "Light",
                "processor": {
                    "cpu_energy": 10,
                    "gpu_energy": 5,
                    "ane_energy": 0,
                    "combined_power": 1000.0,
                    "clusters": [],
                },
                "gpu": {"idle_ratio": 0.9},
            },
            fmt=plistlib.FMT_XML,
        )
        path = self.enterContext(self._temp_file(blob + b"\x00" + blob))
        samples = parse_powermetrics_file(path)
        self.assertEqual(len(samples), 2)


if __name__ == "__main__":
    unittest.main()