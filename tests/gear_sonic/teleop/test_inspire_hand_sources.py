import tempfile
import unittest
from pathlib import Path

from gear_sonic.utils.teleop.solver.hand.inspire_hand_sources import (
    choose_inspire_config_path,
)


class InspireHandSourcesTest(unittest.TestCase):
    def test_prefers_upstream_inspire_config_when_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            upstream = tmpdir_path / "xr_teleoperate" / "assets" / "inspire_hand" / "inspire_hand.yml"
            local = tmpdir_path / "repo" / "inspire_hand.yml"
            upstream.parent.mkdir(parents=True, exist_ok=True)
            local.parent.mkdir(parents=True, exist_ok=True)
            upstream.write_text("upstream: true\n", encoding="utf-8")
            local.write_text("local: true\n", encoding="utf-8")

            chosen = choose_inspire_config_path(preferred_path=upstream, fallback_path=local)

            self.assertEqual(chosen, upstream)

    def test_falls_back_to_local_inspire_config_when_upstream_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            upstream = tmpdir_path / "xr_teleoperate" / "assets" / "inspire_hand" / "inspire_hand.yml"
            local = tmpdir_path / "repo" / "inspire_hand.yml"
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_text("local: true\n", encoding="utf-8")

            chosen = choose_inspire_config_path(preferred_path=upstream, fallback_path=local)

            self.assertEqual(chosen, local)
if __name__ == "__main__":
    unittest.main()
