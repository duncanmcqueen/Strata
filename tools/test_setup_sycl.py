"""Tests for setup.py's Intel Arc (SYCL) pieces on mocked tool output: the cards `sycl-ls --verbose` lists, numbered as
ONEAPI_DEVICE_SELECTOR=level_zero:N numbers them, their ocloc AOT names and VRAM by name; the VRAM the compiled engine's
strata-device reports.  No GPU, no oneAPI, no downloads.

    python -m unittest tools.test_setup_sycl
"""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

# the shape sycl-ls --verbose prints (oneAPI 2026.1, two Arc cards: one platform per card, in Level Zero order)
SYCL_LS = """Platforms: 2
Platform [#1]:
    Version  : 1.6
    Name     : Intel(R) oneAPI Unified Runtime over Level-Zero V2
    Vendor   : Intel(R) Corporation
    Devices  : 1
        Device [#0]:
        Type              : gpu
        Version           : 20.1.0
        Name              : Intel(R) Arc(TM) B580 Graphics
        Vendor            : Intel(R) Corporation
        Driver            : 1.6.33578+15
        DeviceID          : 57867
        Num SubDevices    : 0
        Architecture: intel_gpu_bmg_g21
Platform [#2]:
    Version  : 1.6
    Name     : Intel(R) oneAPI Unified Runtime over Level-Zero V2
    Vendor   : Intel(R) Corporation
    Devices  : 1
        Device [#0]:
        Type              : gpu
        Version           : 12.55.8
        Name              : Intel(R) Arc(TM) A770 Graphics
        Vendor            : Intel(R) Corporation
        Driver            : 1.6.33578+15
        DeviceID          : 22176
        Num SubDevices    : 0
        Architecture: intel_gpu_acm_g10
"""


def completed(stdout: str):
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


class IntelDetection(unittest.TestCase):
    def setUp(self):
        self.win = setup.WIN
        setup.WIN = False

    def tearDown(self):
        setup.WIN = self.win

    def cards(self, text):
        with mock.patch.object(setup, "oneapi_root", return_value=None), \
             mock.patch.object(setup.shutil, "which", return_value="/usr/bin/sycl-ls"), \
             mock.patch.object(setup.Path, "exists", return_value=True), \
             mock.patch.object(setup.subprocess, "run", return_value=completed(text)):
            return setup.intel_gpus()

    def test_two_cards_in_level_zero_order(self):
        g = self.cards(SYCL_LS)
        self.assertEqual([x["index"] for x in g], [0, 1])
        self.assertEqual([x["name"] for x in g], ["Intel(R) Arc(TM) B580 Graphics", "Intel(R) Arc(TM) A770 Graphics"])
        self.assertEqual([x["arch"] for x in g], ["bmg-g21", "dg2-g10"])
        self.assertEqual([x["vram_gb"] for x in g], [12.0, 16.0])
        self.assertTrue(all(x["count"] == 2 for x in g))
        self.assertTrue(all(setup.sycl_problem(x) is None for x in g))

    def test_unknown_architecture_gets_the_jit_image(self):
        g = self.cards(SYCL_LS.replace("intel_gpu_acm_g10", "intel_gpu_future_g99"))
        self.assertEqual(g[1]["arch"], "")
        self.assertEqual(g[1]["arch_raw"], "intel_gpu_future_g99")

    def test_no_tool_no_cards(self):
        with mock.patch.object(setup, "oneapi_root", return_value=None), \
             mock.patch.object(setup.shutil, "which", return_value=None):
            self.assertEqual(setup.intel_gpus(), [])

    def test_windows_has_no_intel_backend(self):
        setup.WIN = True
        self.assertEqual(setup.intel_gpus(), [])

    def test_vram_from_the_engine(self):
        listing = "device 0: Intel(R) Arc(TM) A770 Graphics\n  Level Zero, 512 compute units, 15.9 GiB, sub-group 32\n"
        gpu = {"index": 1, "name": "Intel(R) Arc(TM) A770 Graphics", "vram_gb": 16.0}
        with mock.patch.object(setup.subprocess, "run", return_value=completed(listing)) as run:
            g = setup.sycl_card_vram(Path("/x/engine"), gpu)
        self.assertAlmostEqual(g["vram_gb"], 15.9 * 1.073741824, places=3)
        env = run.call_args.kwargs["env"]
        self.assertEqual(env["ONEAPI_DEVICE_SELECTOR"], "level_zero:1")
        for k, v in setup.SYCL_ENV.items():
            self.assertEqual(env[k], v)

    def test_vram_unchanged_when_the_engine_says_nothing(self):
        gpu = {"index": 0, "name": "x", "vram_gb": 8.0}
        with mock.patch.object(setup.subprocess, "run", return_value=completed("(no GPU device)\n")):
            self.assertEqual(setup.sycl_card_vram(Path("/x/engine"), gpu)["vram_gb"], 8.0)


if __name__ == "__main__":
    unittest.main()
