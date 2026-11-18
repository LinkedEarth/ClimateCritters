''' Tests for the bundled datasets of climatecritters.core.forcing.Forcing.from_csv '''

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestCoreForcingFromCsvDatasets:
    @pytest.mark.parametrize('dataset', ['insolation', 'vieira_tsi'])
    def test_from_csv_dataset_fresh_interpreter_t0(self, dataset):
        '''Bundled datasets load in a fresh interpreter, where importlib.resources
        has not already been imported by another package.'''
        code = (
            "import climatecritters as cc; "
            f"f = cc.Forcing.from_csv(dataset='{dataset}'); "
            "assert len(f.data) > 0"
        )
        result = subprocess.run([sys.executable, '-c', code], cwd=REPO_ROOT,
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
