''' Tests for climatecritters.utils.resample

Naming rules:
1. class: Test{filename}{Class}{method} with appropriate camel case
2. function: test_{method}_t{test_id}

Notes on how to test:
0. Make sure [pytest](https://docs.pytest.org) has been installed: `pip install pytest`
1. execute `pytest {directory_path}` in terminal to perform all tests in all testing files inside the specified directory
2. execute `pytest {file_path}` in terminal to perform all tests in the specified file
3. execute `pytest {file_path}::{TestClass}::{test_method}` in terminal to perform a specific test class/method inside the specified file
4. after `pip install pytest-xdist`, one may execute "pytest -n 4" to test in parallel with number of workers specified by `-n`
5. for more details, see https://docs.pytest.org/en/stable/usage.html
'''

import pytest
import numpy as np
import climatecritters as cc

class TestUtilsResampleDownsample:
    @pytest.mark.parametrize('method, param', [('exponential',[1]), 
                                               ('poisson',[1]),
                                                ('pareto', [1,1]),
                                                ('random_choice',[[1,2],[0.5,0.5]])])
    @pytest.mark.parametrize('return_index', [True,False])
    @pytest.mark.parametrize('seed', [None,42])
    def test_downsample_t0(self,method,param,return_index,seed,gen_ts):
        '''Test from_series method'''
        series = gen_ts
        cc.utils.resample.downsample(series,method=method,param=param,return_index=return_index,seed=seed)