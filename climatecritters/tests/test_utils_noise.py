''' Tests for climatecritters.utils.noise

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

class TestUtilsNoiseFromSeries:
    @pytest.mark.parametrize('method', ['uar1','ar1sim','phaseran'])
    @pytest.mark.parametrize('number', [1,10])
    @pytest.mark.parametrize('seed', [None,42])
    def test_fromseries_t0(self,gen_ts,method,number,seed):
        '''Test from_series method'''
        series = gen_ts
        cc.utils.noise.from_series(target_series=series,method=method,number=number,seed=seed)

class TestUtilsNoiseFromParams:
    @pytest.mark.parametrize('method, noise_param', [('uar1',[1,5]),
                                                     ('CN',[1]),
                                                     ('ar1sim',[1,5]),])
    @pytest.mark.parametrize('number', [1,10])
    @pytest.mark.parametrize('seed', [None,42])
    @pytest.mark.parametrize('time_pattern,settings',[('even',None),
                                                       ('random',None),
                                                       ('specified',{'time':np.arange(50)})])
    def test_fromparams_t0(self,method,noise_param,number,seed,time_pattern,settings):
        '''Test from_series method'''
        cc.utils.noise.from_param(method=method,noise_param=noise_param,number=number,seed=seed,time_pattern=time_pattern,settings=settings)