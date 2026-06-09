'''Place to keep pytest fixtures, etc.'''

import pandas as pd
import numpy as np
import pyleoclim as pyleo
import pytest 

@pytest.fixture
def gen_ts():
    """ Generate realistic-ish Series for testing """
    t,v = pyleo.utils.gen_ts(model='colored_noise',nt=50)
    ts = pyleo.Series(t,v, verbose=False, auto_time_params=True)
    return ts