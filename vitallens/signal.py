# Copyright (c) 2024 Rouast Labs
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from prpy.numpy.signal import moving_average_size_for_response

from vitallens.constants import SECONDS_PER_MINUTE, CALC_HR_MAX, CALC_RR_MAX

def moving_average_size_for_hr_response(sampling_freq):
  return moving_average_size_for_response(sampling_freq, CALC_HR_MAX / SECONDS_PER_MINUTE)

def moving_average_size_for_rr_response(sampling_freq):
  return moving_average_size_for_response(sampling_freq, CALC_RR_MAX / SECONDS_PER_MINUTE)

def detrend_lambda_for_hr_response(sampling_freq):
  return int(0.1614*np.power(sampling_freq, 1.9804))

def detrend_lambda_for_rr_response(sampling_freq):
  return int(4.4248*np.power(sampling_freq, 2.1253))
