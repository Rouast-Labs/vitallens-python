# Copyright (c) 2024 Philipp Rouast
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
import pytest

import sys
sys.path.append('../vitallens-python')

from vitallens.signal import windowed_mean, windowed_freq, reassemble_from_windows

def test_windowed_mean():
  x = np.asarray([0., 1., 2., 3., 4., 5., 6.])
  y = np.asarray([1., 1., 2., 3., 4., 5., 5.])
  out_y = windowed_mean(x=x, window_size=3, overlap=1)
  np.testing.assert_equal(
    out_y,
    y)

@pytest.mark.parametrize("num", [100, 1000])
@pytest.mark.parametrize("freq", [2.35, 4.89, 13.55])
@pytest.mark.parametrize("window_size", [10, 20])
def test_estimate_freq_periodogram(num, freq, window_size):
  # Test data
  x = np.linspace(0, freq * 2 * np.pi, num=num)
  np.random.seed(0)
  y = 100 * np.sin(x) + np.random.normal(scale=8, size=num)
  # Check a default use case with axis=-1
  np.testing.assert_allclose(
    windowed_freq(x=y, window_size=window_size, overlap=window_size//2, f_s=len(x), f_range=(max(freq-2,1),freq+2), f_res=0.05),
    np.full((num,), fill_value=freq),
    rtol=1)
  
def test_reassemble_from_windows():
  x = np.array([[[2.0, 4.0, 6.0, 8.0, 10.0], [7.0, 1.0, 10.0, 12.0, 18.0]],
                [[2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0]]], dtype=np.float32).transpose(1, 0, 2)
  idxs = np.array([[1, 3, 5, 7, 9], [5, 6, 9, 11, 13]], dtype=np.int64)
  out_x, out_idxs = reassemble_from_windows(x=x, idxs=idxs)
  np.testing.assert_equal(
    out_x,
    np.asarray([[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 18.0],
                [2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 11.0]]))
  np.testing.assert_equal(
    out_idxs,
    np.asarray([1, 3, 5, 7, 9, 11, 13]))
  