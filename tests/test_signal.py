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
import pytest

import sys
sys.path.append('../vitallens-python')

from vitallens.signal import reassemble_from_windows

@pytest.mark.parametrize(
  "name, x_batches, idxs_batches, expected_x, expected_idxs",
  [
    # Test case 1: Standard overlap
    (
      "standard_overlap",
      [
        np.array([[2.0, 4.0, 6.0, 8.0, 10.0], [2.0, 3.0, 4.0, 5.0, 6.0]]),
        np.array([[7.0, 1.0, 10.0, 12.0, 18.0], [7.0, 8.0, 9.0, 10.0, 11.0]])
      ],
      [
        np.array([1, 3, 5, 7, 9]),
        np.array([5, 6, 9, 11, 13])
      ],
      np.array([[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 18.0],
                [2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 11.0]]),
      np.array([1, 3, 5, 7, 9, 11, 13])
    ),
    # Test case 2: No overlap between windows.
    (
      "no_overlap",
      [
        np.array([[1, 2], [10, 20]]),
        np.array([[3, 4], [30, 40]])
      ],
      [
        np.array([0, 1]),
        np.array([3, 4])
      ],
      np.array([[1, 2, 3, 4], [10, 20, 30, 40]]),
      np.array([0, 1, 3, 4])
    ),
    # Test case 3: A single window.
    (
      "single_window",
      [
        np.array([[1, 2, 3], [10, 20, 30]])
      ],
      [
        np.array([0, 1, 2])
      ],
      np.array([[1, 2, 3], [10, 20, 30]]),
      np.array([0, 1, 2])
    )
  ]
)
def test_reassemble_from_windows_edge_cases(name, x_batches, idxs_batches, expected_x, expected_idxs):
  out_x, out_idxs = reassemble_from_windows(x=x_batches, idxs=idxs_batches)
  np.testing.assert_allclose(out_x, expected_x, err_msg=f"Failed on case: {name} (data)")
  np.testing.assert_equal(out_idxs, expected_idxs, err_msg=f"Failed on case: {name} (indices)")
