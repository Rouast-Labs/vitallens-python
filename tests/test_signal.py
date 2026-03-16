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
