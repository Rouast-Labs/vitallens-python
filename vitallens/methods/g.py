# Copyright (c) 2026 Rouast Labs
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
import vitallens_core as vc

from vitallens.enums import Method
from vitallens.methods.simple_rppg_method import SimpleRPPGMethod

class GRPPGMethod(SimpleRPPGMethod):
  """The G algorithm by Verkruysse (2008)"""
  def __init__(self):
    """Initialize the `GRPPGMethod`"""
    super(GRPPGMethod, self).__init__()
    self.method = Method.G
    config = vc.SessionConfig(
      model_name="g",
      supported_vitals=["heart_rate"],
      return_waveforms=["ppg_waveform"],
      fps_target=30.0,
      input_size=100,
      n_inputs=64,
      roi_method="face"
    )
    self.parse_config(config, est_window_length=64, est_window_overlap=0)

  def algorithm(
      self,
      rgb: np.ndarray,
      fps: float
    ) -> np.ndarray:
    """Use G algorithm to estimate pulse from rgb signal.

    Args:
      rgb: The rgb signal. Shape (n_frames, 3)
      fps: The rate at which signal was sampled.
    Returns:
      sig: The estimated pulse signal. Shape (n_frames,)
    """
    # Select green channel
    g = rgb[:,1]
    # Invert
    g = -1 * g
    # Return
    return g
