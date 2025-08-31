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

import abc

from vitallens.enums import Mode

class RPPGMethod(metaclass=abc.ABCMeta):
  """Abstract superclass for rPPG methods"""
  def __init__(
      self,
      mode: Mode
    ):
    """Initialize the `RPPGMethod`
    
    Args:
      method: The selected method
      mode: The operation mode
    """
    self.op_mode = mode
  def parse_config(
      self,
      config: dict
    ):
    """Set properties based on the config.
    
    Args:
      config: The method's config dict
    """
    self.roi_method = config['roi_method']
    self.fps_target = config['fps_target']
    self.est_window_length = config['est_window_length']
    self.est_window_overlap = config['est_window_overlap']
    self.est_window_flexible = self.est_window_length == 0
  @abc.abstractmethod
  def __call__(self, frames, faces, fps, override_fps_target, override_global_parse):
    """Run inference. Abstract method to be implemented in subclasses."""
    pass
  @abc.abstractmethod
  def reset(self):
    """Reset. Abstract method to be implemented in subclasses."""
    pass
  