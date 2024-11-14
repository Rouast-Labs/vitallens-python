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

from vitallens.buffer import SignalBuffer, MultiSignalBuffer

@pytest.mark.parametrize("pad_val", [0, -1])
def test_signal_buffer(pad_val):
  # 1 dim
  buffer = SignalBuffer(size=8, ndim=1, pad_val=pad_val)
  with pytest.raises(Exception):
    buffer.get()
  np.testing.assert_allclose(
    buffer.update(signal=[.2, .4,], dt=2),
    np.asarray([pad_val, pad_val, pad_val, pad_val, pad_val, pad_val, .2, .4]))
  np.testing.assert_allclose(
    buffer.update(signal=[.1, .3, .5, .6], dt=2),
    np.asarray([pad_val, pad_val, pad_val, pad_val, .15, .35, .5, .6]))
  np.testing.assert_allclose(
    buffer.update(signal=[.6, .7], dt=1),
    np.asarray([pad_val, pad_val, pad_val, .15, .35, .5, .6, .7]))
  np.testing.assert_allclose(
    buffer.update(signal=[.8, .9, .8, .7, .6], dt=4),
    np.asarray([.35, .5, .6, .75, .9, .8, .7, .6]))
  # 2 dim
  buffer = SignalBuffer(size=4, ndim=2, pad_val=pad_val)
  with pytest.raises(Exception):
    buffer.get()
  np.testing.assert_allclose(
    buffer.update(signal=[[.1, .2,]], dt=1),
    np.asarray([[pad_val, pad_val], [pad_val, pad_val], [pad_val, pad_val], [.1, .2]]))
  np.testing.assert_allclose(
    buffer.update(signal=[[.1, .3], [.5, .6]], dt=3),
    np.asarray([[.1, .2], [pad_val, pad_val], [.1, .3], [.5, .6]]))
  np.testing.assert_allclose(
    buffer.update(signal=[[.3, .2], [.5, .6], [.5, .6]], dt=2),
    np.asarray([[.1, .3], [.4, .4], [.5, .6], [.5, .6]]))

@pytest.mark.parametrize("pad_val", [0, -1])
def test_multi_signal_buffer(pad_val):
  # 1 dim, 2 signals
  buffer = MultiSignalBuffer(size=8, ndim=1, ignore_k=[], pad_val=pad_val)
  with pytest.raises(Exception):
    buffer.get()
  with pytest.raises(Exception):
    buffer.update(signals=[0., 1.])
  out = buffer.update(
    signals={"a": [.2, .4,], "b": [.1, .2]}, dt=1)
  np.testing.assert_allclose(
    out["a"],
    np.asarray([pad_val, pad_val, pad_val, pad_val, pad_val, pad_val, .2, .4]))
  np.testing.assert_allclose(
    out["b"],
    np.asarray([pad_val, pad_val, pad_val, pad_val, pad_val, pad_val, .1, .2]))
  out = buffer.update(
    signals={"a": [.1, .3, .5, .6], "b": [.1, .4, .5]}, dt=2)
  np.testing.assert_allclose(
    out["a"],
    np.asarray([pad_val, pad_val, pad_val, pad_val, .15, .35, .5, .6]))
  np.testing.assert_allclose(
    out["b"],
    np.asarray([pad_val, pad_val, pad_val, pad_val, .1, .15, .4, .5]))
  out = buffer.update(
    signals={"a": [.6, .7], "b": [.3], "c": [.4, .8]}, dt=1)
  np.testing.assert_allclose(
    out["a"],
    np.asarray([pad_val, pad_val, pad_val, .15, .35, .5, .6, .7]))
  np.testing.assert_allclose(
    out["b"],
    np.asarray([pad_val, pad_val, pad_val, .1, .15, .4, .5, .3]))
  np.testing.assert_allclose(
    out["c"],
    np.asarray([pad_val, pad_val, pad_val, pad_val, pad_val, pad_val, .4, .8]))
  # 2 dim, 2 signals
  buffer = MultiSignalBuffer(size=4, ndim=2, pad_val=pad_val, ignore_k=['c'])
  out = buffer.update(
    signals={"a": [[.2, .4,], [.1, .4]], "b": [[.1, .2], [.6, .6]]}, dt=1)
  np.testing.assert_allclose(
    out["a"],
    np.asarray([[pad_val, pad_val], [pad_val, pad_val], [.2, .4], [.1, .4]]))
  np.testing.assert_allclose(
    out["b"],
    np.asarray([[pad_val, pad_val], [pad_val, pad_val], [.1, .2], [.6, .6]]))
  out = buffer.update(
    signals={"a": [[.1, .1,], [.1, .1,], [.1, .1]], "c": [[.1], [.1]]}, dt=2)
  assert len(out) == 1
  np.testing.assert_allclose(
    out["a"],
    np.asarray([[.2, .4], [.1, .25], [.1, .1], [.1, .1]]))
