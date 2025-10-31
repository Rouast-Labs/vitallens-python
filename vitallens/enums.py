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

from enum import IntEnum

class Method(IntEnum):
  VITALLENS = 1           # Automatic model selection
  VITALLENS_1_0 = 2       # Force VitalLens 1.0
  VITALLENS_1_1 = 3       # Force VitalLens 1.1
  VITALLENS_2_0 = 4       # Force VitalLens 2.0
  G = 5
  CHROM = 6
  POS = 7

METHOD_TO_NAME = {
  Method.VITALLENS: 'vitallens',
  Method.VITALLENS_1_0: 'vitallens-1.0',
  Method.VITALLENS_1_1: 'vitallens-1.1',
  Method.VITALLENS_2_0: 'vitallens-2.0',
  Method.G: 'g',
  Method.CHROM: 'chrom',
  Method.POS: 'pos'
}

NAME_TO_METHOD = {
  'vitallens': Method.VITALLENS,
  'vitallens-1.0': Method.VITALLENS_1_0,
  'vitallens-1.1': Method.VITALLENS_1_1,
  'vitallens-2.0': Method.VITALLENS_2_0,
  'g': Method.G,
  'chrom': Method.CHROM,
  'pos': Method.POS
}

class Mode(IntEnum):
  BATCH = 1
  BURST = 2
