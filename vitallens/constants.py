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

from dotenv import load_dotenv
import os
load_dotenv()

# Minima and maxima of derived vitals
CALC_HR_MIN = 40
CALC_HR_MAX = 240
CALC_HR_WINDOW_SIZE = 10
CALC_RR_MIN = 4
CALC_RR_MAX = 60
CALC_RR_WINDOW_SIZE = 30

# API settings
API_MIN_FRAMES = 16
API_MAX_FRAMES = 900
API_OVERLAP = 30
API_URL = "https://api.rouast.com/vitallens-v2"
if 'API_URL' in os.environ:
  API_URL = os.getenv('API_URL')

# Video error message
VIDEO_PARSE_ERROR = "Unable to parse input video. There may be an issue with the video file."

# Disclaimer message
DISCLAIMER = "The provided values are estimates and should be interpreted according to the provided confidence levels ranging from 0 to 1. The VitalLens API is not a medical device and its estimates are not intended for any medical purposes."
