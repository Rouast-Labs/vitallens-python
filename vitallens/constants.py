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

# API settings
API_MIN_FRAMES = 16
API_MAX_FRAMES = 900
API_OVERLAP = 30
API_URL = "https://api.rouast.com/vitallens-v3/file"
if 'API_URL' in os.environ:
  API_URL = os.getenv('API_URL')
API_RESOLVE_URL = "https://api.rouast.com/vitallens-v3/resolve-model"
if 'API_RESOLVE_URL' in os.environ:
  API_RESOLVE_URL = os.getenv('API_RESOLVE_URL')

# For local development against dev endpoints, create a `.env` file in the root
# of the project and set the variables, for example:
# API_URL="https://api-dev.rouast.com/vitallens-dev/file"
# API_RESOLVE_URL="https://api-dev.rouast.com/vitallens-dev/resolve-model"

VITAL_CODES_TO_NAMES = {
  'ppg': 'ppg_waveform',
  'resp': 'respiratory_waveform',
  'hr': 'heart_rate',
  'rr': 'respiratory_rate',
  'hrv_sdnn': 'hrv_sdnn',
  'hrv_rmssd': 'hrv_rmssd',
  'hrv_lfhf': 'hrv_lfhf'
}

# Video error message
VIDEO_PARSE_ERROR = "Unable to parse input video. There may be an issue with the video file."

# Disclaimer message
DISCLAIMER = "The provided values are estimates and should be interpreted according to the provided confidence levels ranging from 0 to 1. The VitalLens API is not a medical device and its estimates are not intended for any medical purposes."
