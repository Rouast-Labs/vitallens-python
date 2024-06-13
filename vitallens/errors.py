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

class VitalLensAPIKeyError(Exception):
  """Exception raised for errors related to the API key."""
  def __init__(self, message="A valid API key is required to use Method.VITALLENS. Get one for free at https://www.rouast.com/api."):
    self.message = message
    super().__init__(self.message)

class VitalLensAPIQuotaExceededError(Exception):
  """Exception raised if quota exceeded."""
  def __init__(self, message="The quota or rate limit associated with your API Key may have been exceeded. Check your account at https://www.rouast.com/api and consider changing to a different plan."):
    self.message = message
    super().__init__(self.message)

class VitalLensAPIError(Exception):
  """Exception raised for internal API errors."""
  def __init__(self, message="Bad request or an error occured in the API."):
    self.message = message
    super().__init__(self.message)
