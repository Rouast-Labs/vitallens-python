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

import base64
import json
import numpy as np
from prpy.numpy.image import parse_image_inputs
from prpy.numpy.physio import CALC_HR_MIN_T, CALC_RR_MIN_T
import pytest
import requests
from unittest.mock import Mock, patch

import sys
sys.path.append('../vitallens-python')

from vitallens.constants import API_MAX_FRAMES, API_MIN_FRAMES, API_URL
from vitallens.enums import Method, Mode
from vitallens.methods.vitallens import VitalLensRPPGMethod, _resolve_model_config
from vitallens.errors import VitalLensAPIKeyError, VitalLensAPIError

def create_mock_response(
    status_code: int,
    json_data: dict
  ) -> Mock:
  """Create a mock response
  
  Args:
    status_code: The desired status code
    json_data: The desired json data
  Returns:
    mock: The Mock response object
  """
  mock_resp = Mock()
  mock_resp.status_code = status_code
  mock_resp.text = json.dumps(json_data)
  mock_resp.json.return_value = json_data
  mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError if status_code >= 400 else None
  return mock_resp

@pytest.fixture
def mock_resolve_config():
  """Provides a default successful mock for the resolve_model_config call."""
  with patch('vitallens.methods.vitallens._resolve_model_config') as mock:
    mock.return_value = {
      'model': 'vitallens-2.0',
      'n_inputs': 5,
      'input_size': 40,
      'fps_target': 30,
      'roi_method': 'upper_body_cropped',
      'signals': {'heart_rate', 'respiratory_rate', 'hrv_sdnn', 'hrv_rmssd', 'hrv_lfhf', 'ppg_waveform', 'respiratory_waveform'}
    }
    yield mock

def create_mock_api_response(
    url: str,
    headers: dict,
    json: dict
  ) -> Mock:
  """Create a mock api response
  
  Args:
    url: The API URL
    headers: The request headers
    json: The request payload
  Returns:
    mock: The Mock response object
  """
  api_key = headers["x-api-key"]
  if api_key is None or not isinstance(api_key, str) or len(api_key) < 30:
    return create_mock_response(
      status_code=403, json_data={"vital_signs": None, "face": None, "state": None, "message": "Error"}) 
  model = json.get("model", "vitallens-1.0")
  video_base64 = json["video"]
  if video_base64 is None:
    return create_mock_response(
      status_code=400, json_data={"vital_signs": None, "face": None, "state": None, "message": "Error"})
  try:
    video = np.frombuffer(base64.b64decode(video_base64), dtype=np.uint8)
    video = video.reshape((-1, 40, 40, 3))
  except Exception as e:
    return create_mock_response(status_code=400, json_data={"vital_signs": None, "face": None, "state": None, "message": f"Error: {e}"})
  if video.shape[0] < API_MIN_FRAMES or video.shape[0] > API_MAX_FRAMES:
    return create_mock_response(status_code=400, json_data={"vital_signs": None, "face": None, "state": None, "message": "Error"})
  return create_mock_response(
    status_code=200,
    json_data={
      "vital_signs": {
        "heart_rate": {"value": 60.0, "unit": "bpm", "confidence": 0.99, "note": "Note"},
        "respiratory_rate": {"value": 15.0, "unit": "bpm", "confidence": 0.97, "note": "Note"},
        "ppg_waveform": {"data": np.random.rand(video.shape[0]).tolist(), "unit": "unitless", "confidence": np.ones(video.shape[0]).tolist(), "note": "Note"},
        "respiratory_waveform": {"data": np.random.rand(video.shape[0]).tolist(), "unit": "unitless", "confidence": np.ones(video.shape[0]).tolist(), "note": "Note"}},
      "face": {"confidence": np.random.rand(video.shape[0]).tolist(), "note": "Note"},
      "state": {"data": np.zeros((256,), dtype=np.float32).tolist(), "note": "Note"},
      "model_used": model,
      "message": "Message"})

@pytest.mark.parametrize("override_fps_target", [None, 15, 10])
@pytest.mark.parametrize("override_global_parse", [False, True])
@pytest.mark.parametrize("requested_model", [Method.VITALLENS, Method.VITALLENS_1_0, Method.VITALLENS_2_0])
@patch('requests.post', side_effect=create_mock_api_response)
def test_VitalLensRPPGMethod_file_mock(mock_post, mock_resolve_config, request, override_fps_target, override_global_parse, requested_model):
  api_key = request.getfixturevalue('test_dev_api_key')
  test_video_path = request.getfixturevalue('test_video_path')
  test_video_ndarray = request.getfixturevalue('test_video_ndarray')
  test_video_faces = request.getfixturevalue('test_video_faces')
  method = VitalLensRPPGMethod(mode=Mode.BATCH,
                               api_key=api_key,
                               requested_model=requested_model)
  data, unit, conf, note, live = method(
    inputs=test_video_path, faces=test_video_faces,
    override_fps_target=override_fps_target,
    override_global_parse=override_global_parse)
  assert all(key in data for key in method.signals)
  assert all(key in unit for key in method.signals)
  assert all(key in conf for key in method.signals)
  assert all(key in note for key in method.signals)
  assert data['ppg_waveform'].shape == (test_video_ndarray.shape[0],)
  assert conf['ppg_waveform'].shape == (test_video_ndarray.shape[0],)
  assert data['respiratory_waveform'].shape == (test_video_ndarray.shape[0],)
  assert conf['respiratory_waveform'].shape == (test_video_ndarray.shape[0],)
  assert live.shape == (test_video_ndarray.shape[0],)

@pytest.mark.parametrize("override_fps_target", [None, 15, 10])
@pytest.mark.parametrize("long", [False, True])
@pytest.mark.parametrize("override_global_parse", [False, True])
@pytest.mark.parametrize("requested_model", [Method.VITALLENS, Method.VITALLENS_1_0, Method.VITALLENS_2_0])
@patch('requests.post', side_effect=create_mock_api_response)
def test_VitalLensRPPGMethod_mock(mock_post, mock_resolve_config, request, long, override_fps_target, override_global_parse, requested_model):
  api_key = request.getfixturevalue('test_dev_api_key')
  test_video_ndarray = request.getfixturevalue('test_video_ndarray')
  test_video_fps = request.getfixturevalue('test_video_fps')
  test_video_faces = request.getfixturevalue('test_video_faces')
  method = VitalLensRPPGMethod(mode=Mode.BATCH,
                               api_key=api_key,
                               requested_model=requested_model)
  if long:
    n_repeats = (API_MAX_FRAMES * 3) // test_video_ndarray.shape[0] + 1
    test_video_ndarray = np.repeat(test_video_ndarray, repeats=n_repeats, axis=0)
    test_video_faces = np.repeat(test_video_faces, repeats=n_repeats, axis=0)
  data, unit, conf, note, live = method(
    inputs=test_video_ndarray, faces=test_video_faces,
    fps=test_video_fps, override_fps_target=override_fps_target,
    override_global_parse=override_global_parse)
  assert all(key in data for key in method.signals)
  assert all(key in unit for key in method.signals)
  assert all(key in conf for key in method.signals)
  assert all(key in note for key in method.signals)
  assert data['ppg_waveform'].shape == (test_video_ndarray.shape[0],)
  assert conf['ppg_waveform'].shape == (test_video_ndarray.shape[0],)
  assert data['respiratory_waveform'].shape == (test_video_ndarray.shape[0],)
  assert conf['respiratory_waveform'].shape == (test_video_ndarray.shape[0],)
  assert live.shape == (test_video_ndarray.shape[0],)

@pytest.mark.parametrize("process_signals", [True, False])
@pytest.mark.parametrize("n_frames", [16, 250])
def test_VitalLens_API_valid_response(request, process_signals, n_frames):
  api_key = request.getfixturevalue('test_dev_api_key')
  test_video_ndarray = request.getfixturevalue('test_video_ndarray')
  test_video_fps = request.getfixturevalue('test_video_fps')
  test_video_faces = request.getfixturevalue('test_video_faces')
  frames, *_ = parse_image_inputs(
    inputs=test_video_ndarray, fps=test_video_fps, target_size=40,
    roi=test_video_faces[0].tolist(), library='prpy', scale_algorithm='bilinear')
  headers = {"x-api-key": api_key}
  payload = {"video": base64.b64encode(frames[:n_frames].tobytes()).decode('utf-8'), "origin": "vitallens-python"}
  if process_signals: payload['fps'] = str(30)
  response = requests.post(API_URL, headers=headers, json=payload)
  response_body = json.loads(response.text)
  assert response.status_code == 200
  assert all(key in response_body for key in ["face", "vital_signs", "state", "message"])
  vital_signs = response_body["vital_signs"]
  assert all(key in vital_signs for key in ["ppg_waveform", "respiratory_waveform"])
  ppg_waveform_data = np.asarray(response_body["vital_signs"]["ppg_waveform"]["data"])
  ppg_waveform_conf = np.asarray(response_body["vital_signs"]["ppg_waveform"]["confidence"])
  resp_waveform_data = np.asarray(response_body["vital_signs"]["respiratory_waveform"]["data"])
  resp_waveform_conf = np.asarray(response_body["vital_signs"]["respiratory_waveform"]["confidence"])
  assert ppg_waveform_data.shape == (n_frames,)
  assert ppg_waveform_conf.shape == (n_frames,)
  assert resp_waveform_data.shape == (n_frames,)
  assert resp_waveform_conf.shape == (n_frames,)
  t = n_frames/test_video_fps 
  if process_signals and t >= CALC_HR_MIN_T: assert "heart_rate" in vital_signs
  else: assert "heart_rate" not in vital_signs
  if process_signals and t >= CALC_RR_MIN_T: assert "respiratory_rate" in vital_signs
  else: assert "respiratory_rate" not in vital_signs
  live = np.asarray(response_body["face"]["confidence"])
  assert live.shape == (n_frames,)
  state = np.asarray(response_body["state"]["data"])
  assert state.shape == (256,)

def test_VitalLens_API_wrong_api_key(request):
  test_video_ndarray = request.getfixturevalue('test_video_ndarray')
  test_video_fps = request.getfixturevalue('test_video_fps')
  test_video_faces = request.getfixturevalue('test_video_faces')
  frames, *_ = parse_image_inputs(
    inputs=test_video_ndarray, fps=test_video_fps, target_size=40,
    roi=test_video_faces[0].tolist(), library='prpy', scale_algorithm='bilinear')
  headers = {"x-api-key": "WRONG_API_KEY"}
  payload = {"video": base64.b64encode(frames[:16].tobytes()).decode('utf-8'), "origin": "vitallens-python"}
  response = requests.post(API_URL, headers=headers, json=payload)
  assert response.status_code == 403

def test_VitalLens_API_no_video(request):
  api_key = request.getfixturevalue('test_dev_api_key')
  headers = {"x-api-key": api_key}
  payload = {"some_key": "irrelevant", "origin": "vitallens-python"}
  response = requests.post(API_URL, headers=headers, json=payload)
  assert response.status_code == 400

def test_VitalLens_API_no_parseable_video(request):
  api_key = request.getfixturevalue('test_dev_api_key')
  headers = {"x-api-key": api_key}
  payload = {"video": "not_parseable", "origin": "vitallens-python"}
  response = requests.post(API_URL, headers=headers, json=payload)
  assert response.status_code == 422

# TODO Fails
# def test_resolve_model_config_errors():
#   """Tests failure modes of the config resolution."""
#   with patch('requests.get') as mock_get:
#     # Test 403 Forbidden
#     mock_get.return_value = create_mock_response(403, {"message": "Invalid API Key"})
#     with pytest.raises(VitalLensAPIKeyError, match="Invalid API Key"):
#       _resolve_model_config("invalid_key", Method.VITALLENS)
        
#     # Test 500 Internal Server Error
#     mock_get.return_value = create_mock_response(500, {"message": "Server Error"})
#     with pytest.raises(VitalLensAPIError, match="Server Error"):
#       _resolve_model_config("valid_key", Method.VITALLENS)

def test_VitalLens_API_integration(mock_resolve_config, request):
  api_key = request.getfixturevalue('test_dev_api_key')
  method = VitalLensRPPGMethod(requested_model=Method.VITALLENS, api_key=api_key, mode=Mode.BATCH)
  assert method.resolved_model.name == 'VITALLENS_2_0'
  assert method.input_size == 40
  