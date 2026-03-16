import json
import numpy as np
import os
import pytest
from unittest.mock import patch

import sys
sys.path.append('../vitallens-python')

from vitallens.client import VitalLens, Method

@pytest.mark.parametrize("method_arg", [Method.G, "chrom", "pos"])
@pytest.mark.parametrize("detect_faces", [True, False])
@pytest.mark.parametrize("file", [True, False])
@pytest.mark.parametrize("export", [True, False])
def test_VitalLens(request, method_arg, detect_faces, file, export):
  vl = VitalLens(method=method_arg, detect_faces=detect_faces, export_to_json=export)
  if file:
    test_video_path = request.getfixturevalue('test_video_path')
    result = vl(test_video_path, faces = None if detect_faces else [247, 57, 440, 334], export_filename="test")
  else:
    test_video_ndarray = request.getfixturevalue('test_video_ndarray')
    test_video_fps = request.getfixturevalue('test_video_fps')
    result = vl(test_video_ndarray, fps=test_video_fps, faces = None if detect_faces else [247, 57, 440, 334], export_filename="test")
  assert len(result) == 1
  assert np.asarray(result[0]['face']['coordinates']).shape == (630, 4)
  assert np.asarray(result[0]['face']['confidence']).shape == (630,)
  assert np.asarray(result[0]['waveforms']['ppg_waveform']['data']).shape == (630,)
  assert np.asarray(result[0]['waveforms']['ppg_waveform']['confidence']).shape == (630,)
  np.testing.assert_allclose(result[0]['vitals']['heart_rate']['value'], 60, atol=10)
  assert result[0]['vitals']['heart_rate']['confidence'] == 1.0
  if export:
    test_json_path = os.path.join("test.json")
    assert os.path.exists(test_json_path)
    with open(test_json_path, 'r') as f:
      data = json.load(f)
    assert np.asarray(data[0]['face']['coordinates']).shape == (630, 4)
    assert np.asarray(data[0]['face']['confidence']).shape == (630,)
    assert np.asarray(data[0]['waveforms']['ppg_waveform']['data']).shape == (630,)
    assert np.asarray(data[0]['waveforms']['ppg_waveform']['confidence']).shape == (630,)
    np.testing.assert_allclose(data[0]['vitals']['heart_rate']['value'], 60, atol=10)
    assert data[0]['vitals']['heart_rate']['confidence'] == 1.0
    os.remove(test_json_path)

def test_VitalLens_API(request):
  api_key = request.getfixturevalue('test_dev_api_key')
  vl = VitalLens(method=Method.VITALLENS, api_key=api_key, detect_faces=True, export_to_json=False)
  test_video_ndarray = request.getfixturevalue('test_video_ndarray')
  test_video_fps = request.getfixturevalue('test_video_fps')
  result = vl(test_video_ndarray, fps=test_video_fps, faces=None, export_filename="test")
  assert len(result) == 1
  assert np.asarray(result[0]['face']['coordinates']).shape == (630, 4)
  assert np.asarray(result[0]['waveforms']['ppg_waveform']['data']).shape == (630,)
  assert np.asarray(result[0]['waveforms']['ppg_waveform']['confidence']).shape == (630,)
  assert np.asarray(result[0]['waveforms']['respiratory_waveform']['data']).shape == (630,)
  assert np.asarray(result[0]['waveforms']['respiratory_waveform']['confidence']).shape == (630,)
  assert 'hrv_sdnn' in result[0]['vitals']
  assert 'hrv_rmssd' in result[0]['vitals']
  np.testing.assert_allclose(result[0]['vitals']['heart_rate']['value'], 60, atol=0.5)
  np.testing.assert_allclose(result[0]['vitals']['heart_rate']['confidence'], 1.0, atol=0.1)
  np.testing.assert_allclose(result[0]['vitals']['respiratory_rate']['value'], 13, atol=1.0)
  np.testing.assert_allclose(result[0]['vitals']['respiratory_rate']['confidence'], 1.0, atol=0.1)
  supports_hrv = result[0].get('model_used', '') == 'vitallens-2.0'
  if supports_hrv:
    np.testing.assert_allclose(result[0]['vitals']['hrv_sdnn']['value'], 40, atol=20)
    np.testing.assert_allclose(result[0]['vitals']['hrv_rmssd']['value'], 30, atol=20)
  assert not os.path.exists("test.json")

def test_VitalLens_proxies():
  """Test that proxies are passed down correctly."""
  proxies = {"https": "http://10.10.1.10:3128"}
  with patch('vitallens.client.VitalLensRPPGMethod') as MockRPPG:
    _ = VitalLens(method="vitallens-2.0", api_key="test", proxies=proxies)
    MockRPPG.assert_called_with(
      api_key="test",
      requested_model_name="vitallens-2.0",
      proxies=proxies
    )

def test_VitalLens_auth_offloading():
  """Test initialization without API key (for proxy auth offloading)."""
  proxies = {"https": "http://auth-proxy:8080"}
  with patch('vitallens.client.VitalLensRPPGMethod') as MockRPPG:
    _ = VitalLens(method="vitallens", api_key=None, proxies=proxies)
    MockRPPG.assert_called_with(
      api_key=None,
      requested_model_name="vitallens",
      proxies=proxies
    )