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

import numpy as np
import pytest

import sys
sys.path.append('../vitallens-python')

from vitallens.enums import Method
from vitallens.signal import reassemble_from_windows, assemble_results

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

@pytest.mark.parametrize("signals", [('ppg_waveform',), ('respiratory_waveform',), ('ppg_waveform','respiratory_waveform')])
@pytest.mark.parametrize("can_provide_confidence", [True, False])
@pytest.mark.parametrize("min_t_too_long", [True, False])
def test_assemble_results(signals, can_provide_confidence, min_t_too_long):
  sample_video_hr = 73.2
  sample_video_rr = 15.
  sig_ppg_ir = [-0.30989337,-0.34880125,-0.51679734,-0.8264251,-1.16504221,-1.42803273,-1.50548274,-1.27397076,-0.75049456,-0.06296925,0.55622022,1.03613509,1.33531581,1.49509228,1.48178068,1.37909062,1.2243668,1.12657893,1.0408531,0.88764567,0.69828361,0.53705074,0.51951649,0.53721108,0.56165349,0.50940172,0.41704494,0.26956212,0.1083913,-0.01021207,-0.05248365,-0.05707812,0.03133779,0.11226076,0.22743949,0.11899808,-0.07210865,-0.30002545,-0.33543694,-0.43031631,-0.47661662,-0.59501978,-0.63097275,-0.74791876,-0.87790608,-1.05945451,-1.24667428,-1.49371626,-1.6371542,-1.71251337,-1.71386794,-1.75321702,-1.82464616,-1.83582955,-1.73761775,-1.64881252,-1.06284365,-0.2302026,0.87515657,1.56372653,1.93155176,2.02379647,1.98393698,1.87873691,1.74305679,1.48084475,1.19594038,0.90211362,0.73778882,0.6274904,0.56723345,0.47625067,0.37326869,0.24843488,0.11191973,-0.04886665,-0.24323905,-0.42655308,-0.58504122,-0.72104162,-0.80480045,-0.82562858,-0.50066783,-0.05626648,0.5518156,0.6590093,0.73628254,0.65119874,0.77386199,0.72584785,0.65558003,0.49750651,0.39234535,0.21854192,0.14756766,0.08148715,0.06862309,-0.05206853,-0.19597003,-0.38785157,-0.53376963,-0.72356122,-0.924077,-1.120926,-1.28434747,-1.43064954,-1.5675398,-1.43950115,-0.81664647,0.16796192,1.08697704,1.55443842,1.68625269,1.52211757,1.39033429,1.1453985,0.93370787,0.57538922,0.22143884,-0.08772584,-0.33665,-0.48391509,-0.61660534,-0.72096744,-0.87922138,-1.07761103,-1.28375948,-1.51790998,-1.68144848,-1.47123422,-1.10773505,-0.47217152,0.0946784,0.83295787,1.28496731,1.42196718,1.25495147,1.02965942,0.8611786]
  sig_resp = [1.51216802,1.50955699,1.51529447,1.52743288,1.65720539,1.7819858,1.77276978,1.63947779,1.47796079,1.30538046,1.13795781,0.99811216,0.82452553,0.71412734,0.72596083,0.76797981,0.76781212,0.69270434,0.58791597,0.50698886,0.47142443,0.35067527,0.2118323,0.16459586,0.22390303,0.40910957,0.5626543,0.65655808,0.64031414,0.69680373,0.75727817,0.74380619,0.62069997,0.43008055,0.31596059,0.23553526,0.18816279,0.09700333,-0.08084037,-0.23348016,-0.33336491,-0.42039115,-0.56651639,-0.75424524,-0.94991853,-1.14872326,-1.28733549,-1.42709312,-1.55680766,-1.67107495,-1.73852152,-1.79410228,-1.83959645,-1.85642986,-1.87528841,-1.85077548,-1.83781359,-1.83267193,-1.85641745,-1.90721566,-1.95186978,-1.99866851,-2.03258962,-2.10955241,-2.13515795,-2.12067511,-2.04354335,-1.91670117,-1.73937042,-1.59369458,-1.40989926,-1.18404334,-0.97035116,-0.758191,-0.6093685,-0.45600976,-0.36745855,-0.25499029,-0.17721783,-0.08083689,-0.00516917,0.05230045,0.10007278,0.20457223,0.3421804,0.45500407,0.54078405,0.51011294,0.46511979,0.41234301,0.38203958,0.31548583,0.21042761,0.09049446,-0.03285828,-0.05941505,-0.03440629,0.00453256,0.03482003,0.02501604,0.05357855,0.0906964,0.14722884,0.20599569,0.23433949,0.24324672,0.25765686,0.27466344,0.25850446,0.28364076,0.34324296,0.44705979,0.50849328,0.55803365,0.57982195,0.58001413,0.61137475,0.68543343,0.72630508,0.64339287,0.56898251,0.53510479,0.54739451,0.63840937,0.6771955,0.64794777,0.62852041,0.5871531,0.63854356,0.6999562,0.76363539,0.76443195,0.73940662,0.70987721,0.61985124,0.61910382,0.56261351,0.46039093,0.37440005]
  sig, train_signals, pred_signals = [], [], []
  for s in ('ppg_waveform','respiratory_waveform'):
    if s in signals:
      sig.append(sig_ppg_ir if s == 'ppg_waveform' else sig_resp)
      train_signals.append('ppg_waveform' if s == 'ppg_waveform' else 'respiratory_waveform')
      pred_signals.append('ppg_waveform' if s == 'ppg_waveform' else 'respiratory_waveform')
      pred_signals.append('heart_rate' if s == 'ppg_waveform' else 'respiratory_rate')
  sig = np.asarray(sig)
  conf = np.ones_like(sig)
  if can_provide_confidence:
    live = np.asarray([0.01083279,0.005833,0.00820795,0.32068461,0.72727495,0.73995125,0.68828821,0.76356381,0.86223269,0.66235924,0.8745808,0.9736979,0.99009115,0.99616373,0.98721313,0.98010886,0.98385429,0.9717201,0.93585575,0.8679142,0.90865433,0.93440485,0.98587644,0.96389133,0.97187209,0.92145288,0.95829332,0.95605373,0.98511922,0.99141932,0.98829323,0.98694253,0.99081576,0.88806206,0.95761001,0.8804996,0.87235355,0.93236756,0.92263389,0.67303586,0.76512074,0.66042209,0.72830695,0.6841411,0.57755405,0.57206297,0.875808,0.8154847,0.91685879,0.91369033,0.87684751,0.90468764,0.88777065,0.84473473,0.8203187,0.82623559,0.93446434,0.989048,0.98233426,0.99410701,0.9945882,0.99369907,0.99628174,0.9965958,0.99624705,0.99346352,0.9840489,0.96598941,0.98651248,0.97259837,0.97288722,0.94726074,0.97161913,0.9443177,0.97780919,0.92675537,0.97956222,0.97669303,0.98266715,0.98055196,0.98956901,0.99485874,0.96439457,0.97913766,0.97569704,0.97491169,0.97939396,0.98797679,0.97168183,0.97424555,0.97713304,0.93872136,0.94148386,0.78307801,0.86659586,0.78297913,0.85506177,0.66137809,0.7253961,0.92985141,0.93096375,0.87163657,0.90946668,0.96046972,0.96768141,0.93454564,0.92034447,0.96902579,0.87022018,0.98491764,0.98937058,0.98426282,0.99290872,0.9882791,0.99606001,0.99252284,0.99147999,0.95046616,0.9427911,0.93811965,0.93727344,0.90602505,0.94663155,0.89995974,0.95777148,0.94477475,0.97495407,0.94638491,0.9741165,0.94961405,0.9778741,0.96273482,0.97291636,0.95187879,0.99234462,0.98897183,0.99798048,0.99894512,0.99634731])
  else:
    live = np.ones((sig.shape[1],))
  fps = 30.
  out_data, out_conf, out_live, out_note, out_live = assemble_results(sig=sig,
                                                                      conf=conf,
                                                                      live=live,
                                                                      fps=fps,
                                                                      train_sig_names=train_signals,
                                                                      pred_signals=pred_signals,
                                                                      method=Method.G,
                                                                      can_provide_confidence=True,
                                                                      min_t_hr=8. if min_t_too_long else 2.,
                                                                      min_t_rr=8. if min_t_too_long else 4.)
  if 'ppg_waveform' in signals:
    np.testing.assert_allclose(out_data['ppg_waveform'], sig_ppg_ir)
    'ppg_waveform' in out_note['ppg_waveform']
    if min_t_too_long:
      'heart_rate' not in out_data
    else:
      np.testing.assert_allclose(out_data['heart_rate'], sample_video_hr, atol=0.5)
  if 'respiratory_waveform' in signals:
    np.testing.assert_allclose(out_data['respiratory_waveform'], sig_resp)
    'respiratory_waveform' in out_note['respiratory_waveform']
    if min_t_too_long:
      'respiratory_rate' not in out_data
    else:
      np.testing.assert_allclose(out_data['respiratory_rate'], sample_video_rr, atol=1.)
