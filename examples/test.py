
import sys
sys.path.append('../vitallens-python')
import vitallens

vl = vitallens.VitalLens()
vl(inputs='examples/test.mp4')
