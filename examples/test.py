
import sys
sys.path.append('../vitallens-python')
import vitallens

vl = vitallens.VitalLens(method=vitallens.Method.POS)
result = vl(video='examples/test.mp4')

import matplotlib.pyplot as plt

print(result)

plt.plot(result[0]['pulse']['sig'])
plt.show()
