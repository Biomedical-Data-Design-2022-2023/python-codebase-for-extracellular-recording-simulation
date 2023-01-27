import numpy as np
from pathlib import Path
import neuron
import LFPy
import os
import json
import matplotlib.pylab as plt
from pprint import pprint
import sys

import MEArec as mr
import MEAutility as mu

tem = mr.load_templates('./process_Neuropixels2-64/templates_allen.h5')

rec = mr.load_recordings('./process_Neuropixels2-64/recordings_allen.h5')
rawdata = np.array(rec.recordings[()]).astype(np.int16)

bytedata = bytes(rawdata)
with open('./process_Neuropixels2-64/recordings_eawdata.bin','wb') as new_f:
    new_f.write(bytedata)

print('done')