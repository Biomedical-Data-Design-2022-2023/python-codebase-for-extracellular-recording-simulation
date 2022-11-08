# import modules
import LFPy
from LFPy import Cell, Synapse, LineSourcePotential
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import os
# create Cell
cell = Cell(morphology=''.join([os.getcwd(),'/sim/LFPy_single_cell_sim/morphologies/Y_example.hoc']),
            passive=True,  # NEURON 'pas' mechanism
            tstop=100,  # ms
            pt3d = True
           )
# create Synapse
synapse = Synapse(cell=cell,
                  idx=cell.get_idx("soma[0]"),  # soma segment index
                  syntype='Exp2Syn',  # two-exponential synapse
                  weight=0.005,  # max conductance (uS)
                  e=0,  # reversal potential (mV)
                  tau1=0.5,  # rise time constant
                  tau2=5.,  # decay time constant
                  record_current=True,  # record synapse current
                 )
synapse.set_spike_times(np.array([20., 40]))  # set activation times
# create extracellular predictor
lsp = LineSourcePotential(cell=cell,
                          x=np.zeros(11) + 10,  # x-coordinates of contacts (µm)
                          y=np.zeros(11),  # y-coordinates
                          z=np.arange(11)*20,  # z-coordinates
                          sigma=0.3,  # extracellular conductivity (S/m)
                         )
# execute simulation
cell.simulate(probes=[lsp])  # compute measurements at run time
# plot results
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
axes[0].plot(cell.tvec, synapse.i)
axes[0].set_ylabel('i_syn (nA)')
axes[1].plot(cell.tvec, cell.somav)
axes[1].set_ylabel('V_soma (nA)')
axes[2].pcolormesh(cell.tvec, lsp.z, lsp.data, shading='auto')
axes[2].set_ylabel('z (µm)')
axes[2].set_xlabel('t (ms)')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(221)
zips = []
for x, y in cell.get_pt3d_polygons(projection=('x', 'y')):
    zips.append(list(zip(y, x)))
polycol = PolyCollection(zips,
                         edgecolors='none',
                         facecolors='gray')
ax.add_collection(polycol)
ax.axis(ax.axis('equal'))
ax.set_xlabel('x')
ax.set_ylabel('y')

ax = fig.add_subplot(222)
zips = []
for z, y in cell.get_pt3d_polygons(projection=('z', 'y')):
    zips.append(list(zip(z, y)))
polycol = PolyCollection(zips,
                         edgecolors='none',
                         facecolors='gray')
ax.add_collection(polycol)
ax.axis(ax.axis('equal'))
ax.set_xlabel('z')
ax.set_ylabel('y')

ax = fig.add_subplot(223)
zips = []
for x, z in cell.get_pt3d_polygons(projection=('x', 'z')):
    zips.append(list(zip(x, z)))
polycol = PolyCollection(zips,
                         edgecolors='none',
                         facecolors='gray')
ax.add_collection(polycol)
ax.axis(ax.axis('equal'))
ax.set_xlabel('x')
ax.set_ylabel('z')
plt.show()

print('hello')