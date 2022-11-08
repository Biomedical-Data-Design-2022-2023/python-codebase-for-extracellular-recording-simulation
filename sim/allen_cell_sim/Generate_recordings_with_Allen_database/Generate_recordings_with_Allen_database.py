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

# Function to load Allen cells in LFPy
def return_allen_cell(cell_model_folder, dt=2**-5, start_T=0, end_T=1):    
    cell_model_folder = Path(cell_model_folder)
    cwd = os.getcwd()
    os.chdir(cell_model_folder)
    
    # compile mechanisms
    mod_folder = "modfiles"
    os.chdir(mod_folder)
    os.system('nrnivmodl') # nrnivmodl is a NEURON command, compile special version of NEURON with custom mechanisms
    os.chdir('..')
    neuron.load_mechanisms(mod_folder)
    params = json.load(open("fit_parameters.json", 'r'))

    celsius = params["conditions"][0]["celsius"] # temperature
    reversal_potentials = params["conditions"][0]["erev"]
    v_init = params["conditions"][0]["v_init"]
    active_mechs = params["genome"]
    neuron.h.celsius = celsius

    cell_parameters = {
        'morphology': 'reconstruction.swc', # swc file
        'v_init': v_init,  # initial membrane potential
        'passive': False,  # turn on NEURONs passive mechanism for all sections
        'nsegs_method': 'lambda_f',  # spatial discretization method
        'lambda_f': 200.,  # frequency where length constants are computed
        'dt': dt,  # simulation time step size
        'tstart': start_T,  # start time of simulation, recorders start at t=0
        'tstop': end_T,  # stop simulation at 100 ms.
    }

    cell = LFPy.Cell(**cell_parameters)

    # check and add active_mechs and reversal_potentials to neuron.h 
    for sec in neuron.h.allsec():
        sec.insert("pas")
        sectype = sec.name().split("[")[0]
        for sec_dict in active_mechs:
            if sec_dict["section"] == sectype:
                # print(sectype, sec_dict)
                if not sec_dict["mechanism"] == "":
                    sec.insert(sec_dict["mechanism"])
                exec ("sec.{} = {}".format(sec_dict["name"], sec_dict["value"]))

        for sec_dict in reversal_potentials:
            if sec_dict["section"] == sectype:
                # print(sectype, sec_dict)
                for key in sec_dict.keys():
                    if not key == "section":
                        exec ("sec.{} = {}".format(key, sec_dict[key]))
    
    os.chdir(cwd)

    return cell

def plot_cell_projections(cell):
    fig = plt.figure()
    ax_xy = fig.add_subplot(2,2,1)
    ax_xz = fig.add_subplot(2,2,2)    
    ax_yz = fig.add_subplot(2,2,3)    
    
    for i, (x,y,z,d) in enumerate(zip(cell.x, cell.y, cell.z,cell.d)):
        if i in cell.get_idx('soma'):
            ax_xy.plot(x, y, color='k', lw=d)
            ax_xz.plot(x, z, color='k', lw=d)
            ax_yz.plot(y, z, color='k', lw=d)
        else:
            ax_xy.plot(x, y,lw=d)
            ax_xz.plot(x, z,lw=d)
            ax_yz.plot(y, z,lw=d)
        
    ax_xy.axis('equal')
    ax_xz.axis('equal')
    ax_yz.axis('equal')
    ax_xy.set_xlabel('x')
    ax_xy.set_ylabel('y')
    ax_xz.set_xlabel('x')
    ax_xz.set_ylabel('z')
    ax_yz.set_xlabel('y')
    ax_yz.set_ylabel('z')

    return fig

if __name__ == '__main__':
    cell_folder = './sim/allen_cell_sim/morphologies/mouse_VISI_L4/'
    # cell = return_allen_cell(cell_folder)
    # fig = plot_cell_projections(cell)

    template_params = mr.get_default_templates_params()
    template_params['seed'] = 0
    pprint(template_params)
    
    # cell, v, i = mr.run_cell_model(cell_folder, verbose=True, save=False, 
    #                            custom_return_cell_function=return_allen_cell, 
    #                            **template_params)

    # plot somatic membrane potential and transmembrane current
    # fig = plt.figure()
    # ax_v = fig.add_subplot(1,2,1)
    # ax_i = fig.add_subplot(1,2,2)
    # _ = ax_v.plot(v.T)
    # _ = ax_i.plot(i[:, 0, :].T)

    ######## simulate extracellular action potentials
    # 3D rotate cells
    template_params['rot'] = '3drot'
    # generate 10 templates (random rotation, location)
    template_params['n'] = 10

    # MEA probe
    print(mu.return_mea_list())
    template_params['probe'] = 'Neuropixels2-64'
    
    # simulate 
    # eaps, locs, rots = mr.simulate_templates_one_cell(cell_folder, 
    #                                               intra_save_folder='allen_sim', params=template_params,
    #                                               verbose=True, custom_return_cell_function=return_allen_cell)

    # check
    # print(eaps.shape)
    # print(locs.shape)
    # print(rots.shape)

    ########## generate EAPs for all cell models and assembling a template library
    cell_models = [p for p in Path('./sim/allen_cell_sim/morphologies/mouse_VISI_L4/').iterdir()]
    print(cell_models)

    cell_types = {'487245118': 'spiny','479427369': 'spiny'} 

    # init
    templates, template_locations, template_rotations, template_celltypes = [], [], [], []
    for cell in cell_models:
        eaps, locs, rots = mr.simulate_templates_one_cell(cell_folder+cell.name, intra_save_folder='allen_sim', 
                                                        params=template_params, verbose=True, 
                                                        custom_return_cell_function=return_allen_cell)
        # find cell type
        cell_type = None
        for k, v in cell_types.items():
            if k in str(cell):
                cell_type = v
                break
        print("Cell", cell, "is", cell_type)
        
        # if first cell, initialize the arrays
        if len(templates) == 0:
            templates = eaps
            template_locations = locs
            template_rotations = rots
            template_celltypes = np.array([cell_type]*len(eaps))
        else:
            templates = np.vstack((templates, eaps))
            template_locations = np.vstack((template_locations, locs))
            template_rotations = np.vstack((template_rotations, rots))
            template_celltypes = np.concatenate((template_celltypes, np.array([cell_type]*len(eaps))))

    # check
    print(templates.shape)
    print(template_locations.shape)
    print(template_rotations.shape)
    print(template_celltypes.shape)

    # build a TemplateGenerator object
    temp_dict = {'templates': templates, 
                'locations': template_locations, 
                'rotations': template_rotations,
                'celltypes': template_celltypes}
    info = {}
    info['params'] = template_params
    info['electrodes'] = mu.return_mea_info(template_params['probe'])
    tempgen = mr.TemplateGenerator(temp_dict=temp_dict, info=info)
    mr.plot_templates(tempgen)
    mr.save_template_generator(tempgen=tempgen, filename='allen/templates_allen.h5')

    ################# generate recordings
    rec_params = mr.get_default_recordings_params()
    pprint(rec_params)
    # tell excitatory and inhibitory
    rec_params['cell_types'] = {'excitatory': ['spiny'], 'inhibitory': ['aspiny']}
    # simulate params: 30 second, excitatory,  inhibitory, um minimium distance between cells, uV minimium amplitude 
    rec_params['spiketrains']['duration'] = 30
    rec_params['spiketrains']['n_exc'] = 5 # less than sum of templates
    rec_params['spiketrains']['n_inh'] = 0
    rec_params['templates']['min_dist'] = 5
    rec_params['templates']['min_amp'] = 30

    recgen = mr.gen_recordings(params=rec_params, templates='allen/templates_allen.h5', verbose=True)

    ax_st = mr.plot_rasters(recgen.spiketrains)
    ax_temp = mr.plot_templates(recgen)
    ax_rec = mr.plot_recordings(recgen, start_time=0, end_time=5, overlay_templates=True, lw=0.5)

    mr.save_recording_generator(recgen=recgen, filename='allen/recordings_allen.h5')

    print('Done!')