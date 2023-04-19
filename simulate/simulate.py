import numpy as np
from pathlib import Path
import neuron
import LFPy
import os
import json
import matplotlib.pyplot as plt
from pprint import pprint
import sys
import pickle as pkl

import MEArec as mr
import MEAutility as mu

from utils import create_empty_recordings, create_empty_templates

# Function to load Allen cells in LFPy, intracellular simulation
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

def generate_template(template_params, cell_model_path, cell_sim_model_path, template_path):
    """ wrapper to call mr.TemplateGenerator() with specifically Allen neuron model

    Args:
        template_params (dict): parameters for generating templates
        cell_model_path (path): path to the RAW Allen cell models 
        cell_sim_model_path (path): path to save the simulated cell models
        template_path (path): path to save the generated template .h5 file
    """

    pprint(template_params)
    
    ########## generate EAPs for all cell models and assembling a template library
    cell_types = ['spiny','sparsely_spiny','aspiny']

    spiny_cell_models = [p for p in Path().joinpath(cell_model_path,'spiny').iterdir()]
    print(spiny_cell_models)
    sparsely_spiny_cell_models = [p for p in Path().joinpath(cell_model_path,'sparsely_spiny').iterdir()]
    print(sparsely_spiny_cell_models)
    aspiny_cell_models = [p for p in Path().joinpath(cell_model_path,'aspiny').iterdir()]
    print(aspiny_cell_models)

    # init
    templates, template_locations, template_rotations, template_celltypes = [], [], [], []
    
    for cell in spiny_cell_models:
        
        # find cell type
        cell_type = cell_types[0]
        print("Cell", cell, "is", cell_type)

        try:
            eaps, locs, rots = mr.simulate_templates_one_cell(cell_model_path / cell_type / cell.name, intra_save_folder=cell_sim_model_path, 
                                                        params=template_params, verbose=True, 
                                                        custom_return_cell_function=return_allen_cell)
        except SystemExit:
            print("Cell", cell, "is bad. Excluded..")
            continue


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

    for cell in sparsely_spiny_cell_models:
        
        # find cell type
        cell_type = cell_types[1]
        print("Cell", cell, "is", cell_type)
        
        try:
            eaps, locs, rots = mr.simulate_templates_one_cell(cell_model_path / cell_type / cell.name, intra_save_folder=cell_sim_model_path, 
                                                        params=template_params, verbose=True, 
                                                        custom_return_cell_function=return_allen_cell)
        except SystemExit:
            print("Cell", cell, "is bad. Excluded..")
            continue
        
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

    for cell in aspiny_cell_models:
        
        # find cell type
        cell_type = cell_types[2]
        print("Cell", cell, "is", cell_type)
        
        try:
            eaps, locs, rots = mr.simulate_templates_one_cell(cell_model_path / cell_type / cell.name, intra_save_folder=cell_sim_model_path, 
                                                        params=template_params, verbose=True, 
                                                        custom_return_cell_function=return_allen_cell)
        except SystemExit:
            print("Cell", cell, "is bad. Excluded..")
            continue
        
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
    mr.save_template_generator(tempgen=tempgen, filename=(template_path / 'templates.h5').as_posix())

def generate_recording(rec_params, template_path, recording_path):
    """ wrapper to call mr.gen_recordings() with specifically Allen neuron model

    Args:
        rec_params (dict): parameters for generating recordings
        template_path (Path): path to templates
        recording_path (Path): path to save the generated recordings .h5 file
    """    

    pprint(rec_params)

    recgen = mr.gen_recordings(params=rec_params, templates=(template_path / 'templates.h5').as_posix(), verbose=2,tmp_folder='./')

    mr.save_recording_generator(recgen=recgen, filename=(recording_path / 'recordings.h5').as_posix())
