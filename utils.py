from pathlib import Path
import numpy as np

import MEArec as mr
import MEAutility as mu

def create_empty_study(study_name,study_dir):
    """ A function to generate empty folders of a new study

    Args:
        study_name (str): study name
        study_dir (Path): study dir to create the new study empty folder   

    Returns:
        Path: Path to the new study
    """     

    study_path =  Path(study_dir) / study_name
    study_path.mkdir(parents=True, exist_ok=False)

    p = Path().joinpath(study_path,'cell_sim_model')
    p.mkdir(parents=False, exist_ok=False)

    p = Path().joinpath(study_path,'templates')
    p.mkdir(parents=False, exist_ok=False)

    p = Path().joinpath(study_path,'recordings')
    p.mkdir(parents=False, exist_ok=False)

    return study_path

def create_empty_templates(study_path, template_name):
    """ A function to generate empty templates in a study

    Args:
        study_path (Path): Path to the study
        template_name (str): template name   

    Returns:
        Path: Path to the new templates
    """    

    p = Path().joinpath(study_path,'templates',template_name)
    p.mkdir(parents=False, exist_ok=False)

    return p

def create_empty_recordings(study_path, recording_name):
    """ A function to generate empty recordings in a study

    Args:
        study_path (Path): Path to the study
        recording_name (str): recording name

    Returns:
        Path: Path to the new recordings
    """      

    recording_path = Path().joinpath(study_path,'recordings',recording_name)
    recording_path.mkdir(parents=False, exist_ok=False)

    p = Path().joinpath(recording_path,'probe')
    p.mkdir(parents=False, exist_ok=False)

    p = Path().joinpath(recording_path,'sorting_results')
    p.mkdir(parents=False, exist_ok=False)

    p = Path().joinpath(recording_path,'comparing_results')
    p.mkdir(parents=False, exist_ok=False)

    return recording_path

def create_empty_sortings(recording_path,algo_list):
    """ A function to generate empty sorting result folders for algo_list 

    Args:
        recording_path (Path): Path to a specific recording
        algo_list (list): name of algos

    Returns:
        list: Path list to each algo sorting results folder
    """    

    sorting_path = recording_path / 'sorting_results'

    sorting_path_list = []

    for i,name in enumerate(algo_list):
        p = sorting_path / name
        p.mkdir(parents=False,exist_ok=False)
        sorting_path_list.append(p)

    return sorting_path

def extract_rawdata_bin(recording_path):
    """ Some algorithms accept rawdata bin file

    Args:
        recording_path (Path): Path to recording
    """    

    rec = mr.load_recordings((recording_path / 'recordings.h5').as_posix())

    rawdata = np.array(rec.recordings[()]).astype(np.int16)

    bytedata = bytes(rawdata)
    with open((recording_path / 'recordings_rawdata.bin').as_posix(),'wb') as new_f:
        new_f.write(bytedata)

def get_GT_templates(recording_path,ms_before=1,ms_after=2):
    """get GT templates from h5 file

    Args:
        recording_path (Path): path to recording
        ms_before (int, optional): template time before spike peak (msec). Defaults to 1.
        ms_after (int, optional): template time after spike peak (msec). Defaults to 2.

    Raises:
        ValueError: Wrong rec.templates shape!

    Returns:
        ndarray: GT templates (N,ch,(ms_before+ms_after)*fs/1e3)
    """    

    rec = mr.load_recordings(recording_path / 'recordings.h5')
    sh = rec.templates.shape
    peak = np.unravel_index(np.argmax(np.abs(rec.templates)),rec.templates.shape)
    peak_idx = peak[-1]
    fs = rec.params['recordings']['fs']

    templates = rec.templates.astype(np.float32)
    if len(sh) == 5: # drift & jitter
        templates = templates[:,:,:,:,peak_idx-int(np.floor((ms_before)*fs/1e3)):peak_idx+int(np.floor((ms_after)*fs/1e3))]
        templates = np.mean(templates,axis=1)
        templates = np.mean(templates,axis=1)
    elif len(sh) == 4: # jitter
        templates = templates[:,:,:,peak_idx-int(np.floor((ms_before)*fs/1e3)):peak_idx+int(np.floor((ms_after)*fs/1e3))]
        templates = np.mean(templates,axis=1)    
    else:
        raise ValueError('Wrong rec.templates shape!')

    return templates