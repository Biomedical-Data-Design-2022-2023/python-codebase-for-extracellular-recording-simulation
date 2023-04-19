# python-codebase-for-extracellular-recording-simulation

## Codebase structure
- algo (folder to save different sorting alogrithm source code)
  - *algoname1*
    - ...
  - *algoname2*
  - ...
- cell_models (folder to save raw cell model groups from Allen)
  - *cellmodelsgroup1*
  - *cellmodelsgroup2*
  - ...
- simulate (code for generating simulated datasets)
- compare (code for comparison)
- studies (folder to save several studies)
  - *studyname1*
    - cell_sim_model (folder to save simulated cell models in this study)
    - templates (can generate several templates from one group of cell_model, folder to save these templates)
      - *templatesname1*
        - templates.h5 (from `MEArec.save_template_generator()`, has params)
        - *probename*.yaml (a copy of probe file used in generating templates by MEArec)
        - template_params.pkl (params to generate this templates.h5)
      - *templatesname2*
      - ...
    - recordings (can generate several recordings from one set of templates, folder to save these recordings)
      - *recordingname1* 
        - recordings.h5 (from `MEArec.save_recording_generator()`, has params)
        - recordings.bin (extract only raw data as int16 bin file, as some sorting algos require)
        - recording_params.pkl (params to generate this recordings.h5)
        - info.pkl (general infomation of this recording, like path to cell_models, cell_sim_models, templates)
        - probe (folder to save probe files with different formats, as some sorting algos require; should have the same information to the probe file in template folder)
          - *probeforalgo1*.prb
          - *probeforalgo2*.mat
          - ...
        - sorting_results (folder to save all algos result files)
          - *algoname1*
          - *algoname2*
          - ...
        - comparing_results
          - fig (folder to save figure plots)
            - ...
          - ...
      - *recordingname2*
      - ...
  - *studyname2*
    - ...

## Comparing results

In general, we can have 2 categories of comparing results: spike time comparing and template waveform comparing.

### Spike time comparing
- `time_agreement_matrix()`
- `jaccard_score_matrix()`
- `ccg_matrix()` (from `pykilosort`)

### Template waveform comparing
- `cosine_similarity_matrix()`
- `simscore_matrix()` (svd, similar to `pykilosort`)

## About my study *mouse_VISp_L5_128ch*

This study was created for a systematic comparison between some different algorithms. The neural models are from The Allen Cell Type Database.

Algorithms:
- kilosort2.5 (KS25)
- kilosort3 (KS3)
- herdingspikes (HS2)
- spykingcircus (SC)
- spykingcircus2 (SC2)
- tridesclous (TC)
- tridesclous2 (TC2)
- mountainsort5 (MS5)

Generated recordings (simulated datasets):

|Condition|recording_toy|recording_onlydrift_slow|recording_onlydrift_fast|recording_onlydrift_slowANDfast|recording_drift|
|:-:|:-:|:-:|:-:|:-:|:-:|
|Jitter|$\surd$|$\surd$|$\surd$|$\surd$|$\surd$|
|Shape modulation|$\times$|$\times$|$\times$|$\times$|$\surd$|
|Bursting|$\surd$|$\surd$|$\surd$|$\surd$|$\surd$|
|Noise|distance-correlated|distance-correlated|distance-correlated|distance-correlated|distance-correlated|
|Drifting|$\times$|slow|fast|slow & fast|slow & fast|
