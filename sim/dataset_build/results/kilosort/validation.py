import numpy as np
import matplotlib.pyplot as plt
import pandas
import seaborn as sns
from sklearn.decomposition import PCA
# import umap # need numpy-1.21.6, use: pip install umap-learn
from confidence_ellipse import confidence_ellipse
from sklearn.metrics import jaccard_score
from scipy.optimize import linear_sum_assignment

import MEArec as mr
import MEAutility as mu

'''
See definition of result files:
https://github.com/kwikteam/phy-contrib/blob/master/docs/template-gui.md#analysis
'''

def gt_proc():
    sim_path = 'G:\\JHU\\Study\\Biomedical_Data_Design\\neurosim\\sim\\dataset_build\\process_Neuropixels2-64'
    sim_tem_file = 'templates_allen.h5'
    sim_rec_file = 'recordings_allen.h5'

    sim_tem = mr.load_templates(sim_path+'\\'+sim_tem_file)
    rec = mr.load_recordings(sim_path+'/'+sim_rec_file)
    # rawdata = np.array(rec.recordings[()]).astype(np.int16)

    sim_tem_cutout = sim_tem.params['params']['cut_out'] # before / after spike peak, 2 / 5 ms
    sim_tem_dt = sim_tem.params['params']['dt'] # template time period, corresponding to 32,000 Hz
    rec_fs = rec.params['recordings']['fs'] # 32,000 Hz

    templates = np.array(rec.templates[()]).astype(np.float32)
    templates_reshape = np.reshape(templates,(np.shape(templates)[0]*np.shape(templates)[1],np.shape(templates)[2],np.shape(templates)[3]))

    start = 141
    end = 202
    sim_tem_templates = templates_reshape[:,:,start:end] # corresponding to kilosort template time length
    sim_tem_templates = (sim_tem_templates - np.mean(sim_tem_templates,axis=(1,2),keepdims=True)) / np.std(sim_tem_templates,axis=(1,2),keepdims=True) * 0.016
    
    sim_tem_spiketrain = [np.array(rec.spiketrains[i]) for i in range(len(rec.spiketrains))]

    print('gt_proc done')
    return sim_tem_templates,sim_tem_spiketrain,rec_fs


def kilosort_proc():
    run = 10

    kilosort_tem_templates = []
    file = 'templates.npy'
    start = 21
    for id in range(run):
        data = np.load('.\\'+str(id)+'\\'+file)[:,start:,:]
        data = data - np.mean(data,axis=(1,2),keepdims=True)
        templates = np.transpose(data,(0,2,1)) # first 21 sampling is zero. kilosort_tem_templates, mean=0, std=0.016 
        
        kilosort_tem_templates.append(templates) 
        
    print('Read '+file+' done')

    kilosort_tem_labels = []
    file = 'cluster_KSLabel.tsv'
    # define: good = 1, mua = 0
    for id in range(run):
        with open('.\\'+str(id)+'\\'+file,'r') as f:
            content = pandas.read_csv(f,sep='\t')
            labels = [1 if list(content['KSLabel'])[i] == 'good' else 0 for i in range(len(list(content['KSLabel'])))]
            kilosort_tem_labels.append(labels)
    print('Read '+file+' done')

    file = ['spike_templates.npy','spike_times.npy']
    kilosort_spike_templates = []
    kilosort_spike_times = []
    for id in range(run):
        kilosort_spike_templates.append(np.load('.\\'+str(id)+'\\'+file[0]))
        kilosort_spike_times.append(np.load('.\\'+str(id)+'\\'+file[1]))

    print('Read '+file[0]+' and '+file[1]+' done')

    print('kilosort_proc done')
    return kilosort_tem_templates,kilosort_tem_labels,kilosort_spike_templates,kilosort_spike_times

if __name__ == '__main__':
    sim_tem_templates,sim_tem_spiketrain,rec_fs = gt_proc() # number*channel*time
    kilosort_tem_templates,kilosort_tem_labels,kilosort_spike_templates,kilosort_spike_times = kilosort_proc() # list, len=run, each is number_i * channel * time

    all_kilosort_tem_templates = np.concatenate(kilosort_tem_templates,axis=0)
    all_templates = np.concatenate([sim_tem_templates,all_kilosort_tem_templates],axis=0)
    all_temp_features = np.reshape(all_templates,(np.shape(all_templates)[0],np.shape(all_templates)[1]*np.shape(all_templates)[2]))

    # labels: sim = -1, run = 0-9
    all_run_labels = np.concatenate([np.ones(np.shape(kilosort_tem_templates[i])[0])*i for i in range(len(kilosort_tem_templates))]).astype(np.int32)
    
    sim_cell_labels = np.kron(np.arange(60),np.ones(10)).astype(np.int32)

    all_labels = np.concatenate([-np.ones(np.shape(sim_tem_templates)[0]).astype(np.int32),all_run_labels])
    
    # embedding

    pca = PCA(n_components=2)
    embedding = pca.fit_transform(all_temp_features)

    # embed = umap.UMAP()
    # embedding = embed.fit_transform(all_temp_features)

    fig,ax = plt.subplots(1,1)
    # plot gt
    ax.scatter(
    embedding[:600, 0],
    embedding[:600, 1],
    alpha = 1,
    c='k') # c=[sns.color_palette(n_colors=60)[x] for x in sim_cell_labels])
    for i in range(60):
        confidence_ellipse(embedding[10*i:10*(i+1), 0],embedding[10*i:10*(i+1), 1],ax,edgecolor=sns.color_palette(n_colors=60)[i])
    # plot results
    ax.scatter(
    embedding[600:, 0],
    embedding[600:, 1],
    alpha = 0.1,
    c = 'r') # c=[sns.color_palette(n_colors=10)[x] for x in all_run_labels])
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('Embedding of neuron units', fontsize=24)

    # spikes
    # sim
    fig,ax = plt.subplots(1,1)
    for i in range(len(sim_tem_spiketrain)):
        ax.scatter(sim_tem_spiketrain[i],np.ones(len(sim_tem_spiketrain[i]))*i,s=0.5)

    # kilosort
    for run_i in range(len(kilosort_spike_templates)):
        fig,ax = plt.subplots(1,1)
        for tem_i in range(np.shape(kilosort_tem_templates[run_i])[0]):
            ax.scatter(kilosort_spike_times[run_i][np.where(kilosort_spike_templates[run_i]==tem_i)]/rec_fs,np.ones(len(kilosort_spike_times[run_i][np.where(kilosort_spike_templates[run_i]==tem_i)]))*tem_i,s=0.5)
    
    # calc Jaccard
    # bool time sample matrix
    interval = 320
    for run_i in range(len(kilosort_spike_templates)):
        Jaccard_matrix = np.zeros((len(sim_tem_spiketrain),np.shape(kilosort_tem_templates[run_i])[0]))
        for sim_tem_i in range(len(sim_tem_spiketrain)):
            
            sim_tem_spiketrain_bool = [False for i in range(6000)]
            for T in sim_tem_spiketrain[sim_tem_i]:
                sim_tem_spiketrain_bool[np.floor(T*32000/interval).astype(np.int32)] = True
            
            for tem_i in range(np.shape(kilosort_tem_templates[run_i])[0]):
                
                kilosort_tem_spiketrain_bool = [False for i in range(6000)]
                for T in kilosort_spike_times[run_i][np.where(kilosort_spike_templates[run_i]==tem_i)]:
                    kilosort_tem_spiketrain_bool[np.floor(T/interval).astype(np.int32)] = True

                Jaccard_matrix[sim_tem_i,tem_i] = jaccard_score(sim_tem_spiketrain_bool,kilosort_tem_spiketrain_bool)
        
        answer = linear_sum_assignment(Jaccard_matrix,maximize=True)

        matrix = np.zeros_like(Jaccard_matrix)
        for i in range(60):
            matrix[:,i] = Jaccard_matrix[:,answer[1][i]]

        complementary = []
        for i in range(72):
            if i not in answer[1]:
                complementary.append(i)
        matrix[:,60:] = Jaccard_matrix[:,complementary]

        fig = plt.plot()
        plt.imshow(matrix)
        plt.xlabel('Kilosort3 Neuron Units')
        plt.ylabel('Simulated Neuron Units')
        plt.title('Correlation matrix of spike trains')
        plt.colorbar()



        print('Run '+str(run_i)+' Jaccard matrix finished')
            

            


    print('done')
