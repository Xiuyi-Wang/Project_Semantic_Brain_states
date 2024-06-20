# calculate the spatial correlation between the semantic maps and the gradient maps
# at the group level
import os
import nibabel as nib
from brainsmash.mapgen.base import Base
from brainsmash.mapgen.stats import pearsonr, pairwise_r
from scipy import stats
from scipy.io import loadmat
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

path_grad = '/home/xiuyi/Data/HCP/11_FC_xDF_z_mean_group/for_correlation_with_semantic_maps'
path_output = '/home/xiuyi/Data/Task_Gradient/results/fmriprep_xcp_abcd_Shaefer_indi/correlation_with_gradient/group_level'
os.makedirs(path_output, exist_ok=True)

path_feat = '/home/xiuyi/Data/Task_Gradient/results/fmriprep_xcp_abcd_Shaefer_indi/feature_simi/level_3/ses-01/stat_FeatureMatching_thre_p_no_thre'
contrasts_feat =['Matching',
                 'Non-Matching',
                 'Matching_and_Non-Matching',
                 'Matching_pm',
                 'Matching_pm_neg',
                 'Non-Matching_pm',
                 'Non-Matching_vs_Matching-pm']


path_asso = '/home/xiuyi/Data/Task_Gradient/results/fmriprep_xcp_abcd_Shaefer_indi/association/level_3/ses-02/stat_Association_thre_p_no_thre'
contrasts_asso = ['Associated',
                  'Non-Associated',
                  'Associated_yes_and_no',
                  'Associated_pm',
                  'Associated_pm_neg',
                  'Non-Associated_pm',
                  'Associated_no_vs_yes_pm']

contrasts_all = []
grad_all = []
r_all = []

for sem in ['feat','asso']:
    if sem == 'feat':
        path_sem = path_feat
        contrasts = contrasts_feat

    if sem == 'asso':
        path_sem = path_asso
        contrasts = contrasts_asso

    for contrast in contrasts:
        file_sem = os.path.join(path_sem,'%s_t_map.ptseries.nii'%(contrast))

        data_sem = nib.load(file_sem).get_fdata()[0]

        r_all_each_contrast = []
        for grad_id in range(1,4):
            file_grad = os.path.join(path_grad,'HCP_rest_FC_gradient_%s_kernel-None_embedding-dm.ptseries.nii'%(grad_id))
            data_grad = nib.load(file_grad).get_fdata()[0]

            r, p = stats.pearsonr(data_sem, data_grad)
            print(contrast,'grad %s'%(grad_id), r, p)
            r_all_each_contrast.append(r)

            contrasts_all.append(contrast)
            grad_all.append(grad_id)
            r_all.append(r)



fig_all = os.path.join(path_output, 'sem_HCP_rest_FC_gradient_all.png' )
file_stat = os.path.join(path_output, 'sem_HCP_rest_FC_gradient_all.xlsx' )


if not os.path.exists(fig_all):

    data = {'Group': contrasts_all,
            'Value':  r_all,
            'Hue':grad_all}

    # Create a DataFrame from the data
    df = pd.DataFrame(data)

    # Plot the violin plot
    fig, ax = plt.subplots(figsize=(35,48))
    sns.stripplot(x='Group', y='Value', hue='Hue', data=df, size=25,jitter=True)

    # Add labels and title
    plt.xticks(rotation=90,fontsize=20)
    plt.yticks(fontsize=40)
    plt.xlabel('')
    plt.ylabel('r value',fontsize=40)
    plt.title('semantic task gradient correlation', fontsize=40)

    # Show the plot
    plt.savefig(fig_all)
    plt.close()

    if not os.path.exists(file_stat):
        df.to_excel(file_stat,index=False)