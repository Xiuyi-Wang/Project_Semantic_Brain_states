# calculate the spatial correlation between the semantic maps and the gradient maps
# at the individual level
import os
import nibabel as nib
from brainsmash.mapgen.base import Base
from brainsmash.mapgen.stats import pearsonr, pairwise_r
from scipy import stats
from scipy.io import loadmat
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
from contextlib import redirect_stdout

path_grad = '/home/xiuyi/Data/HCP/11_FC_xDF_z_mean_group/for_correlation_with_semantic_maps'
path_output = '/home/xiuyi/Data/Task_Gradient/results/fmriprep_xcp_abcd_Shaefer_indi/correlation_with_gradient/indi_level'
os.makedirs(path_output, exist_ok=True)

path_feat = '/home/xiuyi/Data/Task_Gradient/results/fmriprep_xcp_abcd_Shaefer_indi/feature_simi/level_2'
contrasts_feat =['Matching',
                 'Non-Matching',
                 'Matching_and_Non-Matching',
                 'Matching_pm',
                 'Matching_pm_neg',
                 'Non-Matching_pm',
                 'Non-Matching_vs_Matching-pm']


path_asso = '/home/xiuyi/Data/Task_Gradient/results/fmriprep_xcp_abcd_Shaefer_indi/association/level_2'
contrasts_asso = ['Associated',
                  'Non-Associated',
                  'Associated_yes_and_no',
                  'Associated_pm',
                  'Associated_pm_neg',
                  'Non-Associated_pm',
                  'Associated_no_vs_yes_pm']

folder_stat ='t_surf'

contrasts_all = []
grad_all = []
r_all = []
z_all = []
subj_all = []

for sem in ['feat','asso']:
    if sem == 'feat':
        path_sem = path_feat
        contrasts = contrasts_feat
        ses_ID = 'ses-01'
        task ='FeatureMatching'

    if sem == 'asso':
        path_sem = path_asso
        contrasts = contrasts_asso
        ses_ID = 'ses-02'
        task = 'Association'

    for contrast in contrasts:

        for grad_id in range(1,4):

            file_grad = os.path.join(path_grad,'HCP_rest_FC_gradient_%s_kernel-None_embedding-dm.ptseries.nii'%(grad_id))
            data_grad = nib.load(file_grad).get_fdata()[0]

            subj_IDs = os.listdir(path_sem)

            for subj_ID in subj_IDs:
                filename = '%s_%s_task-%s_%s_t_test_t_cope1.ptseries.nii'%(subj_ID,ses_ID,task,contrast)
                file_sem = os.path.join(path_sem,subj_ID,ses_ID,folder_stat,filename)

                if os.path.exists(file_sem):
                    data_sem = nib.load(file_sem).get_fdata()[0]

                    r, p = stats.pearsonr(data_sem, data_grad)
                    print(contrast,'grad %s'%(grad_id), r, p)
                    z =  np.arctanh(r)

                    contrasts_all.append(contrast)
                    grad_all.append(grad_id)
                    r_all.append(r)
                    z_all.append(z)
                    subj_all.append(subj_ID)



fig_all = os.path.join(path_output, 'sem_HCP_rest_FC_gradient_indi.png' )
file_stat = os.path.join(path_output, 'sem_HCP_rest_FC_gradient_indi.xlsx' )
file_stat_task_txt = os.path.join(path_output, 'sem_HCP_rest_FC_gradient_indi_compare_tasks.txt' )
file_stat_grad_txt = os.path.join(path_output, 'sem_HCP_rest_FC_gradient_indi_compare_grads.txt' )


#
# if not os.path.exists(fig_all):
if True:
    data = {'Group': contrasts_all,
            'r_value':  r_all,
            'z_value':  z_all,
            'Hue':grad_all,
            'Subj':subj_all}

    # Create a DataFrame from the data
    df = pd.DataFrame(data)

    # Plot the violin plot
    fig, ax = plt.subplots(figsize=(40,40))
    sns.barplot(x='Group', y='r_value', hue='Hue', data=df)

    # Add labels and title
    plt.xticks(rotation=90,fontsize=20)
    plt.yticks(fontsize=40)
    plt.xlabel('')
    plt.ylabel('r value',fontsize=40)
    plt.title('semantic task gradient correlation', fontsize=40)

    # Show the plot
    plt.savefig(fig_all)
    plt.close()

    # if not os.path.exists(file_stat):
    df.to_excel(file_stat_task_txt,index=False)

# calculate the t test;
# same gradient, compare different contrasts
for grad_id in range(1,4):

    for contrast_id in range(len(contrasts)):

        contrast_feat = contrasts_feat[contrast_id]
        contrast_asso = contrasts_asso[contrast_id]

        df_feat = df[(df['Hue'] ==grad_id) & (df['Group'] ==contrast_feat)]
        df_asso = df[(df['Hue'] ==grad_id) & (df['Group'] ==contrast_asso)]

        subj_feat = df_feat['Subj']
        subj_asso = df_asso['Subj']

        # find the intersection of two lists
        subj_common = list(set(subj_feat) & set(subj_asso))

        # Select rows where Column2 is in the list of values
        df_feat_common = df_feat[df_feat['Subj'].isin(subj_common)]
        df_asso_common = df_asso[df_asso['Subj'].isin(subj_common)]

        # Sort the DataFrame based on the values of Column1 in ascending order
        df_feat_common_sort = df_feat_common.sort_values(by='Subj')
        df_asso_common_sort = df_asso_common.sort_values(by='Subj')

        # get the z value
        data_feat =  df_feat_common_sort['z_value']
        data_asso =  df_asso_common_sort['z_value']

        # calculate the mean value
        mean_feat = np.mean( data_feat )
        mean_asso = np.mean( data_asso )

        t, p = ttest_rel(data_feat, data_asso)

        # save the correlation output to the txt document
        with open(file_stat_task_txt, 'a') as f:
            with redirect_stdout(f):

                print (contrast_feat,  contrast_asso, 'gradient ', grad_id)
                print ('mean z =  ', np.round(mean_feat,3), ' mean z = ', np.round(mean_asso,3))
                print ('t =     ',  np.round(t,3), '    p =    ', np.round(p,3))
                print ('                       ')
                print ('                       ')

# calculate the t test; compare different gradient
for contrast_id in range(len(contrasts)):

    contrast_feat = contrasts_feat[contrast_id]
    contrast_asso = contrasts_asso[contrast_id]


    df_feat_grad_1 = df[(df['Hue'] ==1) & (df['Group'] ==contrast_feat)]
    df_feat_grad_2 = df[(df['Hue'] ==2) & (df['Group'] ==contrast_feat)]
    df_feat_grad_3 = df[(df['Hue'] ==3) & (df['Group'] ==contrast_feat)]

    df_asso_grad_1 = df[(df['Hue'] == 1) & (df['Group'] == contrast_asso)]
    df_asso_grad_2 = df[(df['Hue'] == 2) & (df['Group'] == contrast_asso)]
    df_asso_grad_3 = df[(df['Hue'] == 3) & (df['Group'] == contrast_asso)]

    subj_feat = df_feat_grad_1['Subj']
    subj_asso = df_asso_grad_1['Subj']

    # find the intersection of two lists
    subj_common = list(set(subj_feat) & set(subj_asso))

    # Select rows where Column2 is in the list of values
    df_feat_grad_1_common =   df_feat_grad_1[df_feat_grad_1['Subj'].isin(subj_common)]
    df_feat_grad_2_common =   df_feat_grad_2[df_feat_grad_2['Subj'].isin(subj_common)]
    df_feat_grad_3_common =   df_feat_grad_3[df_feat_grad_3['Subj'].isin(subj_common)]

    df_asso_grad_1_common = df_asso_grad_1[df_asso_grad_1['Subj'].isin(subj_common)]
    df_asso_grad_2_common = df_asso_grad_2[df_asso_grad_2['Subj'].isin(subj_common)]
    df_asso_grad_3_common = df_asso_grad_3[df_asso_grad_3['Subj'].isin(subj_common)]

    # Sort the DataFrame based on the values of Column1 in ascending order
    df_feat_grad_1_common_sort = df_feat_grad_1_common.sort_values(by='Subj')
    df_feat_grad_2_common_sort = df_feat_grad_2_common.sort_values(by='Subj')
    df_feat_grad_3_common_sort = df_feat_grad_3_common.sort_values(by='Subj')

    df_asso_grad_1_common_sort = df_asso_grad_1_common.sort_values(by='Subj')
    df_asso_grad_2_common_sort = df_asso_grad_2_common.sort_values(by='Subj')
    df_asso_grad_3_common_sort = df_asso_grad_3_common.sort_values(by='Subj')

    # get the z value
    data_feat_grad_1 =  df_feat_grad_1_common_sort['z_value']
    data_feat_grad_2 =  df_feat_grad_2_common_sort['z_value']
    data_feat_grad_3 =  df_feat_grad_3_common_sort['z_value']

    data_asso_grad_1 =  df_asso_grad_1_common_sort['z_value']
    data_asso_grad_2 =  df_asso_grad_2_common_sort['z_value']
    data_asso_grad_3 =  df_asso_grad_3_common_sort['z_value']

    data_feat = [data_feat_grad_1, data_feat_grad_2, data_feat_grad_3]
    data_asso = [data_asso_grad_1, data_asso_grad_2, data_asso_grad_3]

    for i in range(2):
        data_feat_i = data_feat[i]
        data_asso_i = data_asso[i]

        mean_feat_i = np.mean(  data_feat_i )
        mean_asso_i = np.mean(  data_asso_i )
        for j in range(i+1,3):
            data_feat_j = data_feat[j]
            data_asso_j = data_asso[j]

            mean_feat_j = np.mean(data_feat_j)
            mean_asso_j = np.mean(data_asso_j)

            diff_feat =  np.array(data_feat_i.values)  - np.array( data_feat_j.values)
            diff_asso =  np.array(data_asso_i.values)  - np.array( data_asso_j.values)

            # calculate the mean value
            mean_feat = np.mean( diff_feat )
            mean_asso = np.mean( diff_asso )

            t, p = ttest_rel(diff_feat, diff_asso)

            # save the correlation output to the txt document
            with open(file_stat_grad_txt, 'a') as f:
                with redirect_stdout(f):
                    print (contrast_feat,'grad_%s = %s ; grad_%s = %s '%(str(i+1),   mean_feat_i, str(j+1) ,    mean_feat_j))
                    print (contrast_asso,'grad_%s = %s ; grad_%s = %s '%(str(i+1),   mean_asso_i, str(j+1) ,    mean_asso_j))

                    print (contrast_feat,  contrast_asso, ' gradient %s vs grad %s ' %(i+1,j+1))
                    print ('mean diff z =  ', np.round(mean_feat,3), ' mean diff z = ', np.round(mean_asso,3))
                    print ('t =     ',  np.round(t,3), '    p =    ', np.round(p,3))
                    print ('                       ')
                    print ('                       ')






