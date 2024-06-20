# %%
import numpy as np
import os
import nibabel as nib
import sys
current_directory = os.getcwd()
functions_directory = os.path.join(current_directory, '..', 'Functions')
sys.path.append(functions_directory)
from function_plot_Kong_ptseries_dlabel_use_fig_color import plot_Kong_parcellation_pos_neg

# %%
path_base_input = os.path.join(current_directory,'../Data/brain_map')
path_output = os.path.join(current_directory,'../Results/SM_Figure_2')
os.makedirs(path_output, exist_ok=True)

fig_num = 'SM_Fig_2d'

# %%
folder_FM_Asso = 'compare_FeatureMatching_with_Association/compare_feat_with_asso_Shaefer_indi/stat_maps'
suffix_FM_Asso = 'Matching_and_Non-Matching_vs_Associated_yes_and_no_t_FDR_0.05'

path_FM_Asso = os.path.join(path_base_input, folder_FM_Asso, '%s.ptseries.nii'%(suffix_FM_Asso))

nii_FM_Asso = nib.load(path_FM_Asso)
data_FM_Asso = nii_FM_Asso.get_fdata()[0]
data_FM_Asso_reverse = data_FM_Asso*(-1)

fig_FM_Asso_reverse_file = os.path.join(path_output, '%s_%s.png'%(fig_num, suffix_FM_Asso))
plot_Kong_parcellation_pos_neg(data_FM_Asso_reverse, fig_FM_Asso_reverse_file, color = 'psych', title='Association versus feature matching')


