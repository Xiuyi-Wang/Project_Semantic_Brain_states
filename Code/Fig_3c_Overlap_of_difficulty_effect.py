# %%
import numpy as np
import os
import sys
import nibabel as nib
current_directory = os.getcwd()
functions_directory = os.path.join(current_directory, '..', 'Functions')
sys.path.append(functions_directory)
from function_plot_Kong_ptseries_dlabel_specific_parcels import plot_Kong_parcellation

# %%
#define paths
path_base_input = os.path.join(current_directory,'../Data/brain_map')
path_base_output = os.path.join(current_directory,'../Results/Figure_3')
os.makedirs(path_base_output, exist_ok=True)

fig_num = 'Fig_3c'

folder_FM = 'stat_FeatureMatching_FDR_p_0.05'
folder_Asso = 'stat_Association_FDR_p_0.05'
folder_Spatial = 'stat_Spatial_FDR_p_0.05'
folder_Math = 'stat_Maths_FDR_p_0.05'

suffix_FM = 'Non-Matching_vs_Matching-pm_t_map_FDR_0.05'
suffix_Asso = 'Associated_no_vs_yes_pm_t_map_FDR_0.05'
suffix_Spatial = 'task-Spatial_contrast-Hard_vs_Easy_t_map_FDR_0.05'
suffix_Math = 'task-Maths_contrast-Hard_vs_Easy_t_map_FDR_0.05'
suffix_conjunction = 'Matching_Associated_conjunction_t_FDR_0.05'

path_FM = os.path.join(path_base_input, folder_FM, '%s.ptseries.nii'%(suffix_FM))
path_Asso = os.path.join(path_base_input, folder_Asso, '%s.ptseries.nii'%(suffix_Asso))
path_Spatial = os.path.join(path_base_input, folder_Spatial, '%s.ptseries.nii'%(suffix_Spatial))
path_Math = os.path.join(path_base_input, folder_Math, '%s.ptseries.nii'%(suffix_Math))

# %%
nii_FM = nib.load(path_FM)
nii_Asso = nib.load(path_Asso)
nii_Spatial = nib.load(path_Spatial)
nii_Math = nib.load(path_Math)

data_FM = nii_FM.get_fdata()[0]
data_Asso = nii_Asso.get_fdata()[0]
data_Spatial = nii_Spatial.get_fdata()[0]
data_Math = nii_Math.get_fdata()[0]


# %%
data_FM_modif = np.where(np.isnan(data_FM), np.nan, np.where(data_FM > 0, 1, -1))
data_Asso_modif = np.where(np.isnan(data_Asso), np.nan, np.where(data_Asso > 0, 2, -2))

data_FM_Asso_modified = np.nan_to_num(data_FM_modif, nan=0) + np.nan_to_num(data_Asso_modif, nan=0)
data_FM_Asso_modified = np.where(data_FM_Asso_modified == 0, np.nan, data_FM_Asso_modified)


fig_FM_Asso_modif_file = os.path.join(path_base_output, '%s_%s.png'%(fig_num, suffix_conjunction))
plot_Kong_parcellation(data_FM_Asso_modified, fig_FM_Asso_modif_file, cmap='FM_Asso_conjunction', title='Overlap of difficulty effect')


