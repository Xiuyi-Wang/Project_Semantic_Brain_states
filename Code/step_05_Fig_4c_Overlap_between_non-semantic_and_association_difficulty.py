# %%
import numpy as np
import os
import nibabel as nib
import sys
current_directory = os.getcwd()
functions_directory = os.path.join(current_directory, '..', 'Functions')
sys.path.append(functions_directory)
from function_plot_Kong_ptseries_dlabel_specific_parcels import plot_Kong_parcellation

# %%
path_base_input = os.path.join(current_directory,'../Data/brain_map')
path_output = os.path.join(current_directory,'../Results/Figure_4')
os.makedirs(path_output, exist_ok=True)

fig_num = 'Fig_4c'

folder_Spatial = 'stat_Spatial_FDR_p_0.05'
folder_Math = 'stat_Maths_FDR_p_0.05'
folder_FM = 'stat_FeatureMatching_FDR_p_0.05'
folder_Asso = 'stat_Association_FDR_p_0.05'

suffix_FM = 'Non-Matching_vs_Matching-pm_t_map_FDR_0.05'
suffix_Asso = 'Associated_no_vs_yes_pm_t_map_FDR_0.05'
suffix_Spatial = 'task-Spatial_contrast-Hard_vs_Easy_t_map_FDR_0.05'
suffix_Math = 'task-Maths_contrast-Hard_vs_Easy_t_map_FDR_0.05'

# %%
path_FM = os.path.join(path_base_input, folder_FM, '%s.ptseries.nii'%(suffix_FM))
path_Asso = os.path.join(path_base_input, folder_Asso, '%s.ptseries.nii'%(suffix_Asso))
path_Spatial = os.path.join(path_base_input, folder_Spatial, '%s.ptseries.nii'%(suffix_Spatial))
path_Math = os.path.join(path_base_input, folder_Math, '%s.ptseries.nii'%(suffix_Math))

# %%
nii_FM = nib.load(path_FM)
nii_Asso = nib.load(path_Asso)
nii_Spatial = nib.load(path_Spatial)
nii_Math = nib.load(path_Math)

# %%
data_FM = nii_FM.get_fdata()[0]
data_Asso = nii_Asso.get_fdata()[0]
data_Spatial = nii_Spatial.get_fdata()[0]
data_Math = nii_Math.get_fdata()[0]

# %%
#get positive
data_FM[data_FM<0] = np.nan
data_Asso[data_Asso<0] = np.nan
data_Spatial[data_Spatial<0] = np.nan
data_Math[data_Math<0] = np.nan 

# %%
# Initialize an array with zeros
Asso_overlap_spatial_or_math = np.full((400,), np.nan)

# Mark positions where data_Asso is not nan and both data_Spatial and data_Math are nan as 1
Asso_overlap_spatial_or_math[np.isfinite(data_Asso) & np.isnan(data_Spatial) & np.isnan(data_Math)] = 1

Asso_overlap_spatial_or_math[np.isfinite(data_Asso) & np.isfinite(data_FM)] = 3

# Mark positions where data_Asso is not nan and at least one of data_Spatial or data_Math is not nan as 2
Asso_overlap_spatial_or_math[np.isfinite(data_Asso) & (np.isfinite(data_Spatial) | np.isfinite(data_Math))] = 2

# %%
fig_Asso_overlap_Spatial_or_math = os.path.join(path_output, '%s_Overlap_between_non-semantic_and_association_difficulty.png'%(fig_num))
plot_Kong_parcellation(Asso_overlap_spatial_or_math, fig_Asso_overlap_Spatial_or_math, cmap = 'Asso',title = 'Overlap between non-semantic and Asso difficulty')


