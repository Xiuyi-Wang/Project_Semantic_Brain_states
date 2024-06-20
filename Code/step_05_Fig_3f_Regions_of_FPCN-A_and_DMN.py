# %%
import os
import nibabel as nib
import sys
current_directory = os.getcwd()
functions_directory = os.path.join(current_directory, '..', 'Functions')
sys.path.append(functions_directory)
from function_plot_Kong_ptseries_dlabel_use_fig_color import plot_Kong_parcellation_pos_neg

# %%
path_base_input = os.path.join(current_directory,'../Data/brain_map')
path_output = os.path.join(current_directory,'../Results/Figure_4')
os.makedirs(path_output, exist_ok=True)

fig_num = 'Fig_4a'

folder_Spatial = 'stat_Spatial_FDR_p_0.05'
suffix_Spatial = 'task-Spatial_contrast-Hard_vs_Easy_t_map_FDR_0.05'


path_Spatial = os.path.join(path_base_input, folder_Spatial, '%s.ptseries.nii'%(suffix_Spatial))
nii_Spatial = nib.load(path_Spatial)
data_Spatial = nii_Spatial.get_fdata()[0]
fig_spatial_file = os.path.join(path_output, '%s_%s.png'%(fig_num, suffix_Spatial))

#change scale and plot
plot_Kong_parcellation_pos_neg(data_Spatial, fig_spatial_file, color = 'psych', title='Difficulty effect of spatial working memory', scale=[-10.6,13.22])

# %%



