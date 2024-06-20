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
path_output = os.path.join(current_directory,'../Results/Figure_6')
os.makedirs(path_output, exist_ok=True)

fig_num = 'Fig_6b'

# %%
folder_math = 'stat_Maths_thre_p_no_thre'
suffix_math = 'task-Maths_contrast-Hard_vs_Easy_t_map'

path_math = os.path.join(path_base_input, folder_math, '%s.ptseries.nii'%(suffix_math))

# %%
nii_math = nib.load(path_math)
data_math = nii_math.get_fdata()[0]
fig_math_file = os.path.join(path_output, '%s_%s.png'%(fig_num, suffix_math))
plot_Kong_parcellation(data_math, fig_math_file, cmap = 'nipy_spectral', title = 'Difficulty effect of math')


