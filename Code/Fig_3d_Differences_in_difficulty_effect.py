# %%
import numpy as np
import os
import pandas as pd
import sys
current_directory = os.getcwd()
functions_directory = os.path.join(current_directory, '..', 'Functions')
sys.path.append(functions_directory)
from function_plot_Kong_ptseries_dlabel_specific_parcels import plot_Kong_parcellation

# %%
path_template_file = os.path.join(current_directory, '../Templates/Atlas/Parcellation_Kong_2022/Schaefer_417.xlsx')
path_output = os.path.join(current_directory,'../Results/Figure_3')
os.makedirs(path_output, exist_ok=True)

fig_num = 'Fig_3f'

df_scheafer = pd.read_excel(path_template_file)
df_parcel_network = df_scheafer[['parcel_ID', 'network']]

# %%
dmn = ['DefaultA', 'DefaultB', 'DefaultC']
conta = ['ContA']
array_dmn = np.zeros((400,))
array_conta = np.zeros((400,)) 

dmn_id = df_parcel_network[df_parcel_network['network'].isin(dmn)]
conta_id = df_parcel_network[df_parcel_network['network'].isin(conta)]

dmn_list = dmn_id['parcel_ID'].tolist()
conta_list = conta_id['parcel_ID'].tolist()

# Fill in 1 at specified row and column indices
for index_conta_row in conta_list:
# for index_dan_column in conta_list:
    array_conta[index_conta_row-1] = -1

for index_dmn_row in dmn_list:
# for index_dmn_column in dmn_list:
    array_dmn[index_dmn_row-1] = 1
    
array_conta = array_conta.astype(int).flatten()
array_dmn = array_dmn.astype(int).flatten()

array_conta_dmn = array_conta+array_dmn
array_modif = np.where(array_conta_dmn == 0, np.nan, array_conta_dmn)

# %%
fig_FPCNA_DMN = os.path.join('%s_FPCN-A_and_DMN.png'%(fig_num))
path_conta_dmn = os.path.join(path_output, fig_FPCNA_DMN)
plot_Kong_parcellation(array_modif, path_conta_dmn, cmap= 'FPCNA_DMN',  title='Regions of FPCN-A and DMN')

# %%



