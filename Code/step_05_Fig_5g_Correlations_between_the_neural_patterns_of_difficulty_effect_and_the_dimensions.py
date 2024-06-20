# %%
import os
import nibabel as nib
from scipy import stats
from scipy.io import loadmat
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import font_manager as fm

# %%
current_directory = os.getcwd()

path_base_input = os.path.join(current_directory, '../Data/brain_map/correlation_with_gradient/indi_level/sem_HCP_rest_FC_gradient_indi.xlsx')
path_output = os.path.join(current_directory,'../Results/Figure_5')
os.makedirs(path_output, exist_ok=True)

font_file = os.path.join(current_directory, '../Templates/Atlas/Font/arial.ttf')

fig_num = 'Fig_5g'

# %%
data = pd.read_excel(path_base_input)

contrasts_all = ['Non-Matching_vs_Matching-pm', 'Associated_no_vs_yes_pm']

data_selected = data[data['Group'].isin(contrasts_all)]
data_selected.loc[data_selected['Hue'] == 3, 'r_value'] *= -1

# %%
# Calculate the mean of r_values by hue under each group
mean_values = data_selected.groupby(['Group', 'Hue'])['r_value'].mean().reset_index()

# Rename the 'r_values' column to 'r_value_mean'
mean_values = mean_values.rename(columns={'r_value': 'r_value_mean'})

new_df = pd.DataFrame(mean_values)
merged_df = pd.merge(data_selected, new_df, on=['Group', 'Hue'])
merged_df['Hue'] = merged_df['Hue'].replace({1: 'Component_1', 2: 'Component_2', 3: 'Component_3'})

palette = {'Component_1' : '#01cbcb',
           'Component_2' : '#00356d',
           'Component_3' : '#fe9900'}

palette_hue = {'Non-Matching_vs_Matching-pm':'#e8e830', 
               'Associated_no_vs_yes_pm':'#FF4500'}

order = ['Associated_no_vs_yes_pm', 'Non-Matching_vs_Matching-pm']

# %%
# Plot the violin
fig, ax = plt.subplots(figsize=(12,5))
fontprob_label_y=fm.FontProperties(fname=font_file, size=16)
fontprob_label=fm.FontProperties(fname=font_file, size=15)
sns.set(font_scale=0.7)
sns.set_style("white")

ax =sns.violinplot(ax = ax, x='Hue', y='r_value', data=merged_df, hue='Group', palette=palette_hue, scale_hue=False,cut=1.8,saturation=20, bw=.31, scale = 'width', width = 0.8,linewidth=0.7, hue_order=order)
sns.stripplot(ax = ax,x='Hue', y='r_value', data=merged_df, hue='Group',  jitter=0.08,dodge=True, linewidth=1, size=5, palette=palette_hue, hue_order=order)

# # # Plot the mean values on top of the violins
sns.stripplot(x='Hue', y='r_value_mean', hue='Group', data=merged_df, dodge=True, jitter=False, linewidth=2, size=13, palette=palette_hue, hue_order=order)
y_min, y_max=ax.get_ylim()
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
# Set the y-axis limits
plt.legend('')
plt.ylim(-0.8, 1)

ax.set_ylabel('r value',  fontproperties=fontprob_label_y)
ax.set_xlabel('',  fontproperties=fontprob_label_y)

plt.setp(ax.get_yticklabels(), fontproperties=fontprob_label)

ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)

# Set the axis scale faces inward
plt.rcParams['xtick.direction']='in'
plt.rcParams['ytick.direction']='in'

plt.legend('').set_visible(False)
plt.ylim(-1, 1)
plt.xlim(-0.5,2.45)

plt.title('Correlations between the neural patterns of difficulty effect and the dimensions')

figfile = os.path.join(path_output, '%s_Correlations_between_the_neural_patterns_of_difficulty_effect_and_the_dimensions.png'%(fig_num))
plt.savefig(figfile, dpi=300)

# %%



