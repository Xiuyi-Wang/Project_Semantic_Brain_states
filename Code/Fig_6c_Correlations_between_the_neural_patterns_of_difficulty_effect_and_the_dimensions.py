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

path_FM_Asso_grad_input = os.path.join(current_directory, '../Data/brain_map/correlation_with_gradient/indi_level/sem_HCP_rest_FC_gradient_indi.xlsx')
path_Spatial_Math_grad_input = os.path.join(current_directory, '../Data/brain_map/correlation_with_gradient/indi_level/nonsem_HCP_rest_FC_gradient_indi.xlsx')
path_output = os.path.join(current_directory,'../Results/Figure_6')
os.makedirs(path_output, exist_ok=True)

font_file = os.path.join(current_directory, '../Templates/Atlas/Font/arial.ttf')

fig_num = 'Fig_6c'

# %%
data_sem = pd.read_excel(path_FM_Asso_grad_input)
data_nosem = pd.read_excel(path_Spatial_Math_grad_input)

contrasts_all_sem = ['Non-Matching_vs_Matching-pm', 'Associated_no_vs_yes_pm']
contrasts_all_nosem = ['Spatial_Hard_vs_Easy', 'Maths_Hard_vs_Easy']

data_selected_sem = data_sem[data_sem['Group'].isin(contrasts_all_sem)]
data_selected_sem.loc[data_selected_sem['Hue'] == 3, 'r_value'] *= -1
data_selected_sem.loc[data_selected_sem['Hue'] == 3, 'z_value'] *= -1
data_selected_nosem = data_nosem[data_nosem['Group'].isin(contrasts_all_nosem)]

# %%
# Organize semantic data
mean_values_sem = data_selected_sem.groupby(['Group', 'Hue'])['r_value'].mean().reset_index()
mean_values_sem = mean_values_sem.rename(columns={'r_value': 'r_value_mean'})

new_df_sem = pd.DataFrame(mean_values_sem)
merged_df_sem = pd.merge(data_selected_sem, new_df_sem, on=['Group', 'Hue'])
merged_df_sem['Hue'] = merged_df_sem['Hue'].replace({1: 'Component_1',  2: 'Component_2',3: 'Component_3'})

# %%
# Organize non-semantic data
mean_values_nosem = data_selected_nosem.groupby(['Group', 'Hue'])['r_value'].mean().reset_index()
mean_values_nosem = mean_values_nosem.rename(columns={'r_value': 'r_value_mean'})

new_df_nosem = pd.DataFrame(mean_values_nosem)
merged_df_nosem = pd.merge(data_selected_nosem, new_df_nosem, on=['Group', 'Hue'])
merged_df_nosem['Hue'] = merged_df_nosem['Hue'].replace({1: 'Component_1',  2: 'Component_2',3: 'Component_3'})

# %%
# Concatenate semantic data and non-semantic data into one dataframe
df_concatenated = pd.concat([merged_df_sem, merged_df_nosem], axis=0)

contrasts_all_sem = ['Non-Matching_vs_Matching-pm', 'Associated_no_vs_yes_pm']
contrasts_all_nosem = ['Spatial_Hard_vs_Easy', 'Maths_Hard_vs_Easy']

# %%
# Define palette color
palette = {'Non-Matching_vs_Matching-pm' : '#e8e830',
           'Associated_no_vs_yes_pm' : '#FF4500',
           'Spatial_Hard_vs_Easy':'#0d3359', 
           'Maths_Hard_vs_Easy':'#1a8087',
           'Component_1' : '#E1C855',
           'Component_2' : '#E07B54',
           'Component_3' : '#51B1B7'}

#Define group plot order
order = ['Associated_no_vs_yes_pm','Spatial_Hard_vs_Easy', 'Maths_Hard_vs_Easy', 'Non-Matching_vs_Matching-pm']

# %%
fig, ax = plt.subplots(figsize=(10,5))
fontprob_label_y=fm.FontProperties(fname=font_file, size=12)
fontprob_label=fm.FontProperties(fname=font_file, size=12)
sns.set(font_scale=0.7)
sns.set_style("white")

sns.violinplot(ax = ax, x='Hue', y='r_value', data=df_concatenated, hue='Group', palette=palette, scale_hue=False,saturation=1, bw=.3, scale = 'width', width = 0.8, hue_order=order, linewidth=.75)

sns.stripplot(ax = ax,x='Hue', y='r_value', data=df_concatenated, hue='Group',  dodge=0.35, linewidth=.5, size=3, palette=palette,  hue_order=order, edgecolor = '#696969')

# # # Plot the mean values on top of the violins
sns.stripplot(x='Hue', y='r_value_mean', hue='Group', data=df_concatenated, dodge=True, jitter=False, linewidth=1,  size=5, palette=palette,  hue_order=order, edgecolor = '#696969')
y_min, y_max=ax.get_ylim()

ax.set_ylabel('r value',  fontproperties=fontprob_label_y)
ax.set_xlabel('',  fontproperties=fontprob_label_y)

#labeling
ax.set_xticklabels([ 'Component_1', 'Component_2', 'Component_3'],fontproperties=fontprob_label)
plt.setp(ax.get_yticklabels(), fontproperties=fontprob_label, )

ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)

# # Set the axis scale faces inward
plt.rcParams['xtick.direction']='in'
plt.rcParams['ytick.direction']='in'

# Set the y-axis limits
plt.legend('').set_visible(False)
plt.ylim(-0.75, 0.9)

figfile = os.path.join(path_output, '%s_Correlations_between_the_neural_patterns_of_difficulty_effect_and_the_dimensions.png'%(fig_num))
plt.savefig(figfile, dpi=300)
plt.close()

# %%
# Statistic analysis between senmantic and non-semantic in each demension group 
columns = ['hue','group_nosem', 'group_sem', 'mean_z_nosem', 'mean_z_sem','t_value', 'p_value','significant','mean_r_nosem', 'mean_r_sem']
df_statistic = pd.DataFrame(columns=columns)

#statistic ttest-ind
hue_all = ['Component_1','Component_2','Component_3']

contrasts_all_sem = ['Non-Matching_vs_Matching-pm', 'Associated_no_vs_yes_pm']
contrasts_all_nosem = ['Spatial_Hard_vs_Easy', 'Maths_Hard_vs_Easy']

for hue in hue_all:
    data_hue = df_concatenated[df_concatenated['Hue'].isin([hue])]
    print('---------------')
    print(hue)
    
    for nosem in contrasts_all_nosem:
        for sem in contrasts_all_sem:
            print(nosem)
            print(sem)
            
            data_interest_pair = data_hue[data_hue['Group'].isin([sem,nosem])]
            
            # Assuming dataFrame
            nosem_data = data_interest_pair[data_interest_pair['Group'] == nosem]
            sem_data = data_interest_pair[data_interest_pair['Group'] == sem]
            
            # Calculate r values
            r_nosem_mean = nosem_data['r_value'].mean()
            r_sem_mean = sem_data['r_value'].mean()

            # Calculate z values
            z_nosem_mean = nosem_data['z_value'].mean()
            z_sem_mean = sem_data['z_value'].mean()
                      

            selected_z_values_nosem = data_interest_pair.loc[data_interest_pair['Group'] == nosem, 'z_value'].values
            selected_z_values_sem = data_interest_pair.loc[data_interest_pair['Group'] == sem, 'z_value'].values

            z_nosem_array = np.array(selected_z_values_nosem)
            z_sem_array = np.array(selected_z_values_sem)
            
            # Conducting the independent t-test
            t_statistic, p_value = stats.ttest_ind(z_nosem_array, z_sem_array)

            print("T-statistic:", t_statistic)
            print("P-value:", p_value)
            
            # Interpretation of the result
            alpha = 0.05  # common threshold for statistical significance
            if p_value < alpha:
                significant = 'yes'
                print("The difference between groups is statistically significant.")
            else:
                significant = 'no'
                print("There is no statistically significant difference between the groups.")
            
            # Append a new row to the DataFrame
            df_statistic = df_statistic.append({
                'hue': hue,
                'group_nosem': nosem,
                'group_sem': sem,
                'mean_z_nosem': z_nosem_mean,
                'mean_z_sem': z_sem_mean,
                't_value': t_statistic,
                'p_value': p_value,
                'significant': significant,
                'mean_r_nosem':r_nosem_mean, 
                'mean_r_sem':r_sem_mean
            }, ignore_index=True)

# %%
file_output = os.path.join(path_output, 'Fig_6c_nosem_vs_sem_statistic_results.xlsx')
df_statistic.to_excel(file_output, index=False)

# %%



