# analyse the relationship between RT, other variables
# global similarity, feature similarity
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from function_check_behaviour import check_behaviour_asso
import researchpy as rp


path_base =  '/home/xiuyi/Data/Task_Gradient/data_behaviour/data_analysis/clean/task-Association'

path_output_base = '/home/xiuyi/Data/Task_Gradient/data_behaviour/data_analysis/clean/mixed_linear_effect/asso'

os.makedirs(path_output_base,exist_ok=True)

col_subj = 'subj_ID'
col_feature_order = 'feature_order'
col_deci = 'feature similarity decision'
col_run_ID = 'scan_ID'
col_key_press_order ='key_press_order'
col_corr_1 = 'correct_1'
col_corr_2 = 'correct_2'
col_key_1 = 'key_press_1'
col_key_2 = 'key_press_2'
col_key   = 'key_press'
col_probe = 'probe'

col_RT_1 = 'RT_1'
col_RT_2 = 'RT_2'
col_RT = 'response_time'
col_corr_score = 'correct_score'
col_corr = 'correct'
col_trial_type = 'trial_type'
col_group = 'group'

col_global_simi_old = '10_global_simi'
col_global_simi_new = 'global_simi'

col_feature_simi_old = '11_feature_simi'
col_feature_simi_new = 'feature_simi'

col_association_old = '13_association'
col_association_new = 'association'

col_word2vec_old = '12_word2vec'
col_word2vec_new = 'word2vec'

trial_types_1 =['Associated','Non-Associated','Wrong']
trial_types_2 =['Non-Associated','Associated','Wrong']

thre = 24
# step 1: list all the files
files = os.listdir(path_base)
files.sort()
dfs = []

files_update = []
for file in files:
    if not file.startswith('.~'):
        files_update.append(file)
        
for file in files_update:


    subj_ID, R_ID, stim_ID,  decision_ID, date,time,run,run_ID = file.split('_')

    data = pd.read_excel(os.path.join(path_base,file))

    num_no_button_total = check_behaviour_asso(data)

    # check the behaviour performance
    # if the accuracy is lower than the threshold, delete this run
    # if not, read the data
    if num_no_button_total > thre:

        print (subj_ID, run_ID[0],' is deleted')

    else:

        if decision_ID == '1':
            trial_types = trial_types_1

        elif decision_ID == '2':
            trial_types = trial_types_2

        # choose the minimum response time and score among the two RTs and two scores.
        data[col_RT] = data[[col_RT_1, col_RT_2]].min(axis=1)

        data[col_key] = data[[col_key_1, col_key_2]].min(axis=1)

        data[col_trial_type]= data[col_key]
        data = data.replace({col_trial_type: {1: trial_types[0], 2: trial_types[1],999:trial_types[2]}})

        # add the necessary columns_trial
        data[col_subj] = subj_ID
        data[col_key_press_order] = int(decision_ID)
        data[col_run_ID] = int(run_ID[0])

        dfs.append(data)

# merge all the dfs
df_merged = pd.concat(dfs)

# skip the no button press trials
df_merged = df_merged[df_merged[col_key]!='Wrong']


# add the group col for mixed linear effect model
df_merged[col_group]=1

df_merged.rename(columns={col_global_simi_old:col_global_simi_new,col_feature_simi_old:col_feature_simi_new, col_association_old: col_association_new,col_word2vec_old:col_word2vec_new},inplace=True)
df_merged[col_subj] = pd.to_numeric(df_merged[col_subj])

# step 2: examine the relationship between RT and other variables
# only choose the correct trials
df_asso = df_merged[df_merged[col_trial_type] == 'Associated']
df_nonasso = df_merged[df_merged[col_trial_type] == 'Non-Associated']

cols_part = [col_trial_type,'response_time', 'key_press_order', 'subj_ID', 'scan_ID', 'group', 'global_simi','feature_simi','association','word2vec',col_probe]
df_asso_part    = df_asso[cols_part]
df_nonasso_part = df_nonasso[cols_part]

print (rp.summary_cont(df_merged.groupby(col_trial_type)[col_RT]))

# plot the bar
boxplot = df_merged.boxplot([col_RT], by = [col_key],
                     figsize = (16, 9),
                     showmeans = True,
                     notch = True)

boxplot.set_xlabel("Categories")
boxplot.set_ylabel("response time")

#
df_merged_part    = df_merged[cols_part]


# modal_deci = sm.MixedLM.from_formula('response_time ~ C(trial_type)',
#                                      vc_formula = {"subj_ID":"0 + subj_ID", "scan_ID":"0 + scan_ID","key_press_order":"0 + key_press_order","probe":"0+probe"},
#                                      groups = df_merged_part['group'], data =df_merged_part).fit().summary()
#
# print ('decision type RT')
# print (modal_deci)





# save it and then read it to avoid errors
df_asso_part.to_excel(os.path.join(path_output_base,'asso_asso.xlsx'), index=False)

df_asso_part = pd.read_excel(os.path.join(path_output_base,'asso_asso.xlsx'))

df_nonasso_part.to_excel(os.path.join(path_output_base,'asso_nonasso.xlsx'), index=False)

df_nonasso_part = pd.read_excel(os.path.join(path_output_base,'asso_nonasso.xlsx'))

# mixed linear effect
modal_asso_global = sm.MixedLM.from_formula('response_time ~ global_simi',
                                             vc_formula = {"subj_ID":"0 + subj_ID", "scan_ID":"0 + scan_ID","key_press_order":"0 + key_press_order","probe":"0+probe"},
                                             groups = df_asso_part['group'], data =df_asso_part).fit().summary()

print ('association trials global similarity')
print (modal_asso_global)

print ('********************')
print ('********************')

modal_nonasso_global = sm.MixedLM.from_formula('response_time ~ global_simi',
                                                vc_formula = {"subj_ID":"0 + subj_ID", "scan_ID":"0 + scan_ID","key_press_order":"0 + key_press_order","probe":"0+probe"},
                                                groups = df_nonasso_part['group'], data =df_nonasso_part).fit().summary()
print ('nonassociation trials global similarity')
print (modal_nonasso_global)
print ('********************')
print ('********************')


#%% check feature similarity
# mixed linear effect
modal_asso_feature = sm.MixedLM.from_formula('response_time ~ feature_simi',
                                             vc_formula = {"subj_ID":"0 + subj_ID", "scan_ID":"0 + scan_ID","key_press_order":"0 + key_press_order","probe":"0+probe"},
                                             groups = df_asso_part['group'], data =df_asso_part).fit().summary()
print ('association trials feature similarity')
print (modal_asso_feature)

print ('********************')
print ('********************')

modal_nonasso_feature = sm.MixedLM.from_formula('response_time ~ feature_simi',
                                                vc_formula = {"subj_ID":"0 + subj_ID", "scan_ID":"0 + scan_ID","key_press_order":"0 + key_press_order","probe":"0+probe"},
                                                groups = df_nonasso_part['group'], data =df_nonasso_part).fit().summary()

print ('non-asso trials feature similarity')
print (modal_nonasso_feature)


#%% check association
# mixed linear effect
modal_asso_association = sm.MixedLM.from_formula('response_time ~ association',
                                                 vc_formula = {"subj_ID":"0 + subj_ID", "scan_ID":"0 + scan_ID","key_press_order":"0 + key_press_order","probe":"0+probe"},
                                                 groups = df_asso_part['group'], data =df_asso_part).fit().summary()
print ('asso trials association')
print (modal_asso_association)

print ('********************')
print ('********************')

modal_nonasso_association = sm.MixedLM.from_formula('response_time ~ association',
                                                    vc_formula = {"subj_ID":"0 + subj_ID", "scan_ID":"0 + scan_ID","key_press_order":"0 + key_press_order","probe":"0+probe"},
                                                    groups = df_nonasso_part['group'], data =df_nonasso_part).fit().summary()

print ('non-asso trials association')
print (modal_nonasso_association)


#%% check word2vec
# mixed linear effect
modal_asso_word2vec = sm.MixedLM.from_formula('response_time ~ word2vec',
                                              vc_formula = {"subj_ID":"0 + subj_ID", "scan_ID":"0 + scan_ID","key_press_order":"0 + key_press_order","probe":"0+probe"},
                                              groups = df_asso_part['group'], data =df_asso_part).fit().summary()
print ('asso trials word2vec')
print (modal_asso_word2vec)

print ('********************')
print ('********************')

modal_nonasso_word2vec = sm.MixedLM.from_formula('response_time ~ word2vec',
                                                 vc_formula = {"subj_ID":"0 + subj_ID", "scan_ID":"0 + scan_ID","key_press_order":"0 + key_press_order","probe":"0+probe"},
                                                 groups = df_nonasso_part['group'], data =df_nonasso_part).fit().summary()

print ('non-asso trials word2vec')
print (modal_nonasso_word2vec)

#%% check all

modal_asso_full = sm.MixedLM.from_formula('response_time ~ global_simi + feature_simi  + association + word2vec',
                                              vc_formula = {"subj_ID":"0 + subj_ID", "scan_ID":"0 + scan_ID","key_press_order":"0 + key_press_order","probe":"0+probe"},
                                              groups = df_asso_part['group'], data =df_asso_part).fit().summary()
print ('asso trials full')
print (modal_asso_full)

print ('********************')
print ('********************')

modal_nonasso_full = sm.MixedLM.from_formula('response_time ~ global_simi + feature_simi  + association + word2vec',
                                                 vc_formula = {"subj_ID":"0 + subj_ID", "scan_ID":"0 + scan_ID","key_press_order":"0 + key_press_order","probe":"0+probe"},
                                                 groups = df_nonasso_part['group'], data =df_nonasso_part).fit().summary()

print ('non-asso trials full')
print (modal_nonasso_full)

