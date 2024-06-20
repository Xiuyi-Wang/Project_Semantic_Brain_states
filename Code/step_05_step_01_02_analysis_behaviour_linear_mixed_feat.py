# analyse the relationship between RT, Acc and other variables
# global similarity, feature similarity
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from function_check_accuracy_feat import check_behaviour_feat

path_base = '/home/xiuyi/Data/Task_Gradient/data_behaviour/data_analysis/Feature_Matching'
col_subj = 'subj_ID'
col_feature_order = 'feature_order'
col_deci = 'feature similarity decision'
col_run_ID = 'scan_ID'
col_key_press_order ='key_press_order'
col_corr_1 = 'correct_1'
col_corr_2 = 'correct_2'
col_key_1 = 'key_press_1'
col_key_2 = 'key_press_2'
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

thre = 56
# step 1: list all the files
files = os.listdir(path_base)
files.sort()
dfs = []
for file in files:
    subj_ID, R_ID, stim_ID, feature_order, decision_ID, date,time,run,run_ID = file.split('_')

    data = pd.read_excel(os.path.join(path_base,file))

    trials_num_corr = check_behaviour_feat(data)

    # check the behaviour performance
    # if the accuracy is lower than the threshold, delete this run
    # if not, read the data
    if trials_num_corr < thre:

        print (subj_ID, run_ID[0],' is deleted')

    else:
        # choose the minimum response time and score among the two RTs and two scores.
        data[col_RT] = data[[col_RT_1, col_RT_2]].min(axis=1)
        data[col_corr_score] = data[[col_corr_1, col_corr_2]].min(axis=1)

        # write trial type correct_decision_feature
        data[col_corr] = np.where(data[col_corr_score] == 1, 'correct', 'wrong')
        data[col_trial_type] = data[col_corr] + '_' + data[col_deci]

        # add the necessary columns_trial
        data[col_subj] = subj_ID
        data[col_feature_order] = int(feature_order)
        data[col_key_press_order] = int(decision_ID)
        data[col_run_ID] = int(run_ID[0])

        dfs.append(data)

# merge all the dfs
df_merged = pd.concat(dfs)

# add the group col for mixed linear effect model
df_merged[col_group]=1

df_merged.rename(columns={col_global_simi_old:col_global_simi_new,col_feature_simi_old:col_feature_simi_new, col_association_old: col_association_new,col_word2vec_old:col_word2vec_new},inplace=True)
df_merged[col_subj] = pd.to_numeric(df_merged[col_subj])

# step 2: examine the relationship between RT and other variables
# only choose the correct trials
df_match_corr = df_merged[df_merged[col_trial_type] == 'correct_yes']
df_nonmatch_corr = df_merged[df_merged[col_trial_type] == 'correct_no']

df_match_corr_part    = df_match_corr[['response_time', 'key_press_order', 'subj_ID', 'scan_ID', 'group', 'global_simi','feature_simi','association','word2vec']]
df_nonmatch_corr_part = df_nonmatch_corr[['response_time', 'key_press_order', 'subj_ID', 'scan_ID', 'group', 'global_simi','feature_simi','association','word2vec']]

# save it and then read it to avoid errors
df_match_corr_part.to_excel('/home/xiuyi/Data/Task_Gradient/data_behaviour/data_analysis/feat_match_corr.xlsx', index=False)

df_match_corr_part = pd.read_excel('/home/xiuyi/Data/Task_Gradient/data_behaviour/data_analysis/feat_match_corr.xlsx')

df_nonmatch_corr_part.to_excel('/home/xiuyi/Data/Task_Gradient/data_behaviour/data_analysis/feat_nonmatch_corr.xlsx',index=False)

df_nonmatch_corr_part = pd.read_excel('/home/xiuyi/Data/Task_Gradient/data_behaviour/data_analysis/feat_nonmatch_corr.xlsx')

# mixed linear effect
modal_match_global = sm.MixedLM.from_formula('response_time ~ global_simi',
                                        vc_formula = {"subj_ID":"0 + subj_ID", "scan_ID":"0 + scan_ID","key_press_order":"0 + key_press_order"},
                                        groups = df_match_corr_part['group'], data =df_match_corr_part).fit().summary()

print ('matching trials global similarity')
print (modal_match_global)

print ('********************')
print ('********************')

modal_nonmatch_global = sm.MixedLM.from_formula('response_time ~ global_simi',
                                        vc_formula = {"subj_ID":"0 + subj_ID", "scan_ID":"0 + scan_ID","key_press_order":"0 + key_press_order"},
                                        groups = df_nonmatch_corr_part['group'], data =df_nonmatch_corr_part).fit().summary()
print ('non-matching trials global similarity')
print (modal_nonmatch_global)
print ('********************')
print ('********************')


#%% check feature similarity
# mixed linear effect
modal_match_feature = sm.MixedLM.from_formula('response_time ~ feature_simi',
                                        vc_formula = {"subj_ID":"0 + subj_ID", "scan_ID":"0 + scan_ID","key_press_order":"0 + key_press_order"},
                                        groups = df_match_corr_part['group'], data =df_match_corr_part).fit().summary()
print ('matching trials feature similarity')
print (modal_match_feature)

print ('********************')
print ('********************')

modal_nonmatch_feature = sm.MixedLM.from_formula('response_time ~ feature_simi',
                                        vc_formula = {"subj_ID":"0 + subj_ID", "scan_ID":"0 + scan_ID","key_press_order":"0 + key_press_order"},
                                        groups = df_nonmatch_corr_part['group'], data =df_nonmatch_corr_part).fit().summary()

print ('non-matching trials feature similarity')
print (modal_nonmatch_feature)


#%% check association
# mixed linear effect
modal_match_association = sm.MixedLM.from_formula('response_time ~ association',
                                        vc_formula = {"subj_ID":"0 + subj_ID", "scan_ID":"0 + scan_ID","key_press_order":"0 + key_press_order"},
                                        groups = df_match_corr_part['group'], data =df_match_corr_part).fit().summary()
print ('matching trials association')
print (modal_match_association)

print ('********************')
print ('********************')

modal_nonmatch_association = sm.MixedLM.from_formula('response_time ~ association',
                                        vc_formula = {"subj_ID":"0 + subj_ID", "scan_ID":"0 + scan_ID","key_press_order":"0 + key_press_order"},
                                        groups = df_nonmatch_corr_part['group'], data =df_nonmatch_corr_part).fit().summary()

print ('non-matching trials association')
print (modal_nonmatch_association)


#%% check association
# mixed linear effect
modal_match_word2vec = sm.MixedLM.from_formula('response_time ~ word2vec',
                                        vc_formula = {"subj_ID":"0 + subj_ID", "scan_ID":"0 + scan_ID","key_press_order":"0 + key_press_order"},
                                        groups = df_match_corr_part['group'], data =df_match_corr_part).fit().summary()
print ('matching trials word2vec')
print (modal_match_word2vec)

print ('********************')
print ('********************')

modal_nonmatch_word2vec = sm.MixedLM.from_formula('response_time ~ word2vec',
                                        vc_formula = {"subj_ID":"0 + subj_ID", "scan_ID":"0 + scan_ID","key_press_order":"0 + key_press_order"},
                                        groups = df_nonmatch_corr_part['group'], data =df_nonmatch_corr_part).fit().summary()

print ('non-matching trials word2vec')
print (modal_nonmatch_word2vec)



