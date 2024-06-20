# association - sep 15 correct

import os
from nibabel.gifti import read, write, GiftiDataArray, GiftiImage
from nilearn.glm.first_level import run_glm
from nilearn.glm.first_level import make_first_level_design_matrix, check_design_matrix
from nilearn.glm import compute_contrast
from nilearn.glm.contrasts import compute_fixed_effects
from nilearn.glm.contrasts import _compute_fixed_effects_params
from nilearn.glm.contrasts import _compute_fixed_effect_contrast
import numpy as np
import nibabel as nib
import pandas as pd
import glob
from nibabel import load
import scipy
from nilearn import plotting
from nilearn.datasets import fetch_surf_fsaverage
from scipy.stats import ttest_1samp, norm
from numpy import inf
import warnings
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
warnings.filterwarnings('ignore', '.*do not.*', )
warnings.warn('DelftStack')
warnings.warn('Do not show this message')
from wbplot import pscalar
from function_gene_tables_sig_parcel import gene_table_sig_parcels_Schaefer_417
from function_save_parcels_FDR_Kong_atlas import save_dlabel_Kong
from function_plot_Kong_ptseries_dlabel import plot_Kong_dlabel
from statsmodels.stats.multitest import multipletests

# todo the parametric regressor, need to change
types_interest = ['association',  'feature_simi',  'global_simi',  'word2vec']
type_interest = types_interest[0]
# check whether first level and second level has run
# if yes, = 1, skip 4;  if not, run it.
stats_level_1_2 = 12  # 1, has run
stats_level_3 = 12 # 1 has run

subj_spec = {'sub-09': ['2', '3'], 'sub-10': ['1', '2'], 'sub-18': ['1', '2', '3'],
             'sub-23': ['1', '2', '3'], 'sub-24': ['1', '2'], 'sub-33': ['1', '2', '3']}

# these two participants did not press the button for a few trials
# sub-10: run-1; sub-33: run-2, run-3
subj_spec_no_button_press = ['sub-10','sub-33']

subj_delete = ['sub-20']

subj_IDs = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07',
            'sub-08', 'sub-09', 'sub-10', 'sub-12', 'sub-13', 'sub-14', 'sub-15',
            'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20', 'sub-21', 'sub-22',
            'sub-23', 'sub-24', 'sub-26', 'sub-28', 'sub-29', 'sub-30', 'sub-31',
            'sub-32', 'sub-33', 'sub-34']

folder_sub_func = 'func'

folder_sub_dm = 'design_matrix_fir'

folder_sub_glm_1 = 'level_1'

folder_sub_glm_2 = 'level_2'

parcel_template = 'Schaefer_indi'
path_base_input = '/home/xiuyi/Data/Task_Gradient/Nifti/fMRIPrep_cifti_nifti_xcp_abcd/smooth_6_filter_min_0.01_0.08_36P_cifti/ptseries_individual_parcellation_32k'
path_tsv = '/home/xiuyi/Data/Task_Gradient/data_behaviour/data_analysis/event_tsv/%s'%(type_interest)
path_output_base = '/home/xiuyi/Data/Task_Gradient/results/fmriprep_xcp_abcd_Shaefer_indi/%s' % ( type_interest)

file_template = '/home/xiuyi/Data/Atlas/cifti_template/Schaefer_400_17_group.ptseries.nii'
cifti_template = nib.load(file_template)

# association
ses_ID = 'ses-02'
task = 'Association'

run_IDs_full = ['1','2','3','4']

TR = 1.5

hrf_model = 'glover'

col_onset = 'onset'
col_duration = 'duration'
col_trial_type = 'trial_type'
col_RT = 'response_time'
col_modulation = 'modulation'
col_index = 'TR_ID'

trial_types_full_press = ['Associated','Non-Associated']
trial_types_full_no_press = ['Associated','Non-Associated','Wrong']

contrasts_types = ['Associated', 'Non-Associated', 'Associated_yes_and_no', 'Associated_yes_vs_no', 'Associated_no_vs_yes',
                   'Associated_pm', 'Non-Associated_pm', 'Associated_yes_and_no_pm', 'Associated_yes_vs_no_pm', 'Associated_no_vs_yes_pm',
                   'Associated_neg', 'Non-Associated_neg', 'Associated_yes_and_no_neg',
                   'Associated_pm_neg', 'Non-Associated_pm_neg', 'Associated_yes_and_no_pm_neg']

map_types = ['z', 't_values', 'effects', 'variance']

#%% first and second
# check whether first and second level has run
if stats_level_1_2 != 1:

    # first level and second level analysis
    for subj_ID in subj_IDs:

        if subj_ID in subj_spec_no_button_press:
            trial_types_full = trial_types_full_no_press
        if subj_ID not in subj_spec_no_button_press:
            trial_types_full = trial_types_full_press
        # make sure this is not a subject that you would like to delete
        if subj_ID not in subj_delete:

            # if the subject is special, some runs need to be ignored
            if subj_ID in subj_spec.keys():
                run_IDs = subj_spec[subj_ID]

            else:
                run_IDs = run_IDs_full

            # try:

            # check whether file_hctsa exists, if not, run
            path_check = os.path.join(path_output_base, folder_sub_glm_2, subj_ID, ses_ID, '%s_surf' % map_types[-1])
            file_check = os.path.join(path_check, '%s_%s_task-%s_%s_t_test_%s_cope1.ptseries.nii' % (subj_ID, ses_ID, task, contrasts_types[-1], map_types[-1]))

            if not os.path.exists(file_check):
                labels_all = []
                estimates_all = []
                con_vals = []

                for run_ID in run_IDs:

                    filename_fmri = '%s_%s_task-%s_run-%s_space-fsLR_den-91k_desc-residual_smooth_bold.ptseries.nii' % (subj_ID, ses_ID, task, run_ID)
                    filename_events = '%s_%s_task-%s_run-%s_events.tsv' % (subj_ID, ses_ID, task, run_ID)

                    file_fmri = os.path.join(path_base_input, subj_ID, ses_ID, folder_sub_func, filename_fmri)
                    file_events =  os.path.join(path_tsv,subj_ID, ses_ID, folder_sub_func, filename_events)

                    # read the bold data
                    data_fmri = nib.load(file_fmri).get_fdata()
                    num_scans = data_fmri.shape[0]

                    frame_times = TR * (np.arange(num_scans))

                    # read the events
                    data_events = pd.read_table(file_events)

                    # demean for each trial type
                    trial_types = np.unique(data_events[col_trial_type])

                    data_events_demean =[]

                    for trial_type in trial_types:

                        # slice based on trial type
                        data_events_part = data_events[data_events[col_trial_type]==trial_type]

                        # demean for each condition
                        data_events_part[col_modulation] =data_events_part[col_modulation] - data_events_part[col_modulation].mean()

                        data_events_demean.append(data_events_part)

                    # merge all the conditions
                    df_events_demean = pd.concat(data_events_demean,axis=0)

                    # sort it by onset time
                    df_events_demean.sort_values(by = col_onset,inplace=True)

                    # create design matrix with modulation
                    dm_pm = make_first_level_design_matrix(frame_times, df_events_demean)

                    # remove modulation column
                    data_events = data_events[[col_onset,col_duration,col_trial_type]]

                    # get the current available trial types
                    trial_types_curr = list(np.unique(data_events[col_trial_type].values))

                    # get the parametric regressor
                    regressors_pm = dm_pm[trial_types_curr]

                    # rename the parametric regressor, adding _pm to each trial type
                    cols_pm = [col + '_pm' for col in regressors_pm.columns_trial]
                    regressors_pm = regressors_pm.rename(columns = dict(zip(regressors_pm.columns_trial, cols_pm)))
                    # check whether some trial types are missing
                    # find the diff because the full types and curr trial types
                    diff = list(set(trial_types_full) - set(trial_types_curr))

                    # if yes, add two columns_trial here for each missing trial type
                    # The values are 0.
                    # The column names are missing trial type and missing trail type + '_pm'

                    if len(diff) != 0:
                        for trial_type_miss in diff:
                            regressors_pm[trial_type_miss] = 0
                            regressors_pm[trial_type_miss + '_pm'] = 0

                    # create design matrix again which including the parametric regressor
                    design_matrix = make_first_level_design_matrix(frame_times, data_events, drift_model='polynomial', drift_order=3, add_regs=regressors_pm, add_reg_names=list(regressors_pm.columns_trial.values), hrf_model=hrf_model)

                    # check design matrix
                    _, dmtx, names = check_design_matrix(design_matrix)

                    # save design matrix
                    path_output_dm = os.path.join(path_output_base, folder_sub_glm_1, subj_ID, ses_ID)
                    os.makedirs(path_output_dm,exist_ok = True)

                    filename_dm = '%s_%s_task-%s_run-%s_design_matrix.npz'%(subj_ID, ses_ID, task, run_ID)
                    np.savez(os.path.join(path_output_dm, filename_dm), design_matrix=design_matrix)

                    # build the contrast
                    contrast_matrix = np.eye(design_matrix.shape[1])

                    basic_contrasts = dict([(column, contrast_matrix[i]) for i, column in enumerate(design_matrix.columns_trial)])

                    contrasts = {'Associated': basic_contrasts['Associated'],
                                 'Non-Associated': basic_contrasts['Non-Associated'],
                                 'Associated_yes_and_no': basic_contrasts['Associated']  + basic_contrasts['Non-Associated'],
                                 'Associated_yes_vs_no': basic_contrasts['Associated']  - basic_contrasts['Non-Associated'],
                                 'Associated_no_vs_yes': basic_contrasts['Non-Associated']  - basic_contrasts['Associated'],
                                 'Associated_pm': basic_contrasts['Associated_pm'],
                                 'Non-Associated_pm': basic_contrasts['Non-Associated_pm'],
                                 'Associated_yes_and_no_pm': basic_contrasts['Associated_pm'] + basic_contrasts['Non-Associated_pm'],
                                 'Associated_yes_vs_no_pm':basic_contrasts['Associated_pm'] - basic_contrasts['Non-Associated_pm'],
                                 'Associated_no_vs_yes_pm':basic_contrasts['Non-Associated_pm'] - basic_contrasts['Associated_pm'],
                                 'Associated_neg': -basic_contrasts['Associated'],
                                 'Non-Associated_neg': -basic_contrasts['Non-Associated'],
                                 'Associated_yes_and_no_neg': -basic_contrasts['Associated'] - basic_contrasts['Non-Associated'],
                                 'Associated_pm_neg': -basic_contrasts['Associated_pm'],
                                 'Non-Associated_pm_neg': -basic_contrasts['Non-Associated_pm'],
                                 'Associated_yes_and_no_pm_neg': -basic_contrasts['Associated_pm'] - basic_contrasts['Non-Associated_pm']}

                    labels, estimates = run_glm(data_fmri, design_matrix.values)
                    labels_all.append(labels)
                    estimates_all.append(estimates)

                # convert list to array
                labels_all =  np.asarray(labels_all)
                estimates_all = np.asarray(estimates_all)

                # compute fixed effect contrasts
                for index, contrast_id in enumerate(contrasts):

                    # this is an adpation of the function _compute_fixed_effect_contrast
                    con_ = contrasts[contrast_id]

                    # the specific code of  _compute_fixed_effect_contrast

                    contrast = None
                    n_contrasts = 0

                    for i, (lab, res, con_val) in enumerate(zip(labels_all, estimates_all, con_)):

                        if np.all(con_ == 0):
                            warn('Contrast for session %d is null' % i)
                            continue
                        contrast_ = compute_contrast(lab, res, con_)

                        if contrast is None:
                            contrast = contrast_

                        else:
                            contrast = contrast + contrast_
                        n_contrasts += 1

                        if contrast is None:
                            raise ValueError('all contrasts provided were null contrasts')

                        else:
                            contrast_fixed = contrast * (1. / n_contrasts)

                    # save the z map, t_values map, effect and variance map
                    stats = [contrast_fixed.z_score(), contrast_fixed.stat(), contrast_fixed.effect, contrast_fixed.variance]

                    for map_type, out_map in zip(map_types, stats):

                        path_output_glm_fixed = os.path.join(path_output_base, folder_sub_glm_2, subj_ID, ses_ID, '%s_surf' % map_type)
                        os.makedirs(path_output_glm_fixed, exist_ok=True)

                        file_map = os.path.join(path_output_glm_fixed, '%s_%s_task-%s_%s_t_test_%s_cope1.ptseries.nii' % (subj_ID, ses_ID, task, contrast_id, map_type))

                        if not os.path.exists(file_map):
                            if len(out_map.shape) == 1:
                                my_values = out_map.reshape(1, len(out_map))
                            elif len(out_map.shape) == 2:
                                my_values = out_map
                            new_img = nib.Cifti2Image(my_values, header=cifti_template.header, nifti_header=cifti_template.nifti_header)
                            new_img.to_filename(file_map)

                    # save degree of freedom
                    file_dof = os.path.join(path_output_base, folder_sub_glm_2, subj_ID, ses_ID, 'degree_freedom.txt')

                    with open(file_dof, 'w') as f:
                        f.write("degree of freedome %d \n" % contrast_fixed.dof)
                        f.write("test type %s \n" % contrast_fixed.contrast_type)


            # except:
            #     print (subj_ID, ' check the data')

# step 3: run the random effect across participants

path_base_2 = '%s/level_2' % (path_output_base)

path_output_3 = '%s/level_3' % (path_output_base)

thre_p = 'no_thre'


#%% step 3: run the random effect across participants
# use different range for different contrast

path_base_2 = '%s/level_2' % (path_output_base)

path_output_3 = '%s/level_3' % (path_output_base)

space = 'cifti'
cmap = 'nipy_spectral'

thre_p = 'no_thre'

p_thre_corr = 0.05

for contrast in contrasts_types:

    # step 3 - 1: get all the z maps of all the participants for each contrast
    z_maps = glob.glob('%s/sub-???????/%s/z_surf/*_%s_task-%s_%s_t_test_z_cope1.ptseries.nii' % (path_base_2, ses_ID, ses_ID, task, contrast))

    # step 3 - 2: read the data of the z maps
    data_z_maps = []

    for z_map in z_maps:
        # todo change this
        data_z_maps.append(np.ravel(nib.load(z_map).get_fdata()))

    # step 3 - 3: one sample t_values test
    t_values, p_values = ttest_1samp(np.array(data_z_maps), 0)

    # t_values = -3 and t_values = 3, the z value is the same;
    # so change the sign of z based on the sign of t_values
    z_val = norm.isf(p_values)
    z_val_full = np.where(t_values > 0, z_val, (-1) * z_val)

    # do the FDR correction
    q_values = multipletests(p_values, method='fdr_by')[1]

    # find the networks that show significant activations and deactivations
    t_values_thre = np.where(q_values < p_thre_corr, t_values, np.nan)

    path_map = '%s/%s/stat_%s_thre_p_%s' % (path_output_3, ses_ID, task, thre_p)
    os.makedirs(path_map, exist_ok=True)
    stat_map_t = os.path.join( path_map , '%s_t_map.ptseries.nii' % (contrast))
    stat_map_z = os.path.join( path_map , '%s_z_map.ptseries.nii' % (contrast))

    path_map_corr = '%s/%s/stat_%s_FDR_p_%s' % (path_output_3, ses_ID, task,p_thre_corr)
    os.makedirs(path_map_corr, exist_ok=True)

    stat_map_t_FDR = os.path.join(path_map_corr, '%s_t_map_FDR_%s.ptseries.nii' % (contrast, p_thre_corr))

    stat_maps = [stat_map_t, stat_map_z, stat_map_t_FDR]
    stat_values = [t_values, z_val_full,t_values_thre]

    for i in range(len(stat_values)):
        stat_value = stat_values[i]
        stat_map = stat_maps[i]
        if not os.path.exists(stat_map):
            my_values = stat_value.reshape(1, len(stat_value))
            new_img = nib.Cifti2Image(my_values, header=cifti_template.header, nifti_header=cifti_template.nifti_header)
            new_img.to_filename(stat_map)

    # save network names and networks_names
    path_map_corr_network = '%s/%s/stat_%s_FDR_p_%s_networks' % (path_output_3, ses_ID, task, p_thre_corr)
    os.makedirs(path_map_corr_network, exist_ok=True)
    filename_prefix = 'task-%s_contrast-%s_t' % (task, contrast)
    parcel_IDs_pos, parcel_IDs_neg = gene_table_sig_parcels_Schaefer_417(t_values, p_values, path_map_corr_network, filename_prefix)

    # save the dlabel.nii
    path_map_corr_label = '%s/%s/stat_%s_FDR_p_%s_dlabel' % (path_output_3, ses_ID, task, p_thre_corr)
    os.makedirs(path_map_corr_label, exist_ok=True)

    path_map_corr_label_fig = '%s/%s/stat_%s_FDR_p_%s_dlabel_fig' % (path_output_3, ses_ID, task, p_thre_corr)
    os.makedirs(path_map_corr_label_fig, exist_ok=True)

    stat_map_t_FDR_pos = os.path.join(path_map_corr_label, 'task-%s-contrast-%s_map-t_FDR-%s_pos.dlabel.nii' % (task,contrast, p_thre_corr))
    stat_map_t_FDR_neg = os.path.join(path_map_corr_label, 'task-%s-contrast-%s_map-t_FDR-%s_neg.dlabel.nii' % (task,contrast, p_thre_corr))

    fig_map_t_FDR_pos = os.path.join(path_map_corr_label_fig, 'task-%s-contrast-%s_map-t_FDR-%s_pos.png' % (task, contrast, p_thre_corr))
    fig_map_t_FDR_neg = os.path.join(path_map_corr_label_fig, 'task-%s-contrast-%s_map-t_FDR-%s_neg.png' % (task, contrast, p_thre_corr))

    title_base = 'Asso-contrast-%s_map-t_FDR-%s' % ( contrast, p_thre_corr)

    if type(parcel_IDs_pos)==np.ndarray and parcel_IDs_pos.shape[0]> 0:
        if not os.path.exists(stat_map_t_FDR_pos):
            save_dlabel_Kong(parcel_IDs_pos, stat_map_t_FDR_pos)

        # plot it
        if not os.path.exists(fig_map_t_FDR_pos):
            plot_Kong_dlabel(parcel_IDs_pos, fig_map_t_FDR_pos, title=title_base + '_pos',title_position = 5, title_position_vertical= -5)
    print (stat_map_t_FDR_neg)
    if type(parcel_IDs_neg)==np.ndarray and  parcel_IDs_neg.shape[0]> 0:
        if not os.path.exists(stat_map_t_FDR_neg):
            save_dlabel_Kong(parcel_IDs_neg, stat_map_t_FDR_neg)

        if not os.path.exists(fig_map_t_FDR_neg):
            plot_Kong_dlabel(parcel_IDs_neg, fig_map_t_FDR_neg, title=title_base + '_neg', title_position=5, title_position_vertical= -5)



# #
# for contrast in contrasts_types:
#
#     # step 3 - 1: get all the z maps of all the participants for each contrast
#     z_maps = glob.glob('%s/sub-???????/%s/z_surf/*%s_task-%s_%s_t_test_z_cope1.ptseries.nii' % (path_base_2, ses_ID,  ses_ID,task, contrast))
#
#     # step 3 - 2: read the data of the z maps
#     data_z_maps = []
#
#     for z_map in z_maps:
#         data_z_maps.append(np.ravel(load(z_map).get_fdata()))
#
#     # step 3 - 3: one sample t_values test
#     t, pval = ttest_1samp(np.array(data_z_maps), 0)
#
#     # change the sign of z value based on t_values value
#     z_val = norm.isf(pval)
#     z_val_full = np.where(t > 0, z_val, (-1) * z_val)
#
#     # step 3 - 5: plot it
#     path_fig = '%s/%s/fig_%s_thre_p_%s' % (path_output_3, ses_ID, task,thre_p)
#     os.makedirs(path_fig, exist_ok=True)
#
#     # save t_values map as nii
#     path_map = '%s/%s/stat_%s_thre_p_%s' % (path_output_3, ses_ID, task, thre_p)
#     os.makedirs(path_map, exist_ok=True)
#     stat_map_t = os.path.join(path_map, '%s_t_map.ptseries.nii' % (contrast))
#     stat_map_z = os.path.join(path_map, '%s_z_map.ptseries.nii' % (contrast))
#
#     stat_maps = [stat_map_t, stat_map_z]
#     stat_values = [t, z_val_full]
#
#     for i in range(2):
#         stat_value = stat_values[i]
#         stat_map = stat_maps[i]
#
#         if not os.path.exists(stat_map):
#             my_values = np.array(stat_value).reshape(1, len(stat_value))
#             new_img = nib.Cifti2Image(my_values, header=cifti_template.header, nifti_header=cifti_template.nifti_header)
#             new_img.to_filename(stat_map)