# check the activation of the words vs nonwords
# fMRIPrep, xcp_abcd, individual parcellation

# build the glm to investigate the language vs non-language

import os
from nilearn.glm.first_level import run_glm
from nilearn.glm.first_level import make_first_level_design_matrix, check_design_matrix
from nilearn.glm import compute_contrast
import numpy as np
import nibabel as nib
import pandas as pd
import glob
from nibabel import load
from scipy.stats import ttest_1samp, norm
import warnings
from function_save_cifti import save_cifti_ptseries
from function_gene_tables_sig_parcel import gene_table_sig_parcels_Schaefer_417
from statsmodels.stats.multitest import multipletests
from function_save_parcels_FDR_Kong_atlas import save_dlabel_Kong
warnings.filterwarnings('ignore', '.*do not.*', )
warnings.warn('DelftStack')
warnings.warn('Do not show this message')

path_input_base = '/home/xiuyi/Data/Task_Gradient_localizers/Brain_data/Nifti/fMRIPrep_cifti_nifti_xcp_abcd/smooth_6_filter_min_0.01_0.08_36P_cifti/ptseries_individual_parcellation_32k'

path_tsv_base = '/home/xiuyi/Data/Task_Gradient_localizers/Behaviour_data/tsv'
path_base_output = '/home/xiuyi/Data/Task_Gradient_localizers/result/fMRIPrep_xcp_Schaefer_indi'

os.makedirs(path_base_output, exist_ok=True)

task = 'Spatial'
trial_types_full = ['correct_Easy', 'correct_Hard', 'wrong_Easy', 'wrong_Hard']
contrast_types = ['Easy','Hard', 'Easy_vs_Hard', 'Hard_vs_Easy','Easy_and_Hard','Easy_neg','Hard_neg', 'Easy_and_Hard_neg']

# participants with excessive head motion need to be deleted
sub_IDs_special = ['sub-08','sub-29','sub-03']

ses_ID = 'ses-02'
run_IDs = ['run-1','run-2']
sub_IDs = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05',
           'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10',
           'sub-11', 'sub-12', 'sub-13', 'sub-14', 'sub-15',
           'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20',
           'sub-21', 'sub-23', 'sub-24', 'sub-25', 'sub-26',
           'sub-27', 'sub-28', 'sub-29', 'sub-30', 'sub-31']

file_suffix = 'space-fsLR_den-91k_desc-residual_smooth_bold.ptseries.nii'

folder = 'func'
hrf_model = 'glover'
TR = 3
col_onset = 'onset'
col_duration = 'duration'
col_trial_type = 'trial_type'
col_RT = 'response_time'
col_modulation = 'modulation'
col_index = 'TR_ID'




folder_sub_glm_1 = 'level_1'

folder_sub_glm_2 = 'level_2'

map_types = ['z', 't_values', 'effects', 'variance']

folders = os.listdir(path_input_base)

folders.sort()

for sub_ID in sub_IDs:

    if sub_ID not in sub_IDs_special:
        print (sub_ID)
        # check whether the output has been saved before
        path_check = os.path.join(path_base_output, folder_sub_glm_2, sub_ID, ses_ID,'%s_surf' % map_types[-1])

        file_check = os.path.join(path_check, '%s_%s_task-%s_%s_t_test_%s_cope1.ptseries.nii' % (sub_ID, ses_ID, task, contrast_types[-1], map_types[-1]))

        if not os.path.exists(file_check):
            # go through each run
            labels_all = []
            estimates_all = []
            con_vals = []

            for run_ID in run_IDs:
                filename_brain =  '%s_%s_task-%s_%s_%s'%(sub_ID,ses_ID,task,run_ID,file_suffix)

                file_fmri = os.path.join(path_input_base, sub_ID, ses_ID, folder, filename_brain)

                file_events = os.path.join(path_tsv_base, sub_ID, ses_ID, folder, '%s_%s_task-%s_%s_events.tsv' % (sub_ID, ses_ID, task, run_ID))

                # check whether each file eixsts
                if not os.path.exists(file_fmri):
                    print (filename_brain,'not exist')

                elif not os.path.exists(file_events):
                    print (sub_ID, ses_ID,run_ID,'tsv file not exist')

                else:
                    # load the brain data and tsv data
                    data_fmri = nib.load(file_fmri).get_fdata()
                    num_scans = data_fmri.shape[0]

                    frame_times = TR * (np.arange(num_scans))

                    # read the events
                    data_events = pd.read_table(file_events)

                    # remove the modulation column
                    data_events.drop(col_modulation, axis=1, inplace=True)

                    # get the current available trial types
                    trial_types_curr = list(np.unique(data_events[col_trial_type].values))

                    # check whether some trial types are missing
                    # find the diff because the full types and curr trial types
                    diff = list(set(trial_types_full) - set(trial_types_curr))

                    # if yes, add one column here for each missing trial type
                    # The values are 0.
                    if len(diff) != 0:

                        # the regressors_confounds.shape should be 144 * len(diff)
                        # but you may get 1 * len(diff)
                        regressors_confounds = pd.DataFrame(index=range(num_scans), columns=diff)
                        if len(diff)==1:
                            regressors_confounds[diff[0]] = 0
                        else:
                            for trial_type_miss in diff:
                                regressors_confounds[trial_type_miss] = 0

                        # create design matrix
                        design_matrix = make_first_level_design_matrix(frame_times, data_events, drift_model='polynomial', drift_order=3, add_regs=regressors_confounds, add_reg_names=regressors_confounds.columns.values, hrf_model=hrf_model)

                    else:
                        design_matrix = make_first_level_design_matrix(frame_times, data_events, drift_model='polynomial', drift_order=3, hrf_model=hrf_model)

                    # check design matrix
                    _, dmtx, names = check_design_matrix(design_matrix)

                    # save design matrix
                    path_output_dm = os.path.join(path_base_output,  folder_sub_glm_1, sub_ID, ses_ID )
                    os.makedirs(path_output_dm, exist_ok=True)

                    filename_dm = '%s_%s_task-%s_%s_design_matrix.npz' % (sub_ID, ses_ID, task, run_ID)
                    np.savez(os.path.join(path_output_dm, filename_dm), design_matrix=design_matrix)

                    # build the contrast
                    contrast_matrix = np.eye(design_matrix.shape[1])

                    basic_contrasts = dict([(column, contrast_matrix[i]) for i, column in enumerate(design_matrix.columns)])
                    contrasts = {'Easy': basic_contrasts['correct_Easy'],
                                 'Hard': basic_contrasts['correct_Hard'],
                                 'Easy_and_Hard': basic_contrasts['correct_Easy'] + basic_contrasts['correct_Hard'],
                                 'Easy_vs_Hard': basic_contrasts['correct_Easy'] - basic_contrasts['correct_Hard'],
                                 'Hard_vs_Easy': basic_contrasts['correct_Hard'] - basic_contrasts['correct_Easy'],
                                 'Easy_neg': - basic_contrasts['correct_Easy'],
                                 'Hard_neg': - basic_contrasts['correct_Hard'],
                                 'Easy_and_Hard_neg': - basic_contrasts['correct_Easy'] - basic_contrasts['correct_Hard']
                                 }

    #
                    labels, estimates = run_glm(data_fmri, design_matrix.values)

                    labels_all.append(labels)
                    estimates_all.append(estimates)

            # convert list to t_values
            labels_all = np.asarray(labels_all)
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
                    stats = [contrast_fixed.z_score(), contrast_fixed.stat(), contrast_fixed.effect,
                             contrast_fixed.variance]

                    # save these stat maps as func.gii
                    for map_type, out_map in zip(map_types, stats):

                        path_output_glm_fixed = os.path.join(path_base_output, folder_sub_glm_2, sub_ID, ses_ID,  '%s_surf' % map_type)
                        os.makedirs(path_output_glm_fixed, exist_ok=True)

                        file_map = os.path.join(path_output_glm_fixed, '%s_%s_task-%s_%s_t_test_%s_cope1.ptseries.nii' % (sub_ID, ses_ID, task, contrast_id, map_type))

                        if not os.path.exists(file_map):
                            if len(out_map.shape) == 1:
                                my_values = out_map.reshape(1, len(out_map))
                            elif len(out_map.shape) == 2:
                                my_values = out_map
                            save_cifti_ptseries(my_values, file_map,parcel='Schaefer')


                    # save degree of freedom
                    file_dof = os.path.join(path_base_output,folder_sub_glm_2, sub_ID, ses_ID, 'degree_freedom.txt')

                    with open(file_dof, 'w') as f:
                        f.write("degree of freedome %d \n" % contrast_fixed.dof)
                        f.write("test type %s \n" % contrast_fixed.contrast_type)


path_base_2 = '%s/level_2' % (path_base_output)

path_output_3 = '%s/level_3' % (path_base_output)

thre_p = 'no_thre'
p_thre_corr = 0.05

for contrast in contrast_types:
    print (contrast)

    # step 3 - 1: get all the z maps of all the participants for each contrast
    z_maps = glob.glob('%s/sub-???????/%s/z_surf/*%s_task-%s_%s_t_test_z_cope1.ptseries.nii' % (path_base_2, ses_ID,  ses_ID,task, contrast))

    # step 3 - 2: read the data of the z maps
    data_z_maps = []

    for z_map in z_maps:
        data_z_maps.append(np.ravel(load(z_map).get_fdata()))

    # step 3 - 3: one sample t_values test
    t_values, p_values = ttest_1samp(np.array(data_z_maps), 0)

    # change the sign of z value based on t_values value
    z_values = norm.isf(p_values)
    z_val_full = np.where(t_values > 0, z_values, (-1) * z_values)

    # do the FDR correction
    q_values = multipletests(p_values, method='fdr_by')[1]

    # find the parcels that show significant activations and deactivations
    t_values_thre = np.where(q_values < p_thre_corr, t_values, np.nan)

    # save t_values map as nii
    path_map = '%s/%s/stat_%s_thre_p_%s' % (path_output_3, ses_ID, task, thre_p)
    os.makedirs(path_map,exist_ok=True)

    stat_map_t = os.path.join(path_map, 'task-%s_contrast-%s_t_map.ptseries.nii' % (task, contrast))
    stat_map_z = os.path.join(path_map, 'task-%s_contrast-%s_z_map.ptseries.nii' % (task, contrast))

    path_map_corr = '%s/%s/stat_%s_FDR_p_%s' % (path_output_3, ses_ID, task,p_thre_corr)
    os.makedirs(path_map_corr, exist_ok=True)

    stat_map_t_FDR = os.path.join(path_map_corr, 'task-%s_contrast-%s_t_map_FDR_%s.ptseries.nii' % (task, contrast, p_thre_corr))

    stat_maps = [stat_map_t, stat_map_z, stat_map_t_FDR]
    stat_values = [t_values, z_val_full,t_values_thre]

    for i in range(len(stat_maps)):
        stat_value = stat_values[i]
        stat_map = stat_maps[i]

        if not os.path.exists(stat_map):
            my_values = np.array(stat_value).reshape(1, len(stat_value))
            save_cifti_ptseries(my_values, stat_map,parcel='Schaefer')

    # save parcel names and networks
    path_map_corr_network = '%s/%s/stat_%s_FDR_p_%s_networks' % (path_output_3, ses_ID, task,p_thre_corr)
    os.makedirs(path_map_corr_network, exist_ok=True)
    filename_prefix = 'task-%s_contrast-%s_t' % (task, contrast)
    parcel_IDs_pos, parcel_IDs_neg = gene_table_sig_parcels_Schaefer_417(t_values, p_values, path_map_corr_network, filename_prefix)

    # save the dlabel.nii
    path_map_corr_label = '%s/%s/stat_%s_FDR_p_%s_dlabel' % (path_output_3, ses_ID, task, p_thre_corr)
    os.makedirs(path_map_corr_label, exist_ok=True)

    stat_map_t_FDR_pos = os.path.join(path_map_corr_label, 'task-%s-contrast-%s_map-t_FDR-%s_pos.dlabel.nii' % (task,contrast, p_thre_corr))
    stat_map_t_FDR_neg = os.path.join(path_map_corr_label, 'task-%s-contrast-%s_map-t_FDR-%s_neg.dlabel.nii' % (task,contrast, p_thre_corr))

    if type(parcel_IDs_pos)==np.ndarray and parcel_IDs_pos.shape[0]> 0:
        if not os.path.exists(stat_map_t_FDR_pos):
            save_dlabel_Kong(parcel_IDs_pos, stat_map_t_FDR_pos)

    if type(parcel_IDs_neg)==np.ndarray and  parcel_IDs_neg.shape[0]> 0:
        if not os.path.exists(stat_map_t_FDR_neg):
            save_dlabel_Kong(parcel_IDs_neg, stat_map_t_FDR_neg)
print ('well done')