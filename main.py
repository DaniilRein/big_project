import numpy as np
import pandas as pd
from nilearn import image, input_data, plotting
from nilearn.glm.first_level import FirstLevelModel
from nilearn.plotting import plot_stat_map
import matplotlib.pyplot as plt
from scipy import stats
import os


def create_events_df():
    """
    Create events DataFrame from the timing information in the 'README' file in the 'raw_data' folder.
    """
    events = pd.DataFrame({
        'onset': [0, 60, 120, 180, 240, 300, 360, 420, 480, 540],
        'duration': [30] * 10,
        'emotion': ['calm', 'afraid', 'delighted', 'depressed',
                    'excited', 'delighted', 'depressed', 'calm',
                    'excited', 'afraid']
    })

    valence_mapping = {
        'calm': 'neutral',
        'afraid': 'negative',
        'delighted': 'positive',
        'depressed': 'negative',
        'excited': 'positive'
    }
    events['trial_type'] = events['emotion'].map(valence_mapping)
    return events


def load_and_prepare_data(subject_id):
    """Load and prepare fMRI data for a single subject"""
    func_file = f'ds005700/sub-{subject_id}/func/sub-{subject_id}_task-fe_bold.nii.gz'
    anat_file = f'ds005700/sub-{subject_id}/anat/sub-{subject_id}_T1w.nii.gz'

    func_img = image.load_img(func_file)
    anat_img = image.load_img(anat_file)

    return func_img, anat_img
# def load_and_prepare_data(subject_id):
#     """Load and prepare fMRI data for a single subject"""
#     # Get the absolute path to the script's directory
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#
#     # Construct paths using os.path.join
#     func_file = os.path.join(script_dir, 'raw_data', f'sub-{subject_id}', 'func',
#                              f'sub-{subject_id}_task-fe_bold.nii.gz')
#     anat_file = os.path.join(script_dir, 'raw_data', f'sub-{subject_id}', 'anat',
#                              f'sub-{subject_id}_T1w.nii.gz')
#
#     # Add error checking
#     if not os.path.exists(func_file):
#         raise FileNotFoundError(f"Functional image not found at: {func_file}")
#     if not os.path.exists(anat_file):
#         raise FileNotFoundError(f"Anatomical image not found at: {anat_file}")
#
#     func_img = image.load_img(func_file)
#     anat_img = image.load_img(anat_file)
#
#     return func_img, anat_img

# %%

def create_masks():
    """
    Create ROI masks for emotional processing regions
    The 'anatomy_atlas_visualizer.py' is used to list the harvard_oxford atlas and manually choose
    the cortical areas of interest.
    Because my research question addresses activity/inhibition in known brain areas associated with emotion
    then there is no need to process the information from the entire brain. My project is based on the past
    literature base, not attempting to discovery anything novel.
    Therefore, by choosing a region of interest(ROI) based on accepted norms about brain area functioning,
    I save up on significant time and computational resources by analyzing only specific parts of each subject's brain data.

    Area + atlas index
    Superior Frontal Gyrus 3
    Middle Frontal Gyrus 4
    Cingulate Gyrus, anterior division 29
    Left Amygdala 10
    Right Amygdala 20
    Left Hippocampus 9
    Right Hippocampus 19
    Frontal Medial Cortex 25
    Frontal Orbital Cortex 33
    Frontal Opercular Cortex 41

    Perhaps change to include more areas when repeating the study.
    """
    from nilearn.datasets import fetch_atlas_harvard_oxford

    cortical = fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    subcortical = fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')

    roi_masks = {
        # Subcortical regions
        'amygdala_left': image.math_img('img == 10', img=subcortical.maps),
        'amygdala_right': image.math_img('img == 20', img=subcortical.maps),
        'hippocampus_left': image.math_img('img == 9', img=subcortical.maps),
        'hippocampus_right': image.math_img('img == 19', img=subcortical.maps),

        # Cortical regions
        'acc': image.math_img('img == 29', img=cortical.maps),
        'sup_frontal_gyrus': image.math_img('img == 3', img=cortical.maps),
        'mid_frontal_gyrus': image.math_img('img == 4', img=cortical.maps),
    }

    # Create bilateral masks
    bilateral_pairs = [
        ('amygdala', 'amygdala_left', 'amygdala_right'),
        ('hippocampus', 'hippocampus_left', 'hippocampus_right'),
    ]

    # Combine two halves of bilateral pairs into a single image
    for name, left, right in bilateral_pairs:
        roi_masks[f'{name}_bilateral'] = image.math_img(
            "img1 + img2",
            img1=roi_masks[left],
            img2=roi_masks[right]
        )

    return roi_masks, cortical, subcortical


def single_subject_analysis(subject_id, events_df, roi_masks):
    """Perform first-level analysis for a single subject"""
    func_img, anat_img = load_and_prepare_data(subject_id)

    model = FirstLevelModel(
        t_r=2.02697,  # Time repetition for each functional scan in seconds, as found in each subjects' .json file
        noise_model='ar1',
        standardize=True,
        hrf_model='spm',
        drift_model='cosine'
    )

    model.fit(func_img, events_df)

    contrasts = {
        'positive_vs_neutral': 'positive - neutral',
        'negative_vs_neutral': 'negative - neutral',
        'positive_vs_negative': 'positive - negative'
    }

    contrast_maps = {}
    for contrast_id, contrast_def in contrasts.items():
        contrast_maps[contrast_id] = model.compute_contrast(contrast_def)

    roi_signals = {}
    for roi_name, roi_mask in roi_masks.items():
        masker = input_data.NiftiMasker(mask_img=roi_mask)
        roi_signals[roi_name] = masker.fit_transform(func_img)

    return contrast_maps, roi_signals


def group_analysis(all_subject_contrasts):
    """Perform group-level analysis"""
    from nilearn.glm import second_level

    group_model = second_level.SecondLevelModel()

    group_results = {}
    for contrast_id in all_subject_contrasts[0].keys():
        contrast_maps = [subj_contrasts[contrast_id]
                         for subj_contrasts in all_subject_contrasts]
        group_results[contrast_id] = group_model.fit(contrast_maps).compute_contrast()

    return group_results


def plot_results(group_results, output_dir):
    """Plot and save group-level results"""
    for contrast_id, stat_map in group_results.items():
        display = plotting.plot_stat_map(
            stat_map,
            threshold=3.0,
            title=contrast_id
        )
        plt.savefig(f'{output_dir}/{contrast_id}_group_map.png')
        plt.close()


def main():
    n_subjects = 40
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)

    events_df = create_events_df()

    all_subject_contrasts = []
    all_subject_roi_signals = []

    roi_masks, _, _ = create_masks()

    for subject_id in range(1, n_subjects + 1):
        subject_id_str = f'{subject_id:02d}'
        print(f'Processing subject {subject_id_str}...')

        contrasts, roi_signals = single_subject_analysis(
            subject_id_str, events_df, roi_masks
        )

        all_subject_contrasts.append(contrasts)
        all_subject_roi_signals.append(roi_signals)

    group_results = group_analysis(all_subject_contrasts)
    plot_results(group_results, output_dir)


if __name__ == '__main__':
    main()