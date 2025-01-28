import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from nilearn import image, plotting
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.second_level import SecondLevelModel
import matplotlib.pyplot as plt
from scipy.stats import norm


def create_project_directories(base_path):
    """
    Create essential directory structure for the project.

    Args:
        base_path (str): Base path for the project directory

    Returns:
        tuple: Paths for project, checkpoints, and results directories
    """
    project_dir = Path(base_path)
    checkpoints_dir = project_dir / 'checkpoints'
    results_dir = project_dir / 'results'

    for dir_path in [checkpoints_dir, results_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    return project_dir, checkpoints_dir, results_dir


# noinspection PyTypeChecker
def save_checkpoint(data, filename: str, checkpoints_dir: Path) -> None:
    """
    Save intermediate processing results.

    Args:
        data: Any Python object to be saved
        filename (str): Name of the checkpoint file
        checkpoints_dir (Path): Directory to save checkpoints
    """
    filepath = checkpoints_dir / filename
    with open(str(filepath), 'wb') as f:
        pickle.dump(data, f)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(filename: str, checkpoints_dir: Path):
    """
    Load previously saved processing results.

    Args:
        filename (str): Name of the checkpoint file
        checkpoints_dir (Path): Directory containing checkpoints

    Returns:
        Loaded data if checkpoint exists, None otherwise
    """
    filepath = checkpoints_dir / filename
    if filepath.exists():
        with open(str(filepath), 'rb') as f:
            return pickle.load(f)
    return None


def create_events_dataframe():
    """
    Create events DataFrame with emotion categories mapped to neutral, positive, negative.

    Returns:
        pd.DataFrame: Events DataFrame
    """
    return pd.DataFrame({
        'onset': [0, 60, 120, 180, 240, 300, 360, 420, 480, 540],
        'duration': [30] * 10,
        'trial_type': ['neutral', 'negative', 'positive', 'negative',
                       'positive', 'positive', 'negative', 'neutral',
                       'positive', 'negative']
    })


def process_single_subject(subject_id, events, checkpoints_dir):
    """
    Process fMRI data for a single subject.

    Args:
        subject_id (int): Subject identifier
        events (pd.DataFrame): Events DataFrame
        checkpoints_dir (Path): Directory for checkpoints

    Returns:
        dict: Contrast maps for the subject
    """
    subject_id_str = f'{subject_id:02d}'
    subject_results = load_checkpoint(f'subject_{subject_id_str}_results.pkl', checkpoints_dir)

    if subject_results is None:
        func_file = f'ds005700/sub-{subject_id_str}/func/sub-{subject_id_str}_task-fe_bold.nii.gz'
        func_img = image.load_img(func_file)

        model = FirstLevelModel(
            t_r=2.02697,
            noise_model='ar1',
            standardize=True,
            hrf_model='spm',
            drift_model='cosine'
        )
        model.fit(func_img, events)

        contrasts = {
            'positive_vs_neutral': 'positive - neutral',
            'negative_vs_neutral': 'negative - neutral',
            'positive_vs_negative': 'positive - negative'
        }

        contrast_maps = {
            contrast_id: model.compute_contrast(contrast_def)
            for contrast_id, contrast_def in contrasts.items()
        }

        subject_results = {'contrasts': contrast_maps}
        save_checkpoint(subject_results, f'subject_{subject_id_str}_results.pkl', checkpoints_dir)
    else:
        print(f"Loaded existing results for subject {subject_id_str}")
        contrast_maps = subject_results['contrasts']

    return contrast_maps


def run_group_analysis(all_subject_contrasts, n_subjects):
    """
    Perform group-level analysis on contrast maps from all subjects.

    Args:
        all_subject_contrasts (list): List of contrast maps for all subjects
        n_subjects (int): Number of subjects

    Returns:
        dict: Group-level results for each contrast
    """
    design_matrix = pd.DataFrame({
        'intercept': np.ones(n_subjects),
    })

    second_level_model = SecondLevelModel(
        smoothing_fwhm=8.0,
        n_jobs=1,
        memory=None,
        verbose=0
    )

    contrasts_to_analyze = [
        'positive_vs_neutral',
        'negative_vs_neutral',
        'positive_vs_negative'
    ]

    group_results = {}

    for contrast_name in contrasts_to_analyze:
        print(f"Running group analysis for {contrast_name}...")

        contrast_maps_list = []
        first_map = all_subject_contrasts[0][contrast_name]
        reference_affine = first_map.affine
        reference_shape = first_map.shape[:3]

        for subject_contrasts in all_subject_contrasts:
            contrast_img = subject_contrasts[contrast_name]

            if len(contrast_img.shape) == 4:
                contrast_img = image.index_img(contrast_img, 0)

            resampled_img = image.resample_img(
                contrast_img,
                target_affine=reference_affine,
                target_shape=reference_shape,
                interpolation='continuous',
                force_resample=True,
                copy_header=True,
                clip=True
            )

            contrast_maps_list.append(resampled_img)

        try:
            second_level_model.fit(contrast_maps_list, design_matrix=design_matrix)
            z_map = second_level_model.compute_contrast(
                output_type='z_score',
                second_level_stat_type='t'
            )
            p_map = second_level_model.compute_contrast(
                output_type='p_value',
                second_level_stat_type='t'
            )
            group_results[contrast_name] = (z_map, p_map)
        except Exception as e:
            print(f"Error processing contrast {contrast_name}: {str(e)}")

    return group_results


def plot_group_results(group_results, results_dir):
    """
    Plot and save group-level analysis results.

    Args:
        group_results (dict): Group-level analysis results
        results_dir (Path): Directory to save plot results

    Returns:
        dict: Plotting results information
    """
    plotting_results = {}

    # Convert Z-score threshold to p-value
    z_threshold = 3.1
    p_val = norm.sf(z_threshold)
    print(f"Z-score threshold {z_threshold} corresponds to p-value < {p_val:.3e}")

    # Plot results for each contrast
    for contrast_name, (z_map, p_map) in group_results.items():
        print(f"Plotting results for {contrast_name}...")

        # Create a new figure for each contrast
        fig = plt.figure(figsize=(15, 5))
        plotting.plot_stat_map(
            z_map,
            threshold=z_threshold,
            display_mode='ortho',
            title=f'Group-level {contrast_name}\n'
                  f'(threshold: z>{z_threshold}, p<{p_val:.3e})',
            figure=fig
        )

        # Save the figure
        output_path = results_dir / f'group_analysis_{contrast_name}(Z scores).png'
        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()

        # Store plotting information
        plotting_results[contrast_name] = {
            'z_threshold': z_threshold,
            'p_value': p_val,
            'output_path': str(output_path)
        }

    return plotting_results


def main():
    """Main pipeline function."""
    # Initialize project directories
    project_dir, checkpoints_dir, results_dir = create_project_directories(
        '/Users/daniilsergeyenko/PycharmProjects/fMRI_project'
    )

    # Create or load events DataFrame
    events = load_checkpoint('events.pkl', checkpoints_dir)
    if events is None:
        events = create_events_dataframe()
        save_checkpoint(events, 'events.pkl', checkpoints_dir)

    # Process all subjects
    n_subjects = 40
    all_subject_contrasts = load_checkpoint('subject_contrasts.pkl', checkpoints_dir)

    if all_subject_contrasts is None:
        all_subject_contrasts = []
        for subject_id in range(1, n_subjects + 1):
            contrast_maps = process_single_subject(subject_id, events, checkpoints_dir)
            all_subject_contrasts.append(contrast_maps)
        save_checkpoint(all_subject_contrasts, 'subject_contrasts.pkl', checkpoints_dir)

    # Run group analysis
    group_results = load_checkpoint('group_results.pkl', checkpoints_dir)
    if group_results is None:
        group_results = run_group_analysis(all_subject_contrasts, n_subjects)
        save_checkpoint(group_results, 'group_results.pkl', checkpoints_dir)

    # Plot results
    plotting_results = load_checkpoint('plotting_results.pkl', checkpoints_dir)
    if plotting_results is None:
        plotting_results = plot_group_results(group_results, results_dir)
        save_checkpoint(plotting_results, 'plotting_results.pkl', checkpoints_dir)


if __name__ == '__main__':
    main()