import numpy as np
from nilearn import plotting, image
from nilearn.datasets import fetch_atlas_harvard_oxford
import matplotlib.pyplot as plt


def explore_atlas():
    """Interactive tool to explore Harvard-Oxford atlas indices"""
    # Fetch both atlases
    print("Fetching atlases...")
    cortical = fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    subcortical = fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')

    # Print available labels
    print("\nCortical regions:")
    for i, label in enumerate(cortical.labels):
        if label:  # Skip empty labels
            print(f"Index {i}: {label}")

    print("\nSubcortical regions:")
    for i, label in enumerate(subcortical.labels):
        if label:  # Skip empty labels
            print(f"Index {i}: {label}")

    def view_region(atlas_type='cort', index=None):
        """View a specific region"""
        atlas = cortical if atlas_type == 'cort' else subcortical
        if index is None:
            return

        # Create mask for the specified index
        mask = image.math_img(f'img == {index}', img=atlas.maps)

        # Create figure with multiple views
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle(f'Region: {atlas.labels[index]}', fontsize=16)

        # Different views
        views = [
            ('x', [-30, 0, 30], 'Sagittal'),
            ('y', [-20, 0, 20], 'Coronal'),
            ('z', [-20, 0, 20], 'Axial'),
            ('ortho', [0, 0, 0], 'Combined')
        ]

        for (mode, cuts, title), ax in zip(views, axes.ravel()):
            if mode == 'ortho':
                display = plotting.plot_roi(
                    mask,
                    display_mode=mode,
                    cut_coords=[0, 0, 0],
                    title=title,
                    axes=ax
                )
            else:
                display = plotting.plot_roi(
                    mask,
                    display_mode=mode,
                    cut_coords=cuts,
                    title=f'{title} view',
                    axes=ax
                )

        plt.tight_layout()
        plt.show()

    while True:
        print("\nOptions:")
        print("1. View cortical region")
        print("2. View subcortical region")
        print("3. Exit")

        choice = input("Enter your choice (1-3): ")

        if choice == '3':
            break
        elif choice in ['1', '2']:
            atlas_type = 'cort' if choice == '1' else 'sub'
            try:
                index = int(input("Enter region index to view: "))
                view_region(atlas_type, index)
            except ValueError:
                print("Please enter a valid number")
        else:
            print("Invalid choice")


if __name__ == "__main__":
    explore_atlas()
