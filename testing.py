import pandas as pd

# Number of subjects
n_subjects = 40

# Initialize the design matrix
design_matrix = pd.DataFrame({
    "intercept": [1] * n_subjects,  # Intercept column
    "Calm": [1, 0, 0, 0, 0] * (n_subjects // 5),  # Calm condition
    "Afraid": [0, 1, 0, 0, 0] * (n_subjects // 5),  # Afraid condition
    "Delighted": [0, 0, 1, 0, 0] * (n_subjects // 5),  # Delighted condition
    "Depressed": [0, 0, 0, 1, 0] * (n_subjects // 5),  # Depressed condition
    "Excited": [0, 0, 0, 0, 1] * (n_subjects // 5),  # Excited condition
    "Positive": [0, 0, 1, 0, 1] * (n_subjects // 5),  # Positive valence
    "Negative": [0, 1, 0, 1, 0] * (n_subjects // 5),  # Negative valence
    "Neutral": [1, 0, 0, 0, 0] * (n_subjects // 5),  # Neutral valence
})

# Display the design matrix
print(design_matrix.head())