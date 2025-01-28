
The code is set-up in a way that makes the process linear instead of
modular by functions using jupyter notebooks. After each step - code block - the results are
saved in the project directory/checkpoints to prevent unnecessary repetitions.

In the original README it is stated that preprocessing was already performed as such:
""
Data have been preprocessed using:
- SPM12
  - Motion correction
  - Slice-timing correction
  - Co-registration
  - Segmentation
  - Normalization to MNI space
  - Smoothing
""
  - 
Averaging the entire 30 seconds of each condition, 2 conditions per emotion group (positive/negative)
then averaging across 40 subjects - allows the basic full-brain t-tests more statistically significant. 

MEG and EEG couldn't have given accurate subcortical reading which is most correlated with emotion.
Then using fMRI for this question is ideal. Also, it saves the trouble of the inverse problem faced by those two methods.

how outsider can use the pipeline.


## Primary Source/Article
Article/s link/s

## Project Data:
Link to project's raw data (first input)

## Project Documentation:
Project Workflow/Pipeline (Either included here or linked to a Drive document)

## To run the project follow this commands:
All command should run under project root/working-directory
```bash 
#install Virtualenv is - a tool to set up your Python environments
pip install virtualenv
#create virtual environment (serve only this project):
python -m venv venv
#activate virtual environment
.\venv\Scripts\activate
+ (venv) should appear as prefix to all command (run next command just after activating venv)
#update venv's python package-installer (pip) to its latest version
python.exe -m pip install --upgrade pip
#install projects packages (Everything needed to run the project)
pip install -e .
#install dev packages (Additional packages for linting, testing and other developer tools)
pip install -e .[dev]
``` 

