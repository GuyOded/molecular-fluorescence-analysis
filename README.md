# README
The following are utilities for the sake of analyzing the results for an experiment in molecular fluorescence.

## Installation
Make sure all dependencies are installed before opening the main notebook file (it is recommended to use a venv)
Make sure the python version 3.12.4 or higher before running.
```code
[optional] python -m venv <name>
[optional] # Activate your venv

pip install -r requirements.txt
```

## Running
Open the ipython notebook file with your favorite program (vscode is recommended) and run all the codeblocks.
All the results for the first and last parts of the experiment are outlined in the notebook.

On the first code block you can find a lot of paths. Make sure to change them to the folder containing the results
of the experiment. The results are expected to be in some tabular file format like excel, ods or csv and have the following form:

| Wavelength \[nm\] | Intensity | Integration Time |
|-------------------|-----------|------------------|
| <values>          | <values>  | <value>          |

### Part 2
For the second part there is a separate file, namely part_2.py.
Make sure to the change the paths constants at the top of the file.
Here the sheets have to be in excel format and contain all the measurements for a specific material in different sheets of the same file.
Each sheet has the following format:

| x \[nm\] | Av       | Avl     |
|----------|----------|---------|
| <values> | <values> | <value> |

Where x, Av, Avl are all outputs of the matlab script partb.m
