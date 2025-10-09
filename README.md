
For access to dataset and models, please contact kierannc@uci.edu


## Set up:
1. Clone repository
2. Create virtual environment
3. Activate virtual environment: on linux command line, type 'source bin/activate'
4. Navigate to /src directory
5. Run scripts and pip install as requested by scripts (run by typing ```python [filename]```, e.g. ```python sep_one_file.py```)

## Useful Scripts Available:
1. ```sep_one_file```:
    When Python script is run, program will request for filepath/filename of a ```.fits``` file. Script will remove open the ```.fits``` file and extract all galaxies larger than 15 pixels in image. Extracted galaxies (aka thumbnails) will be in the same directory as the original fits file. Each image is a ```.fits``` file with cards. Cards contain information such as size of galaxy and distance of galaxy.

2. ```view_thumbnail```:
    Script displays ```.fits``` image(s) using ```matplotlib``` and prints all cards of the image(s) on the command line.  Mainly used to view thumbnails that have been extracted. Takes can display a single ```.fits``` image or can display all the fits images in a directory.

3. ```sep_from_mast```:
    Script will request for a filepath/filename of a ```.csv``` file with *mast_uri* as one of the columns.
        Example input: ```sample_data/sample_uri.csv```.
        ```.csv``` files can be downloaded from the mast database after filtering: https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html.
        In the portal, go to advanced search, select desired filter criteria and select 'export table' to get the csv file.
    Script will run the file just like how the 'sep_one_file.py' runs file. The thumbnails will be the *'output/objects_from_mast_csv'* directory

4. ```train_random_forest```:
    Trains different random forests to determine best hyperparameters. 
    There are 6 parameters to add in the command line when running the script. Template for running script:
        ```python train_random_forest.py [num trees start] [num trees end] [num trees step] [num features start] [num features end] [num features step] [num buckets start] [num buckets end] [num buckets step]``` (e.g. "```python train_random_forest.py 10 51 10 20 31 5 1 10 1```")
    Note that this script will only train the models. Run ```train_save_random_forest.py``` to train and save model.

5. ```plt_random_forest```:
    Script will plot a plot for on ```.csv``` files in the *random_forest_output* directory. **P_spiral_predicted** will be plotted against **P_spiral**. Plots will appear in the same directory. 

6. ```train_save_random_forest```:
    Trains one particular model given a certain hyperparameter and saves the label encoder together with the random forest model. There are 4 parameters to add in the command line when running the script. Template for running script: 
        ```python train_save_random_forest.py [num trees] [num featurs] [file for model to save to (must be .pkl file)] [file to save label encoder to (must be .pkl file)]```.
 
7. ```infer_random_forest```: Still under development. Given a dataset, uses a given random forest and label encoder model to determine probabilty of spirality based on sparcfire inputs. 


***SpArcFiRe is run externally. Refer to sparcfire repository on how to run with it. This repository does not supply code to run sparcfire.***


## Known Errors:
1. None

## Resolved Errors:
1. For extracted thumbnails, the RA and DEC information is the position of the original image and not accurate to the galaxy in the thumbnail  ->  RA and DEC is now decoded from the position of the image. 