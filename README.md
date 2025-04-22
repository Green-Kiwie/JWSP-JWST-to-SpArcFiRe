Set up:
1. clone repository
2. create virtual environment
3. Activate virtual environment: on linux command line, type 'source bin/activate'
4. navigate to /src directory
5. run scripts and pip install as requested by scripts (run by typing "python [filename]", e.g. python sep_one_file.py)

Scripts available:
1. sep_one_file:
    when python script is run, program will request for filepath/filename of a fits file. example input: 'sample_data/image3.fits'
    script will remove open fits file and extract all galaxies larger than 15 pixels in image. 
    extracted galaxies (aka thumbnails) will be in the same directory as the original fits file. Each image is a fits files with cards.
    cards contain information such as size of galaxy and distance of galaxy

2. view_thumbnail:
    script displays a fits image using matplotlib and prints all cards of the image on the commandline. 
    mainly used to view thumbnails that have been extracted.
    takes can display a single fits image or can display all the fits images in a directory

3. sep_from_mast:
    script will request for a filepath/filename of a csv file with mast_uri as one of the columns.
        example input: 'sample_data/sample_uri.csv'
        such csv can be downloaded from the mast database after filtering: https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html
        In the portal, go to advanced search, select desired filter criteria and select 'export table' to get the csv file
    script will run the sep just like how the 'sep_one_file.py' runs sep
    the thumbnails will be the 'output/objects_from_mast_csv' directory

4. train_random_forest:
    trains different random forests to determine which is the best hyperparameters. 
    There are 6 parameters to add in the command line when running the script. Template for running script: "python train_random_forest.py [num trees start] [num trees end] [num trees step] [num features start] [num features end] [num features step] [num buckets start] [num buckets end] [num buckets step]" e.g. "python train_random_forest.py 10 51 10 20 31 5 1 10 1". 

5. plt_random_forest:
    Script will plot a plot for on csv viles in the "random_forest_output" directory. P_spiral_predicted will be plotted against P_spiral.
    Plots will appear in the same directory. 
 


*sparcfire is run externally. Refer to sparcfire repository on how to run with it. This repository does not supply code to run sparcfire. 


known errors:


Resolved errors:
1. for extracted thumbnails, the RA and DEC information is the position of the original image and not accurate to the galaxy in the thumbnail  --  RA and DEC is now decoded from the position of the image. 