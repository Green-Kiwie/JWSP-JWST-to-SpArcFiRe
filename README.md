Set up:
1. clone repository
2. Activate virtual environment: on linux command line, type 'source bin/activate'
3. navigate to /src directory

Scripts available:
1. sep_one_file_py:
    when python script is run, program will request for filepath/filename of a fits file. example input: 'sample_data/image3.fits'
    script will remove open fits file and extract all galaxies larger than 15 pixels in image. 
    extracted galaxies (aka thumbnails) will be in the same directory as the original fits file. Each image is a fits files with cards.
    cards contain information such as size of galaxy and distance of galaxy

2. view_thumbnail:
    script displays the image using matplotlib and prints all cards of the image on the commandline. 
    mainly used to view thumbnails that have been extracted.

3. sep_from_mast:
    script will request for a filepath/filename of a csv file with mast_uri as one of the columns.
        example input: 'sample_data/sample_uri.csv'
        such csv can be downloaded from the mast database after filtering: https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html
        In the portal, go to advanced search, select desired filter criteria and select 'export table' to get the csv file
    script will run the sep just like how the 'sep_one_file.py' runs sep
    the thumbnails will be the 'output/objects_from_mast_csv' directory


known errors:
1. for extracted thumbnails, the RA and DEC information is the position of the original image and not accurate to the galaxy in the thumbnail 