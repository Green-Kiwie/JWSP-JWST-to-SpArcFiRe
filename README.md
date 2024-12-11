Set up:
1. clone repository
2. Activate virtual environment: on linux command line, type 'source bin/activate'
3. navigate to /src directory

Scripts available:
1. sep_one_file_py:
    when python script is run, program will request for filepath/filename of a fits file. example input: 'sample_data.image3.fits'
    script will remove open fits file and extract all galaxies larger than 15 pixels in image. 
    extracted galaxies (aka thumbnails) will be in the same directory as the original fits file. Each image is a fits files with cards.
    cards contain information such as size of galaxy and distance of galaxy

2. view_thumbnail:
    script displays the image using matplotlib and prints all cards of the image on the commandline. 
    mainly used to view thumbnails that have been extracted. 


known errors:
1. for extracted thumbnails, the RA and DEC information is the position of the original image and not accurate to the galaxy in the thumbnail 