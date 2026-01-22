'''
Script to save all galaxies that are present in the Galaxy Zoo 2 catalog and Shamir catalog by RA/DEC.
Only saves galaxies that are "superclean" in GZ2 ("t04_spiral_a08_spiral_weighted_fraction" > 0.95)
'''

import csv

def load_csv(file_path):
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        data = [row for row in reader]
    return data

def main():
    '''
    1) Check GZ2 for superclean spirals.
    2) Cross-match with Shamir catalog by RA/DEC.
    3) Save matched galaxies to new CSV.
    '''
    
    galaxy_zoo_csv = '../csv/galaxy_zoo_2.csv'
    shamir_csv = '../csv/shamir_catalog.csv'

    galaxy_zoo_data = load_csv(galaxy_zoo_csv)
    shamir_data = load_csv(shamir_csv)

    # Check GZ2 for superclean spirals
    superclean_spirals = []
    skipped_rows = 0
    value_error_count = 0
    total_rows = 0
    valid_spiral_fractions = 0
    
    for row in galaxy_zoo_data[1:]:
        total_rows += 1
        try:
            spiral_fraction = float(row[54])  # Column 54 (0-indexed) = "t04_spiral_a08_spiral_weighted_fraction"
            valid_spiral_fractions += 1
            if spiral_fraction >= 0.95:
                superclean_spirals.append([row[1], row[2]])
        except ValueError:
            value_error_count += 1
            if value_error_count <= 3:
                print(f"ValueError - Column 54 value: '{row[54]}' (length {len(row[54])})")
            continue
        except IndexError as e:
            skipped_rows += 1
            if skipped_rows <= 3:
                print(f"IndexError - Row length: {len(row)}, First few fields: {row[:3] if len(row) >= 3 else row}")
            continue
    
    print(f"Found {len(superclean_spirals)} superclean spirals (>= 0.95)")
    
    shamir_dict = {} # Key: (rounded_ra, rounded_dec), Value: none
    for shamir_row in shamir_data[1:]:
        try:
            shamir_ra = float(shamir_row[1])
            shamir_dec = float(shamir_row[2])
        
            key = (shamir_ra, shamir_dec)
            if key not in shamir_dict:
                shamir_dict[key] = []
        except (ValueError, IndexError):
            continue
    
    print(f"Shamir catalog indexed with {len(shamir_dict)} bins")
    print("Cross-matching...")
    
    # Cross-match with Shamir catalog by RA/DEC
    matched_galaxies = []
    num_matches = 0
    tolerance = 0.0001
    
    for i, gz_row in enumerate(superclean_spirals):
        if i % 500 == 0:
            print(f"Processed {i}/{len(superclean_spirals)} spirals with {num_matches} current matches...")
        
        # RA/DEC of the superclean spirals
        ra = float(gz_row[0])      
        dec = float(gz_row[1])
        
        # Iterate over Shamir dictionary for possible matches
        for shamir_key in shamir_dict.keys():
            shamir_ra, shamir_dec = shamir_key

            if abs(ra - shamir_ra) <= tolerance and abs(dec - shamir_dec) <= tolerance:
                matched_galaxies.append([ra, dec])
                num_matches += 1
                break 
    
    print(f"Number of matched galaxies: {num_matches}")

    # Save matched galaxies to new CSV
    output_csv = '../csv/matched_galaxies.csv'
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['RA', 'DEC'])
        writer.writerows(matched_galaxies)

if __name__ == "__main__":
    main()