import requests
from tqdm import tqdm
import csv
import argparse

# Function to write venues to txt
def write_to_txt(venues):
    with open('venues.txt', 'w') as f:
        for key in venues[0]:
            f.write(str(key) + ", ")
        for venue in venues:
            for key in venue:
                f.write(str(venue[key]) + ", ")
            f.write('\n')

# Function to write venues to csv
def write_to_csv(venues):
    fieldnames = venues[0].keys()
    with open('countries.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(venues)

# Remove venue is not in specified city
def city_filter(venues, city):
    for i, venue in enumerate(venues):
        if venue['city'] != city:
            del venues[i]
    return venues

# Function to get data and store in list
def get_data(api_key, per_page, limit, bbox):
    
    response = requests.get(url = "https://wheelmap.org/api/categories/2/nodes/?api_key="+
                                api_key+"&per_page="+per_page+"&bbox="+bbox+"&limit="+
                                limit+"&page=1")

    venues = response.json()['nodes'] # Initialize venues list
    num_pages = response.json()['meta']['num_pages'] # Get number of pages

    for page in tqdm(range(2, num_pages)): # For every page

        response = requests.get(url = "https://wheelmap.org/api/categories/2/nodes/?api_key="+
                                        api_key+"&per_page="+per_page+"&bbox="+bbox+"&limit="+
                                        limit+"&page="+str(page))
        venues += response.json()['nodes'] # Concatenate to venues list

    # Filter out all venues in bbox that are not in specified city
    if config.city != None:
        return city_filter(venues, config.city)

    return venues

def main():
    
    if config.api_key == None:
        print("You need to give api_key as input.")
    
    venues = get_data(config.api_key, config.per_page, config.limit, config.bbox)

    if config.write_to == 'txt':
        write_to_txt(venues)
    elif config.write_to == 'csv':
        write_to_csv(venues)

if __name__ == '__main__':
    
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', type=str,
                        help='Wheelmap API key to get acces.')
    parser.add_argument('--per_page', type=str, default='500',
                        help='Venues per page. Maximum and default is 500. If you want to get all venues in bbox make sure per_page and limit is equal.')
    parser.add_argument('--limit', type=str, default='500',
                        help='limit per page. Maximum and default is 500. If you want to get all venues in bbox make sure per_page and limit is equal.')
    parser.add_argument('--bbox', type=str, default='4.839323, 52.333451, 4.968438, 52.418032',
                        help='lon and lat values defining a bounding box out of which to get data. Default is bbox of Amsterdam')
    parser.add_argument('--write_to', type=str, default='csv',
                        help='File type to save data in. Can be txt or csv.')
    parser.add_argument('--city', type=str, default='Amsterdam',
                        help='City name of which venues need to be pulled. Note that bbox needs to be set correctly for this as well.')

    config = parser.parse_args()
    main()