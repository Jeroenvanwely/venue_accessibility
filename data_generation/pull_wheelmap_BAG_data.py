import requests
from tqdm import tqdm
import csv
import argparse
from rijksdriehoek import rijksdriehoek


rd = rijksdriehoek.Rijksdriehoek() 
print("Original coordinates in WGS'84: {},{}".format(str(52.3761973), str(4.8936216))) 
rd.from_wgs(52.3761973, 4.8936216) 
print("Rijksdriehoek: {},{}".format(str(rd.rd_x), str(rd.rd_y))) 
# lat, lon = rd.to_wgs() 
# print("WGS'84 coordinates converted from RD: {},{}".format(str(lat), str(lon))) 

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

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
    with open('venues.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(venues)

# Function to filter venues
def filter_venues(venues, city, remove_unlabeled, add_bag_data, add_rd_coordinates):

    if add_bag_data: print("Adding BAG data.")
    filtered_venues = []
    
    final_venues = []
    for venue in tqdm(venues):
        if remove_unlabeled and venue['wheelchair'] == 'unknown': continue
        elif city != None and venue['city'] != city: continue
        elif add_bag_data:
            
            street = venue['street']
            housenumber = venue['housenumber']
            if str(type(street)) == "<class 'NoneType'>" or str(type(housenumber)) == "<class 'NoneType'>": 
                continue
            response = requests.get(url = "https://api.data.amsterdam.nl/atlas/search/adres/?q="+street+housenumber)
            
            if response.status_code != 200: continue
            results = response.json()['results']
            
            if len(results) == 0: continue
            response = requests.get(url = results[0]['_links']['self']['href'])
            
            if response.status_code != 200: continue
            response = response.json()
            
            bbox = response['bbox']
            coordinates = response['geometrie']['coordinates']

            venue['coordinates'] = coordinates
            venue['bbox'] = bbox
        
        if add_rd_coordinates:
            lat, lon = float(venue['lat']), float(venue['lon'])
            rd = rijksdriehoek.Rijksdriehoek()
            rd.from_wgs(lat, lon)
            venue['RD_x'], venue['RD_y'] = str(rd.rd_x), str(rd.rd_y)
            venue['tile_code'] = str(int(rd.rd_x/50))+"_"+str(int(rd.rd_y/50))
        
        filtered_venues.append(venue)

    return filtered_venues

# Function to get data and store in list
def get_data(api_key, per_page, limit, bbox):
    
    response = requests.get(url = "https://wheelmap.org/api/categories/2/nodes/?api_key="+api_key+"&per_page="+per_page+"&bbox="+bbox+"&limit="+limit+"&page=1")

    venues = response.json()['nodes'] # Initialize venues list
    num_pages = response.json()['meta']['num_pages'] # Get number of pages

    for page in tqdm(range(2, num_pages)): # For every page

        response = requests.get(url = "https://wheelmap.org/api/categories/2/nodes/?api_key="+api_key+"&per_page="+per_page+"&bbox="+bbox+"&limit="+limit+"&page="+str(page))
        venues += response.json()['nodes'] # Concatenate to venues list

    return filter_venues(venues, config.city, config.remove_unlabeled, config.add_bag_data, config.add_rd_coordinates) # Return filtered venues

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
                        help='Limit per page. Maximum and default is 500. If you want to get all venues in bbox make sure per_page and limit is equal.')
    parser.add_argument('--bbox', type=str, default='4.839323, 52.333451, 4.968438, 52.418032',
                        help='Lon and lat values define a bounding box out of which to get data. Default is bbox of Amsterdam.')
    parser.add_argument('--write_to', type=str, default='csv',
                        help='File type to save data in. Can be txt or csv.')
    parser.add_argument('--city', type=str, default='Amsterdam',
                        help='City name of which venues need to be pulled. Note that bbox needs to be set correctly for this as well.')
    parser.add_argument('--remove_unlabeled', type=str2bool, default=False,
                        help='If remove_unlabeled is true, only labeled venues will be returned.')
    parser.add_argument('--add_bag_data', type=str2bool, default=True,
                    help='If add_bag_data is true, add bag data.')
    parser.add_argument('--add_rd_coordinates', type=str2bool, default=True,
                    help='If add_rd_coordinates is true, add RD coordinates.')

    config = parser.parse_args()
    main()
# 4.923933, 52.360877, 4.927080, 52.360029 small