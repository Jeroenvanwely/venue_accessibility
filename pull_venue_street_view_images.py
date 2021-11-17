import requests
import csv
import argparse
import shutil
import googlemaps
import os
import numpy as np
import math

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_google_images(name, city, address, venue_id, place_id):

    # Request Google images information of venues
    photos_url = "https://maps.googleapis.com/maps/api/place/details/json?place_id="+place_id+"&key="+config.api_key
    photos_response = requests.request("GET", photos_url, headers={}, data={})

    if photos_response.status_code == 200:
        for i, photo in enumerate(photos_response.json()['result']['photos']):
            
            # Request Google image of venue
            photo_reference = photo['photo_reference'] # Get photo reference to place in photo_url
            photo_url = "https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photo_reference="+photo_reference+"&key="+config.api_key
            photo_response = requests.get(url=photo_url, stream=True)

            # Define path to place image
            file_path = 'venue_images/{}_{}_{}/image{}_{}_{}_{}.jpg'.format(name.replace(" ", "_"), 
                                                                            address.replace("+", "_"), 
                                                                            venue_id, i, name.replace(" ", "_"), 
                                                                            address.replace("+", "_"), venue_id)
            # Place image in directory
            if photo_response.status_code == 200:
                with open(file_path, 'wb') as f:
                    photo_response.raw.decode_content = True
                    shutil.copyfileobj(photo_response.raw, f)

def get_streetview_image(name, address, venue_id, api_key, google_images):

    # Request Google Street View image of venue
    url = "https://maps.googleapis.com/maps/api/streetview?size=640x640&location="+address+"&key="+config.api_key+"&fov=120"
    street_view_response = requests.get(url=url, stream=True)

    # Define path to place Google Street View image
    if google_images:
        file_path = 'venue_images/{}_{}_{}/street_view_{}_{}_{}.jpg'.format(name.replace(" ", "_"), 
                                                                            address.replace("+", "_"), venue_id, 
                                                                            name.replace(" ", "_"), 
                                                                            address.replace("+", "_"), 
                                                                            venue_id)
    else:
        file_path = 'venue_images/street_view_{}_{}_{}.jpg'.format(name.replace(" ", "_"), 
                                                                    address.replace("+", "_"), 
                                                                    venue_id)
    # Place Google street view image in directory
    if street_view_response.status_code == 200:
        with open(file_path, 'wb') as f:
            street_view_response.raw.decode_content = True
            shutil.copyfileobj(street_view_response.raw, f)

def main():

    if config.google_images or config.streetview:
        gmaps = googlemaps.Client(key=config.api_key) # Set Google API key

        # Create directory for downloading venue Google images and Street View image
        if not os.path.exists('venue_images/'):
            os.makedirs('venue_images/')

        with open(config.venues_csv, newline='\n') as csvfile: # Open csv file containing venues

            venue_reader = csv.reader(csvfile, delimiter=',') # Define reader

            for venue in venue_reader: # Loop over venues

                # Get essential venue information
                name, venue_id, street, number, city, postal_code = venue[0], venue[7], venue[9], venue[10], venue[11], venue[12]
                # Get address to place in streetview_url
                address = (street + " " + number + " " + city + " " + postal_code).replace(" ", "+")
                
                geocode_result = gmaps.geocode(name + ', ' + city) # Get Google maps API geocode information of venue
                place_id = geocode_result[0]['place_id'] # Get place ID for photos_url
                
                if config.google_images:

                    # Create a directory to place venue images
                    path = 'venue_images/{}_{}_{}'.format(name.replace(" ", "_"), address.replace("+", "_"), venue_id)
                    if not os.path.exists(path):
                        os.makedirs(path)

                    get_google_images(name, city, address, venue_id, place_id)

                if config.streetview:
                    get_streetview_image(name, address, venue_id, config.api_key, config.google_images)
    else:
        print("No images have been extracted. Either google_images or streetview must be True.")

if __name__ == '__main__':
    
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', type=str,
                        help='Wheelmap API key to get acces.')
    parser.add_argument('--venues_csv', type=str, default='venues.csv',
                        help='name of csv file containing venues pulled from wheelmap')
    parser.add_argument('--google_images', type=str2bool, default=False,
                        help='If google_images is true, the Google images of venues in venues.csv will be extracted.')
    parser.add_argument('--streetview', type=str2bool, default=True,
                        help='If streetview is true, the streetview image of venues in venues.csv will be extracted.')

    config = parser.parse_args()
    main()