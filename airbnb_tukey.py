#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import gzip
import numpy as np
import pandas as pd
from xml.dom import minidom
import math
import sys
from math import cos, asin, sqrt, radians, sin
import gmplot 

from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None


#locations
vancouver_latlon = [49.254169, -123.135977]
northvan_latlon = [49.338687, -123.101998]
burnaby_latlon = [49.240465, -122.968028]
richmond_latlon = [49.166662, -123.115976]

#add whichever amenities wanted
amenities_list = ['cafe', 'pub', 'bar', 'ice_cream', 'park', 'library']


#calc distance between amenity and location  
def dist(row, location):
    lat1 = location[0]
    lon1 = location[1]
    lat2 = float(row['lat'])
    lon2 = float(row['lon'])
    p = math.pi/180
    a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p) * cos(lat2*p) * (1- cos((lon2-lon1)*p))/2
    return 12742*asin(sqrt(a))*1000

#parse lat,lon string to retrieve lat
def getlat(x):
    split = x.split(',')
    if split[0] != None:
        return split[0]
    return 0

#parse lat,lon string to retrieve lon
def getlon(x):
    split = x.split(',')
    if split[1] != None:
        return split[1]
    return 0
    

# Single Point Haversine ------------------------------------------------------
# https://datascience.stackexchange.com/questions/49553/combining-latitude-longitude-position-into-single-feature

def single_pt_haversine(lat, lng, degrees=True):
    """
    'Single-point' Haversine: Calculates the great circle distance
    between a point on Earth and the (0, 0) lat-long coordinate
    """
    r = 6371 # Earth's radius (km). Have r = 3956 if you want miles

    # Convert decimal degrees to radians
    if degrees:
        lat, lng = map(radians, [lat, lng])

    # 'Single-point' Haversine formula
    a = sin(lat/2)**2 + cos(lat) * sin(lng/2)**2
    d = 2 * r * asin(sqrt(a)) 

    return d


if __name__ == '__main__':

    location = sys.argv[1]

    if location == 'vancouver':
        location = vancouver_latlon
    elif location == 'northvan':
        location = northvan_latlon
    elif location == 'burnaby':
        location = burnaby_latlon
    elif location == 'richmond':
        location = richmond_latlon
    else:
        print("Enter a City: vancouver, northvan, burnaby, or richmond")
        exit()


    json = gzip.open('amenities-vancouver.json.gz', 'rt', encoding='utf-8')
    amenities_df = pd.read_json(json, lines=True)
    amenities_df = amenities_df.drop(columns=['timestamp'])

    # open parks.csv found from vity of vancouver datasets
    # https://data.vancouver.ca/datacatalogue/index.htm
    # clean parks data and match columns to data dataframe to append
    parks_df = pd.read_csv('vandatasets/parks.csv')
    parks_df = parks_df[['Name','GoogleMapDest']]
    parks_df['lat'] = parks_df['GoogleMapDest'].apply(getlat)
    parks_df['lat'] = pd.to_numeric(parks_df['lat'])
    parks_df['lon'] = parks_df['GoogleMapDest'].apply(getlon)
    parks_df['lon'] = pd.to_numeric(parks_df['lon'])
    parks_df['amenity'] = 'park'
    parks_df['tags'] = '{}'
    parks_df['name'] = parks_df['Name']
    parks_df = parks_df.drop(columns=['Name', 'GoogleMapDest'])

    # append parks info to data 
    amenities_df = amenities_df.append(parks_df, ignore_index = True, sort=True) 

    
    amenities_df1 = amenities_df[['amenity','lat', 'lon']].copy()
    amenities_df1 = amenities_df1.sort_values(by=['amenity'])
#     print(amenities_df1)
    
    
################### 
    # columns for dataframe containing avg lat and lon of each amenity in range of location
    columns = ['amenity', 'lat', 'lon']
    rows = []

    
    # for each amenity type filter by distance and find avg
    for i in amenities_list:
        amenity_df = amenities_df[amenities_df.amenity == i]
        amenity_df.reset_index(drop=True)
        amenity_df['dist'] = amenity_df.apply(lambda x: dist(x, location), axis=1)
        amenity_df = amenity_df[amenity_df['dist'] < 4000]
        # average lat lon of the current amenity type using locations within range
        avg_lat = amenity_df['lat'].sum() / amenity_df.count()
        avg_lon = amenity_df['lon'].sum() / amenity_df.count()
        row = [i, avg_lat.lat, avg_lon.lon]
        rows.append(row)

    # create df
    amenities_latlon = pd.DataFrame(rows, columns = columns)      
    print(amenities_latlon)
    # calculate average of all averages of each amenity
    airbnb_lat = amenities_latlon['lat'].sum() / len(amenities_list)
    airbnb_lon = amenities_latlon['lon'].sum() / len(amenities_list)
#####################

    # Post Hoc Tukey Analysis ---------------------------------------------------------
    amenities_df1['hav_dist'] = [single_pt_haversine(lat, lon) for lat, lon in zip(amenities_df1.lat, amenities_df1.lon)]
    print("\nhav_dist added:\n", amenities_df1)
    
    # create haversine df
    hav_dist = amenities_df1[['amenity','hav_dist']].copy()
    
    posthoc = pairwise_tukeyhsd(hav_dist['hav_dist'], hav_dist['amenity'], alpha=0.05)
    print("\nPost Hoc:\n", posthoc)
    fig = posthoc.plot_simultaneous()
    plt.show()

    # ---------------------------------------------------------------------------------

    # plot optimum airbnb/hotel location
    gmap = gmplot.GoogleMapPlotter(location[0], location[1], 13) 
    gmap.marker(airbnb_lat, airbnb_lon,'cornflowerblue', title = 'Air BnB')

    # personal api key
    gmap.apikey = "AIzaSyAQmtpvowY8lopKJQ2fJQf5YWzlh6NFeVo"
    gmap.draw( "airbnb_map.html" ) 


