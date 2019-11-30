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

#add whichever amenities wanted
amenities_list = ['cafe', 'pub', 'bar', 'ice_cream', 'park']


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
	
	# calculate average of all averages of each amenity
	airbnb_lat = amenities_latlon['lat'].sum() / len(amenities_list)
	airbnb_lon = amenities_latlon['lon'].sum() / len(amenities_list)


	
	# plot optimum airbnb/hotel location
	gmap = gmplot.GoogleMapPlotter(location[0], location[1], 13) 
	gmap.marker(airbnb_lat, airbnb_lon, title = 'Air BnB')


	#latitude_list = [airbnb_lat]
	#longitude_list = [airbnb_lon]
	#gmap.scatter(latitude_list, longitude_list, '# FF0000', size = 40, marker = False) 


	# personal api key
	gmap.apikey = "AIzaSyAQmtpvowY8lopKJQ2fJQf5YWzlh6NFeVo"
	gmap.draw( "airbnb_map.html" ) 

	re = amenities_df[amenities_df.amenity == 'pub']

	plt.scatter(re['lat'], re['lon']) 
	plt.show()









	
    


