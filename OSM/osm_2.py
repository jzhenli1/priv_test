import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np

from shapely.geometry import Point, LineString, Polygon
from scipy.spatial import cKDTree

import re
from shapely import wkt

# Function to extract geo data from OSM
def geodata_to_df(country, city):

    G = ox.graph_from_place(city, network_type='bike')  # download raw geospatial data from OSM

    nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
    nodes["city"], edges["city"] = city, city
    nodes["country"], edges["country"] = country, country

    edges["lat_long"] = edges["geometry"].apply(lambda x: re.sub(r'[^0-9., ]', "", str([re.sub(r'[^0-9. ]', '', str(i)) for i in list(zip(x.xy[1], x.xy[0]))])))
    #edges["geometry"] = edges["geometry"].apply(lambda x: wkt.dumps(x))

    edges["highway"] = edges["highway"].apply(lambda x: ", ".join(x) if x.__class__.__name__=="list" else x)
    edges["name"] = edges["name"].apply(lambda x: ", ".join(x) if x.__class__.__name__=="list" else x)
    edges["maxspeed"] = edges["maxspeed"].apply(lambda x: ", ".join(x) if x.__class__.__name__ == "list" else x)
    edges["ref"] = edges["ref"].apply(lambda x: ", ".join(x) if x.__class__.__name__ == "list" else x)
    edges["reversed"] = edges["reversed"].apply(lambda x: x[0] if x.__class__.__name__ == "list" else x)
    edges["oneway"] = edges["oneway"].apply(lambda x: x[0] if x.__class__.__name__ == "list" else x)

    edges.fillna(-99, inplace=True)
    nodes.fillna(-99, inplace=True)
    edges["name"] = edges["name"].astype(str).replace("-99", None)

    # nodes_and_edges = gpd.sjoin(edges, nodes, how="left", predicate="intersects")

    return G, nodes, edges

#G, nodes, edges = geodata_to_df('Germany', 'Stuttgart')

G = ox.graph_from_place('Stuttgart', network_type='bike')
nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)

# Reset the index of edges --> this is the gdf we are going to work with
edges_reset = edges.reset_index()
edges_reset['index'] = range(len(edges_reset))

# Function to extract features
def osm_features(city):
    # Get nodes with the highway=traffic_signals tag (intersections with traffic lights)
    traffic_nodes = ox.features_from_place(city, tags={"highway": "traffic_signals"}).reset_index(
        )[['highway', 'geometry']].rename(columns={'highway': 'trafficSignals'})

    # Get spots with bicycle parking
    bicycle_parking = ox.features_from_place(city, tags={"amenity": "bicycle_parking"}).reset_index(
        )[['amenity', 'geometry']].rename(columns={'amenity': 'bicycleParking'})

    # Public transit options
    # Get tram stops
    transit_tram = ox.features_from_place(city, tags={"railway": 'tram_stop'}).reset_index(
        )[['railway', 'geometry']].rename(columns={'railway': 'tramStop'})
    # Get bus stops
    transit_bus = ox.features_from_place(city, tags={"highway": 'bus_stop'}).reset_index(
        )[['highway', 'geometry']].rename(columns={'highway': 'busStop'})

    # Get lighting
    lighting = ox.features_from_place(city, tags={'highway': 'street_lamp'}).reset_index(
        )[['highway', 'geometry']].rename(columns={'highway': 'lighting'})
    
    # On street parking
    street_parking_right = ox.features_from_place(city, tags={"parking:right": True}).reset_index(
        )[['geometry','parking:right']]
    street_parking_left = ox.features_from_place(city, tags={"parking:left": True}).reset_index(
        )[['geometry','parking:left']]
    street_parking_both = ox.features_from_place(city, tags={"parking:both": True}).reset_index(
        )[['geometry','parking:both']]
    
    return traffic_nodes, bicycle_parking, transit_tram, transit_bus, lighting, street_parking_right, street_parking_left, street_parking_both


traffic_nodes, bicycle_parking, transit_tram, transit_bus, lighting, street_parking_right, street_parking_left, street_parking_both = osm_features('Stuttgart')


# Nearest Neighbor Analysis
# Convert linestring geometries from edges gdf to a KDTree
coords = np.array([(line.xy[0][0], line.xy[1][0]) for line in edges.geometry])
tree = cKDTree(coords)

# Function to find the nearest edge to a point of the extracted features
def nearest_edges_point(node): 
    # Extracting coordinates from the points
    points_coordinates = node.geometry.apply(lambda point: [point.x, point.y]).to_list()
    # Converting to a NumPy array
    points_array = np.array(points_coordinates)
    # Perform the query
    distances, idx = tree.query(points_array)

    return idx

# Function add the nearest edge index to the extracted features and perform the merge
def merge_nearest_edges(node, edges_reset=edges_reset):
    # Apply the conversions based on geometry type
    node['geometry'] = node['geometry'].apply(lambda geom: geom.interpolate(0.5, normalized=True) if isinstance(geom, LineString) else geom)
    node['geometry'] = node['geometry'].apply(lambda geom: geom.centroid if isinstance(geom, Polygon) else geom)

    # Add the nearest line index to the points GeoDataFrame
    node['nearest_idx'] = nearest_edges_point(node)
    node = node.drop('geometry', axis=1)
    # Use drop_duplicates to keep only the first occurrence of each unique value in 'nearest_idx'
    node = node.drop_duplicates(subset='nearest_idx')

    # Now perform the merge
    edges_reset = edges_reset.merge(node, right_on='nearest_idx', left_on='index', how='left').drop('nearest_idx', axis=1)

    return edges_reset

# Loop through the features and perform the merge
features = [traffic_nodes, bicycle_parking, transit_tram, transit_bus, lighting, 
            street_parking_right, street_parking_left, street_parking_both]

for feature in features:
    edges_reset = merge_nearest_edges(feature, edges_reset=edges_reset)
