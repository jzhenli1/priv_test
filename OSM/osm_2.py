import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np

from shapely.geometry import Point, LineString
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



# Nearest Neighbor Analysis
# Convert linestring geometries to a KDTree
coords = np.array([(line.xy[0][0], line.xy[1][0]) for line in edges.geometry])
tree = cKDTree(coords)

# Reset the index of edges if it's a multi-level index
edges_reset = edges.reset_index()
edges_reset['index'] = range(len(edges_reset))

def nearest_edges_point(node): 
    # Extracting coordinates from the points
    points_coordinates = node.geometry.apply(lambda point: [point.x, point.y]).to_list()
    # Converting to a NumPy array
    points_array = np.array(points_coordinates)
    # Perform the query
    distances, idx = tree.query(points_array)

    return idx




# Get nodes with the highway=traffic_signals tag (intersections with traffic lights)
traffic_nodes = ox.features_from_place('Stuttgart', tags={"highway": "traffic_signals"}).reset_index()[['highway', 'geometry']].rename(columns={'highway': 'trafficSignals'})

# Add the nearest line index to the points GeoDataFrame
traffic_nodes['nearest_idx'] = nearest_edges_point(traffic_nodes)
traffic_nodes = traffic_nodes.drop('geometry', axis=1)
# Use drop_duplicates to keep only the first occurrence of each unique value in 'nearest_idx'
traffic_nodes = traffic_nodes.drop_duplicates(subset='nearest_idx')

# Now perform the merge
edges_reset = edges_reset.merge(traffic_nodes, right_on='nearest_idx', left_on='index', how='left').drop('nearest_idx', axis=1)
