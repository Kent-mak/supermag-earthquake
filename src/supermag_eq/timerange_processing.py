import xarray as xr
import pandas as pd
from haversine import haversine_vector, Unit
import numpy as np
import os
from .station_utils import normalize_coordinates, get_station_coords


def get_sample_ranges(
        catalog: pd.DataFrame,
        geomag: xr.Dataset, 
        time_period: np.timedelta64, 
        year: int, 
        radius: float=200.0) -> tuple[dict, int, int]:

    # needs function that extracts the coordinates for each station
    station_coords = get_station_coords(geomag)
    dataset_ranges = {}
    # iterate through each station
    id = 0
    total_pos = 0
    total_neg = 0
    for station_coord in station_coords:
        
        # get earthquake periods for station
        eq_periods = get_eq_periods_for_station(
            station_coord=station_coord, 
            catalog=catalog, 
            time_period=time_period, 
            radius=radius)
        # get normal periods for station
        normal_periods = get_normal_periods_for_station(
            eq_periods=eq_periods,
            time_period=time_period, 
            year=year)
        
        # print(f"got: {eq_periods[0]}")
        total_pos += eq_periods.shape[0]
        total_neg += normal_periods.shape[0]

        station_dict = {}
        station_dict["negative"] = normal_periods
        station_dict["positive"] = eq_periods

        # print(station_dict)

        dataset_ranges[id] = station_dict
        id += 1

    # print(f"Dataset : {len(dataset_ranges)}")
    # print(f"total positive samples: {total_pos}")
    # print(f"total negative samples: {total_neg}")


    return dataset_ranges, total_pos, total_neg



def get_eq_periods_for_station(
        station_coord: np.ndarray, 
        catalog: pd.DataFrame, 
        time_period: np.timedelta64, 
        radius: float) -> np.ndarray:
    
    station_coord = normalize_coordinates(station_coord.reshape(1, 2))[0]

    coords = np.stack((catalog["latitude"].values, catalog["longitude"].values), axis=1)
    # print(coords)
    coords = normalize_coordinates(coords)
    
    distances = haversine_vector(coords, station_coord, unit=Unit.KILOMETERS, comb=True )[0]  # Fast haversine distance
    # print(distances)
    mask = distances <= radius
    # print(mask)
    filtered_eqs = catalog.loc[mask]
    filtered_eqs = np.array(filtered_eqs["time"], dtype="datetime64[m]")
    # print("here?")
    # filtered_eqs= filtered_eqs.astype("datetime64[m]") 
    # print("here?")
    # Compute time windows efficiently
    eq_periods = np.column_stack([
        filtered_eqs - time_period,  # Start times
        filtered_eqs  # End times
    ])

    return eq_periods


def get_normal_periods_for_station(
        year: int,
        eq_periods: np.ndarray,
        time_period: np.timedelta64,
        ) -> np.ndarray:
    # if eq_periods.shape[0] > 0:
    #     print(f"eq: {eq_periods[0][0]}")
    start_time = np.datetime64(f"{year}-01-01T00:00:00")
    end_of_year = np.datetime64(f"{year+1}-01-01T00:00:00")

    normal_periods = []
    
    eq_index = 0
    end_time = start_time + time_period

    while end_time < end_of_year:
        if eq_index < len(eq_periods) and end_time <= eq_periods[eq_index][0]:  
            # No overlap, normal period
            normal_periods.append([start_time, end_time])
            start_time = end_time 
            end_time = start_time + time_period
            # start_time = start_time + np.timedelta64(1, "m")
        elif eq_index < len(eq_periods):
            # Overlap detected, move start_time to after the earthquake period
            start_time = eq_periods[eq_index][1] 
            end_time = start_time + time_period
            # start_time = start_time + np.timedelta64(1, "m")
            eq_index += 1  # Move to the next earthquake period
        else:
            # No more earthquakes, fill in remaining normal periods
            normal_periods.append([start_time, end_time])
            start_time = end_time
            end_time = start_time + time_period  
            # start_time = start_time + np.timedelta64(1, "m")

    return np.array(normal_periods, dtype=object)



if __name__=="__main__":
    catalog = r"D:\earthquake-prediction\data\catalog\csv\usgs\2024.csv"
    geomag = r"D:\earthquake-prediction\data\geomagnetic_data\supermag\all_stations_all2024.netcdf"

    ds_ranges, total_pos, total_neg = get_sample_ranges(catalog, geomag, time_period=pd.Timedelta(days=7), year=2024)