import xarray as xr
import numpy as np
import pandas as pd
from tqdm import tqdm
import cupy as cp
from dask import delayed, compute

def extract_single_segment(
        ds: xr.Dataset, 
        start_time: np.datetime64, 
        end_time: np.datetime64, 
        vars: list[str], 
        station_id:int) -> xr.DataArray:
    
    selected_vars = ds[vars]

    selected_data = selected_vars.sel(time=slice(start_time, end_time))

    segment = selected_data.sel(vector=station_id)
    segment = segment.dropna(dim="time")

    return segment



def extract_multiple_segments(
        sample_ranges: np.ndarray,
        geomag: xr.Dataset,
        station_idx: int,
        vars: list,
        label: int) -> xr.Dataset:
    
    geomag = geomag
    geomag = geomag[vars].isel(vector=station_idx)

    @delayed
    def slice_geomag(start, end):
        return geomag.sel(time=slice(start, end))

    # print("slicing")
    tasks = [slice_geomag(start + np.timedelta64(1, "m"), end) for start, end in sample_ranges]
    # print("1")
    segments = compute(*tasks)
    # print(f"length: {len(segments)}")
    # print("2")
    valid_segments = [seg for seg in segments if seg.sizes["time"] == 10080]
    removed_count = len(segments) - len(valid_segments)
    if removed_count > 0:
        print(f"Discarded {removed_count} segments that didn't have time=10080.")

    ds = None

    if len(valid_segments) > 0:
        ds = xr.concat(valid_segments, dim="sample", join="override")
    # print("3")

    if ds is not None:
        # ds= ds.assign_coords(sample=("sample", np.arange(len(sample_ranges))))
        ds = ds.assign({"labels": ("sample", np.full(len(valid_segments), label, dtype=int))})
        ds = ds.drop_vars("time")

    
    return ds

def build_dataset(
        dataset_ranges_by_station: dict,
        geomag: xr.Dataset,
        vars:list
        ) -> xr.Dataset:
    
    station_ids = geomag["id"].isel(time=0).values
    ds = []
    
    for i in tqdm(range(len(station_ids))):
        station_dict = dataset_ranges_by_station[i]

        pos_sample_ranges = station_dict["positive"]
        neg_sample_ranges = station_dict["negative"]

        if pos_sample_ranges.shape[0] == 0:
            continue

        pos_ds = extract_multiple_segments(sample_ranges=pos_sample_ranges, geomag=geomag, label=1, station_idx=i, vars=vars)
        neg_ds = extract_multiple_segments(sample_ranges=neg_sample_ranges, geomag=geomag, label=0, station_idx=i, vars=vars)

        if pos_ds is not None and neg_ds is not None:
            # print(f"pos: \n{pos_ds}\n")
            # print(f"neg: \n{neg_ds}\n")
            combined_ds = xr.concat([pos_ds, neg_ds], dim="sample", join="override")
            # print(f"station {i}: \n{combined_ds.isel(sample=0)}\n")
        # elif pos_ds is not None:
        #     combined_ds = pos_ds
        # elif neg_ds is not None:
        #     combined_ds = neg_ds
        else:
            continue

        ds.append(combined_ds)
    
    ds = xr.concat(ds, dim="sample", join="override")

    print(f"final: \n{ds}")
    unique_values, counts = np.unique(ds["labels"].values, return_counts=True)
    print(dict(zip(unique_values, counts)))

    return ds
