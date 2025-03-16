import xarray as xr
import numpy as np
import pandas as pd

def extract_single_segment(
        ds: xr.Dataset, 
        start_time: pd.Timestamp, 
        end_time: pd.Timestamp, 
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
        label: int) -> xr.Dataset:

    geomag = geomag.isel(vector=station_idx)
    segments = [geomag.sel(time=slice(start, end)).expand_dims(sample=[f"station: {station_idx}, label: {label}, idx: {i}"]) 
                for i, (start, end) in enumerate(sample_ranges)]
    
    # Concatenate along the new "sample" dimension
    ds = xr.concat(segments, dim="sample")

    if ds is not None:
        # ds= ds.assign_coords(sample=("sample", np.arange(len(sample_ranges))))
        ds = ds.assign({"labels": ("sample", np.full(len(sample_ranges), label, dtype=int))})

    return ds

def build_dataset(
        dataset_ranges_by_station: dict,
        geomag: xr.Dataset
        ) -> xr.Dataset:
    
    station_ids = geomag["id"].isel(time=pd.Timestamp("2024-01-01 00:00")).values
    ds = []
    
    for i in range(len(station_ids)):
        station_dict = dataset_ranges_by_station[id]

        pos_sample_ranges = station_dict["positive"]
        neg_sample_ranges = station_dict["negative"]

        pos_ds = extract_multiple_segments(sample_ranges=pos_sample_ranges, geomag=geomag, label=1, station_idx=i)
        neg_ds = extract_multiple_segments(sample_ranges=neg_sample_ranges, geomag=geomag, label=0, station_idx=i)

        if pos_ds is not None and neg_ds is not None:
            combined_ds = xr.concat([pos_ds, neg_ds], dim="sample")
        elif pos_ds is not None:
            combined_ds = pos_ds
        elif neg_ds is not None:
            combined_ds = neg_ds
        else:
            raise ValueError("No valid time ranges provided.")

        ds.append(combined_ds)
    
    ds = xr.concat(ds, dim="sample")

    return ds
