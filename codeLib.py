# Standard stuff for code
import pathlib
import xarray
import numpy as np
region=dict(lat=slice(51.25,54.0),lon=slice(-3.5,0.5))
reference_dir = pathlib.Path('/badc/eucleia/data/EUCLEIA/output/MOHC/HadGEM3-A-N216/historical/day/atmos/day/') # EUCLEAI refernece data
hist105_dir = pathlib.Path('/badc/eucleia/data/EUCLEIA/output/MOHC/HadGEM3-A-N216/historicalShort/day/atmos/day/') # EUCLEAI 105 member hist data
nat105_dir = pathlib.Path('/badc/eucleia/data/EUCLEIA/output/MOHC/HadGEM3-A-N216/historicalNatShort/day/atmos/day/') # EUCLEAI 105 member nat data
hist525_dir=pathlib.Path('/gws/nopw/j04/cssp_china/wp1/HadGEM3-A-N216/historicalExt') # CSSP China 525 historical ensemble
nat525_dir=pathlib.Path('/gws/nopw/j04/cssp_china/wp1/HadGEM3-A-N216/historicalNatExt') # CSSP China 525 natural ensemble
output_dir = pathlib.Path('output')
output_dir.mkdir(exist_ok=True,parents=True) # make directory for output if neeed
def time_of_max(da, dim='time'):
    """

    Work out time of maxes using argmax to get index and then select the times.

    """
    bad = da.isnull().all(dim, keep_attrs=True)  # find out where *all* data null

    indx = da.argmax(dim=dim, skipna=False, keep_attrs=True)  # index of maxes
    result = da.time.isel({dim: indx})

    result = result.where(~bad)  # mask data where ALL is missing
    return result


def max_process(da,resample='QS-DEC',timeDim='time'):
    """
    Process a dataset of daily data
    :param da -- DataArray to process
    :param resample -- resampling period to use.
    :return **DataSet** containing max values, time of max & mean value 
    """

    name = da.name
    resamp = da.resample({timeDim:resample}, label='left',skipna=True)    # set up the resample
    mx = resamp.max(keep_attrs=True).rename(name + 'Max') # maximum
    mn= resamp.mean(keep_attrs=True).rename(name+'Mean') # mean
    mxTime = resamp.map(time_of_max,dim=timeDim).rename(name + 'MaxTime') # time of max
    ds = xarray.merge([mn, mx, mxTime])

    return ds

