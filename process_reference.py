import codeLib
import xarray
import numpy as np
import matplotlib.pyplot as plt
import dask

use_cache= True # set false if want to regenerate cached data. Otherwise will just read in data.

def proc_eucleia(root_direct,var,output_file_root,rolling=None,xc='lon',yc='lat',test=False):

    output_file=codeLib.output_dir/f'{output_file_root}_{var}_roll{rolling}.nc'
    if use_cache &  output_file.exists(): # using cache and output file exists. So read it and return
        ts = xarray.load_dataset(output_file)
        return ts

    # now to process
        

    ts_lst=[]
    dirs = list(root_direct.glob('r*'))
    if test:
        dirs= [dirs[0]]  # for testing

    for indx,ensemble in enumerate(dirs):
        direct=ensemble/'latest'
        print(direct," ",var)
        files=list(direct.glob(f'{var}_*.nc'))
        ts=proc_files(files, var,rolling=rolling,xc=xc,yc=yc)
        print("="*60)
        ts = ts.assign_coords(realization=indx)
        ts_lst.append(ts)

    ts=xarray.concat(ts_lst,dim='realization',combine_attrs='override')
    if not test: # no caching when testing.
        ts.to_netcdf(output_file) # write data out
    return ts

def proc_extension(root_direct,var,output_file_root,rolling=None,xc='longitude',yc='latitude',test=False):

    output_file=codeLib.output_dir/f'{output_file_root}_{var}_roll{rolling}.nc'
    if use_cache &  output_file.exists(): # using cache and output file exists. So read it and return
        ts = xarray.load_dataset(output_file)
        return ts

    # now to process
        

    ts_lst=[]
    for realization in range(0,105):
        for physics in range(0,5):
            pattern = f'{var}_day_HadGEM3-A-N216_*_r{realization+1:03d}i1p{physics+1:1d}_*.nc'
            output_file = codeLib.output_dir/f'{output_file_root}_{var}{rolling}_r{realization+1:03d}i1p{physics+1:1d}.nc' # 
            files = list((root_direct/var/'day').glob(pattern))
            if len(files) == 0:
                print("No files found for ",pattern)
                continue 
            print(pattern,len(files))
            try:
                ts=proc_files(files, var,rolling=rolling,xc=xc,yc=yc,outputFile=output_file)
                print("="*60)
                ts = ts.assign_coords(realization=(realization+physics*105))
                ts_lst.append(ts)
            except OSError:
                print("Some problem in ",files)
            if test:
                break # stop processing after first group.
        if test:
            break

    ts=xarray.concat(ts_lst,dim='realization',combine_attrs='override')
    if not test: # no caching when testing.
        ts.to_netcdf(output_file) # write data out
    return ts

def proc_files(files, var,rolling=None,xc='lon',yc='lat',outputFile=None):

    if use_cache and  (outputFile is not None) and outputFile.exists(): # using cache
        ts_mx= xarray.load_dataset(outputfile)
        return ts_mx
    
    chunks={'time':3600,xc:100,yc:100}
    ds = xarray.open_mfdataset(files,combine='nested',chunks=chunks).sortby('time')
    months = [6,7] # JJ only.
    timeSel=ds.time.dt.month.isin(months) 
    if rolling is not None:
        month_sel=[months[0]-1]
        month_sel.extend(months)
        month_sel.append(months[-1]+1)
        timeSel=ds.time.dt.month.isin(month_sel) # extended 
        if rolling > 30: # rolling too large will need more months just compain
            raise NotImplementedError("rolling > 30")
                 
    with dask.config.set(**{'array.slicing.split_large_chunks':True}):
        da=ds[var].sel(time=timeSel).sel({yc:codeLib.region['lat']})
    L= (da[xc] >= (360+codeLib.region['lon'].start) ) | (da[xc] < codeLib.region['lon'].stop)
    da = da.where(L,drop=True)
    print(da.shape)
    wt=np.cos(np.deg2rad(da[yc]))
    ts=da.load().weighted(wt).mean([xc,yc])
    if rolling is not None:
        ts = ts.rolling(time=rolling).mean().rename(f"{var}{rolling}")

    timeSel=(ts.time.dt.month.isin(months)) 
    ts = ts.sel(time=timeSel)
        
    ts_mx = codeLib.max_process(ts)
    timeSel=(ts_mx.time.dt.season == 'JJA') # summer only.
    ts_mx=ts_mx.sel(time=timeSel)
    ts_mx.attrs=ds.attrs
    if outputFile is not None: # write file to cache
        ts_mx.to_netcdf(outputFile)
    return ts_mx

# process the data


results_ref=dict()
results_hist105=dict()
results_nat105=dict()
results_hist525=dict()
results_nat525=dict()
rolling_var=dict(tas=2)
for var in ['tas','tasmax']:
    rolling=rolling_var.get(var)
    ts=proc_eucleia(codeLib.reference_dir,var,'reference_ens',rolling=rolling,test=False)
    results_ref[var]=ts
    # process the  105 member ensemble.
    ts=proc_eucleia(codeLib.hist105_dir,var,'hist105_ens',rolling=rolling,test=False)
    results_hist105[var]=ts
    ts=proc_eucleia(codeLib.nat105_dir,var,'nat105_ens',rolling=rolling,test=False)
    results_nat105[var]=ts
    # process the 525 member ensembles
    ts=proc_extension(codeLib.hist525_dir,var,'hist525_ens',rolling=rolling,test=False)
    results_hist525[var]=ts
    ts=proc_extension(codeLib.nat525_dir,var,'nat525_ens',rolling=rolling,test=False)
    results_nat525[var]=ts

ref_mx=xarray.merge(results_ref.values())
hist105=xarray.merge(results_hist105.values())
nat105=xarray.merge(results_nat105.values())
hist525=xarray.merge(results_hist525.values())
nat525=xarray.merge(results_nat525.values())




