import codeLib
import xarray
import numpy as np
import matplotlib.pyplot as plt
import dask

use_cache= True # set false if want to regenerate cached data. Otherwise will just read in data.

def proc_eucleia(root_direct,var,output_file_root,
                 rolling=None,xc='lon',yc='lat',test=False):

    output_file=codeLib.output_dir/f'{output_file_root}_{var}_roll{rolling}.nc'
    ts_output_file=codeLib.output_dir/f'{output_file_root}_{var}_roll{rolling}_ts.nc'
    if use_cache &  output_file.exists(): # using cache and output file exists. So read it and return
        mx_result = xarray.load_dataset(output_file)
        ts_result = xarray.load_dataset(ts_output_file)
        return mx_result,ts_result


    # now to process
        

    ts_lst=[]
    mx_lst=[]
    dirs = list(root_direct.glob('r*'))
    if test:
        dirs= [dirs[0]]  # for testing

    for indx,ensemble in enumerate(dirs):
        outputFile = codeLib.output_dir/f'{output_file_root}_{var}{rolling}_{ensemble.name}.nc' # 
        print(f"cache file is {outputFile}")
        direct=ensemble/'latest'
        print(direct," ",var)
        files=list(direct.glob(f'{var}_*.nc'))
        mx,ts=proc_files(files, var,rolling=rolling,xc=xc,yc=yc,outputFile=outputFile)
        print("="*60)
        for t,lst in zip([ts,mx],[ts_lst,mx_lst]):
            t = t.assign_coords(realization=indx)
            lst.append(t)


    result_ts=xarray.concat(ts_lst,dim='realization',combine_attrs='override')
    result_mx=xarray.concat(mx_lst,dim='realization',combine_attrs='override')
    if not test: # no caching when testing.
        result_mx.to_netcdf(output_file) # write data out
        result_ts.to_netcdf(ts_output_file) # write data out
    return result_mx,result_ts

def proc_extension(root_direct,var,output_file_root,
                   rolling=None,xc='longitude',yc='latitude',test=False):

    output_file=codeLib.output_dir/f'{output_file_root}_{var}_roll{rolling}.nc'
    ts_output_file=codeLib.output_dir/f'{output_file_root}_{var}_roll{rolling}_ts.nc'
    if use_cache &  output_file.exists(): # using cache and output file exists. So read it and return
        mx_result = xarray.load_dataset(output_file)
        ts_result = xarray.load_dataset(ts_output_file)
        return mx_result,ts_result

    # now to process
        

    mx_lst=[]
    ts_lst=[]
    for realization in range(0,105):
        for physics in range(0,5):
            pattern = f'{var}_day_HadGEM3-A-N216_*_r{realization+1:03d}i1p{physics+1:1d}_*.nc'
            outputFile = codeLib.output_dir/f'{output_file_root}_{var}{rolling}_r{realization+1:03d}i1p{physics+1:1d}.nc' # 
            files = list((root_direct/var/'day').glob(pattern))
            if len(files) == 0:
                print("No files found for ",pattern)
                continue 
            print(pattern,len(files))
            try:
                mx,ts=proc_files(files, var,rolling=rolling,xc=xc,yc=yc,outputFile=outputFile)
                print("="*60)
                for t,lst in zip([ts,mx],[ts_lst,mx_lst]):
                    t = t.assign_coords(realization=(realization+physics*105))
                    lst.append(t)
            except OSError:
                print("Some problem in ",files)
            if test:
                break # stop processing after first group.
        if test:
            break

    result_mx=xarray.concat(mx_lst,dim='realization',combine_attrs='override')
    result_ts=xarray.concat(ts_lst,dim='realization',combine_attrs='override')
    if not test: # no caching when testing.
        result_mx.to_netcdf(output_file) # write data out
        result_ts.to_netcdf(ts_output_file) # write data out
    return result_mx,result_ts

def proc_files(files, var,rolling=None,xc='lon',yc='lat',outputFile=None):
    
    if outputFile is not None:
        ts_outputFile = outputFile.parent/(outputFile.stem+"_ts.nc")
    else:
        ts_outputFile = None

    if use_cache and  (outputFile is not None) and outputFile.exists(): # using cache
        ts_mx= xarray.load_dataset(outputFile)
        if ts_outputFile.exists(): # have ts as well so get it.
            ts=xarray.load_dataset(ts_outputFile)
            return ts_mx,ts 
    
    chunks={'time':3600,xc:100,yc:100}
    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        try:
            ds = xarray.open_mfdataset(files,chunks=chunks)#.sortby('time')
        except ValueError: #some problem so use combine by_coords which is slow
            print("Combining nesting")
            ds = xarray.open_mfdataset(files,chunks=chunks,combine='nested')
        ds=ds.sortby('time') # sort by time.

    months = [6,7] # JJ only.
    timeSel=ds.time.dt.month.isin(months) 
    if rolling is not None:
        month_sel=[months[0]-1]
        month_sel.extend(months)
        month_sel.append(months[-1]+1)
        timeSel=ds.time.dt.month.isin(month_sel) # extended 
        if rolling > 30: # rolling too large will need more months just compain
            raise NotImplementedError("rolling > 30")
                 
    with dask.config.set(**{'array.slicing.split_large_chunks':False}):
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
    if outputFile is not None: # write files to cache
        ts_mx.to_netcdf(outputFile)
        ts.to_netcdf(ts_outputFile)
        
    return ts_mx,ts

# process the data


rolling_var=dict(tas=2)

test=False # If True run in test mode -- only one realization gets processed per case
results_ts=dict()
results_mx=dict()
for dir,name in zip(
        [codeLib.reference_dir,codeLib.hist105_dir,codeLib.nat105_dir,codeLib.hist525_dir,codeLib.nat525_dir],
        ['reference_ens','hist105_ens','nat105_ens','hist525_ens','nat525_ens']):
    
    lst_mx=[]
    lst_ts=[]
    for var in ['tas','tasmax']:
        rolling=rolling_var.get(var)
        print("Processing ",name," ",var,"rolling",rolling, end=' ')
        if dir.name.endswith('Ext'):
            print("Using proc_extension")
            mx,ts=proc_extension(dir,var,name,rolling=rolling,test=test)
        else:
            print("Using proc_eucleia")
            mx,ts=proc_eucleia(dir,var,name,rolling=rolling,test=test)
        lst_mx.append(mx)
        lst_ts.append(ts)
    # end processing var. Now merge into one dataSet the ts and mx info
    results_ts[name]=xarray.merge(lst_ts)
    results_mx[name]=xarray.merge(lst_mx)

# now write everthing out!
for key,ds in results_ts.items():
    file=codeLib.output_dir/("HadGEM-GA6-N216_"+key+'_all_ts.nc')
    ds.to_netcdf(file)

for key,ds in results_mx.items():
    file=codeLib.output_dir/("HadGEM-GA6-N216_"+key+'_all_mx.nc')
    ds.to_netcdf(file)

