# divesite_profiles.py
#
# Program to produce divesite profiles for specified
#  dates, locations.
#
# Joseph B. Zambon
# jbzambon@ncsu.edu
# 25 October 2018

import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
np.set_printoptions(threshold=np.nan)
from numpy import nan
import mpl_toolkits.basemap
from mpl_toolkits.basemap import Basemap
from pydap.client import open_url
import datetime
import matplotlib.dates as mdates
from scipy.interpolate import griddata
from datetime import timezone

site='Eastern Cayman Islands'
# coords=[24.5,-80.4473]
coords=[19.4,-81.0]
tz=[-5,'EST']
xyrange=5      # Degrees of spread for x-y plots
start_date = datetime.datetime(2018,10,26, 0)
end_date = datetime.datetime(2018,10,28, 0)
qv_space=10

# For inline plotting
%pylab inline

roms_url='http://oceanus.meas.ncsu.edu:8080/thredds/dodsC/fmrc/useast_coawst_roms/COAWST-ROMS_SWAN_Forecast_Model_Run_Collection_best.ncd'
print(roms_url)
roms_dataset = open_url(roms_url)
wrf_url='http://oceanus.meas.ncsu.edu:8080/thredds/dodsC/fmrc/useast_coawst_wrf/COAWST-WRF_Forecast_Model_Run_Collection_best.ncd'
print(wrf_url)
wrf_dataset = open_url(wrf_url)

# Let's ingest the latitude and longitude values
r_lon=np.array(roms_dataset['lon_rho'])
r_lat=np.array(roms_dataset['lat_rho'])
w_lon=np.array(wrf_dataset['lon'])
w_lat=np.array(wrf_dataset['lat'])

# Ingest bathymetry
h=np.array(roms_dataset['h'])

def findxy_model(lon1,lat1,lon,lat):
    dis=np.sqrt(np.square(np.subtract(lon,lon1))+np.square(np.subtract(lat,lat1)))
    from numpy import unravel_index
    index=unravel_index(dis.argmin(), dis.shape)
    jj=index[0]
    ii=index[1]
    jj=np.int(jj)
    ii=np.int(ii)
    return jj,ii

# Create time array from which to choose closest time-point
time=np.array(roms_dataset['time']).astype(float)
from matplotlib.dates import date2num
reftime=datetime.datetime(2013,8,30,0,0,0)
reftime=date2num(reftime)
adj_time=np.zeros([len(time)])
for i in range(0,len(time)):
    adj_time[i]=reftime+time[i]/24

#Find ROMS time indices
roms_origin_date = datetime.datetime(2013,8,30,0,0,0)
roms_time=(np.array(roms_dataset['time'][:]))/24+datetime.date.toordinal(roms_origin_date)

roms_start_index = np.where(roms_time==datetime.date.toordinal(start_date))
roms_start_index = roms_start_index[0][0]
roms_start_index=np.int(roms_start_index)
# print(roms_start_index)

roms_end_index = np.where(roms_time==datetime.date.toordinal(end_date))
roms_end_index = roms_end_index[0][0]
roms_end_index=np.int(roms_end_index)
# print(roms_end_index)

#Find WRF time indices
wrf_origin_date = datetime.datetime(2016,9,20,0,0,0)
wrf_time=(np.array(wrf_dataset['time'][:]))/24+datetime.date.toordinal(wrf_origin_date)

wrf_start_index = np.where(wrf_time==datetime.date.toordinal(start_date))
wrf_start_index = wrf_start_index[0][0]
wrf_start_index=np.int(wrf_start_index)
# print(wrf_start_index)

wrf_end_index = np.where(wrf_time==datetime.date.toordinal(end_date))
wrf_end_index = wrf_end_index[0][0]
wrf_end_index=np.int(wrf_end_index)
# print(wrf_end_index)

# find indices of ROMS center point and X-Y range
r_jj,r_ii=findxy_model(coords[1],coords[0],r_lon,r_lat)
# print(r_jj,r_ii)
# print(h[r_jj,r_ii])
r_jj_s,r_ii_s=findxy_model(coords[1]-xyrange,coords[0]-xyrange,r_lon,r_lat)
r_jj_s = max(r_jj_s,0); r_ii_s = max(r_ii_s,0) 
r_jj_e,r_ii_e=findxy_model(coords[1]+xyrange,coords[0]+xyrange,r_lon,r_lat)
# print(r_jj_s,r_jj_e,r_ii_s,r_ii_e)
r_jj_e = min(r_jj_e,shape(r_lon)[0]); r_ii_e = min(r_ii_e,shape(r_lon)[1]);
# print(r_jj_s,r_jj_e,r_ii_s,r_ii_e)
# print(coords[1],xyrange,shape(r_lon)[0])

w_jj,w_ii=findxy_model(coords[1],coords[0],w_lon,w_lat)
# print(w_jj,w_ii)
# print(w_lon[w_jj,w_ii],w_lat[w_jj,w_ii])
w_jj_s,w_ii_s=findxy_model(coords[1]-xyrange,coords[0]-xyrange,w_lon,w_lat)
w_jj_s = max(w_jj_s,0); w_ii_s = max(w_ii_s,0) 
w_jj_e,w_ii_e=findxy_model(coords[1]+xyrange,coords[0]+xyrange,w_lon,w_lat)
w_jj_e = min(w_jj_e,shape(w_lon)[0]); w_ii_e = min(w_ii_e,shape(w_lon)[1]);
# print(w_jj_s,w_jj_e,w_ii_s,w_ii_e)
# print(w_lon[w_jj_s,w_ii_s],w_lat[w_jj_s,w_ii_s])
# print(w_lon[w_jj_e,w_ii_e],w_lat[w_jj_e,w_ii_e])

# Import z-levels of ROMS grid (this is stupid, would be nicer if hosted...)
from scipy.io import loadmat
roms_z=loadmat('useast_z.mat')
roms_z=roms_z['z_r']
roms_z.shape
roms_z=np.flipud(roms_z)
roms_z=roms_z[:,r_jj,r_ii] * 3.28084  #convert to feet

xy_land=np.squeeze(roms_dataset['mask_rho'][r_jj_s:r_jj_e,r_ii_s:r_ii_e])
xy_bathy=np.squeeze(roms_dataset['h'][r_jj_s:r_jj_e,r_ii_s:r_ii_e])
xy_bathy[xy_land==0]=np.nan; xy_bathy=np.ma.array(xy_bathy,mask=np.isnan(xy_bathy)) * -1

xy_wave=np.squeeze(roms_dataset['Hwave'][roms_start_index,r_jj_s:r_jj_e,r_ii_s:r_ii_e]) * 3.28084  #convert to feet
xy_wave[xy_land==0]=np.nan; xy_wave=np.ma.array(xy_wave,mask=np.isnan(xy_wave))
# xy_wave=flipud(xy_wave)
xy_temp=np.squeeze(roms_dataset['temp'][roms_start_index,-1,r_jj_s:r_jj_e,r_ii_s:r_ii_e]) * 9/5 + 32  #convert to Fahrenheit
xy_temp[xy_land==0]=np.nan; xy_temp=np.ma.array(xy_temp,mask=np.isnan(xy_temp))
# xy_temp=flipud(xy_temp)
# xy_u_wind=np.squeeze(wrf_dataset['u_10m_tr'][wrf_start_index,w_jj_s:w_jj_e,w_ii_s:w_ii_e]) * 1.94384  #convert to kt
# xy_v_wind=np.squeeze(wrf_dataset['v_10m_tr'][wrf_start_index,w_jj_s:w_jj_e,w_ii_s:w_ii_e]) * 1.94384  #convert to kt
xy_u_wind=np.squeeze(wrf_dataset['u_10m_tr'][wrf_start_index,w_jj_s:w_jj_e,w_ii_s:w_ii_e]) * 1.94384  #convert to kt
xy_v_wind=np.squeeze(wrf_dataset['v_10m_tr'][wrf_start_index,w_jj_s:w_jj_e,w_ii_s:w_ii_e]) * 1.94384  #convert to kt
xy_mag_wind = (xy_u_wind ** 2 + xy_v_wind ** 2 ) ** 0.5
t_temp=np.squeeze(roms_dataset['temp'][roms_start_index:roms_end_index+1,:,r_jj,r_ii]) * 9/5 + 32  #convert to Fahrenheit
# t_temp=flipud(t_temp)
t_wave=np.squeeze(roms_dataset['Hwave'][roms_start_index:roms_end_index+1,r_jj,r_ii]) * 3.28084  #convert to feet
# t_wave=flipud(t_wave)
t_u_wind=np.squeeze(wrf_dataset['u_10m_tr'][wrf_start_index:wrf_end_index+1,w_jj,w_ii]) * 1.94384  #convert to kt
t_v_wind=np.squeeze(wrf_dataset['v_10m_tr'][wrf_start_index:wrf_end_index+1,w_jj,w_ii]) * 1.94384  #convert to kt
t_mag_wind = (t_u_wind ** 2 + t_v_wind ** 2 ) ** 0.5
t_zeta=np.squeeze(roms_dataset['zeta'][roms_start_index:roms_end_index+1,r_jj,r_ii]) * 3.28084  #convert to feet

tz_temp=np.squeeze(roms_dataset['temp'][roms_start_index:roms_end_index+1,:,r_jj,r_ii]) * 9/5 + 32  #convert to Fahrenheit

# ROMS time
r_time=np.squeeze(roms_dataset['time'][roms_start_index:roms_end_index+1])
r_time_ar = np.array([roms_origin_date + datetime.timedelta(hours=r_time[i]) for i in range(size(r_time))])
# ROMS time adjusted to local
r_time_ar_tz = np.array([roms_origin_date + datetime.timedelta(hours=r_time[i]) + \
                         datetime.timedelta(hours=tz[0]) for i in range(size(r_time))])

# WRF time
w_time=np.squeeze(wrf_dataset['time'][wrf_start_index:wrf_end_index+1])
w_time_ar = np.array([wrf_origin_date + datetime.timedelta(hours=w_time[i]) for i in range(size(w_time))])
# WRF time adjusted to local
w_time_ar_tz = np.array([wrf_origin_date + datetime.timedelta(hours=w_time[i]) + \
                         datetime.timedelta(hours=tz[0]) for i in range(size(w_time))])

figsize(25,25)
matplotlib.rcParams.update({'font.size': 18})

tz_start_date = start_date + datetime.timedelta(hours=tz[0])

plt.suptitle(site + ' ' + str(coords[0]) + '$^\circ$N ' + \
             str(coords[1]) + '$^\circ$W' + '   \nForecast Start: ' + \
             tz_start_date.strftime("%d %b %Y %-I%p" + " " + tz[1]),fontsize=48,family='Helvetica')

plt.subplot2grid((7, 4), (0, 0), rowspan=2) 
map = Basemap(projection='merc',
    resolution='c',lat_0=((np.max(r_lat[r_jj_s:r_jj_e,r_ii_s:r_ii_e])-np.min(r_lat[r_jj_s:r_jj_e,r_ii_s:r_ii_e]))/2),
    lon_0=((np.max(r_lon[r_jj_s:r_jj_e,r_ii_s:r_ii_e])-np.min(r_lon[r_jj_s:r_jj_e,r_ii_s:r_ii_e]))/2),
    llcrnrlon=np.min(r_lon[r_jj_s:r_jj_e,r_ii_s:r_ii_e]),llcrnrlat=np.min(r_lat[r_jj_s:r_jj_e,r_ii_s:r_ii_e]),
    urcrnrlon=np.max(r_lon[r_jj_s:r_jj_e,r_ii_s:r_ii_e]),urcrnrlat=np.max(r_lat[r_jj_s:r_jj_e,r_ii_s:r_ii_e]))
map.drawcoastlines()
map.drawcountries()
map.drawstates()
# map.pcolormesh(r_lon[r_jj_s:r_jj_e,r_ii_s:r_ii_e],r_lat[r_jj_s:r_jj_e,r_ii_s:r_ii_e],\
#                xy_bathy[:,:],cmap='jet',vmin=np.nanmin(xy_bathy),vmax=0,latlon='true')
map.pcolormesh(r_lon[r_jj_s:r_jj_e,r_ii_s:r_ii_e],r_lat[r_jj_s:r_jj_e,r_ii_s:r_ii_e],\
               xy_bathy[:,:],cmap='jet',vmin=-5000,vmax=0,latlon='true')
cbar=map.colorbar(location='bottom')
plt.setp(cbar.ax.get_xticklabels()[0::2], visible=False)
x,y = map(coords[1], coords[0])
map.plot(x,y,'y*',markersize=16)
plt.title(('Bathymetry (m)'),fontsize=24,family='Helvetica')

plt.subplot2grid((7, 4), (0, 1), rowspan=2)
map = Basemap(projection='merc',
    resolution='c',lat_0=((np.max(r_lat[r_jj_s:r_jj_e,r_ii_s:r_ii_e])-np.min(r_lat[r_jj_s:r_jj_e,r_ii_s:r_ii_e]))/2),
    lon_0=((np.max(r_lon[r_jj_s:r_jj_e,r_ii_s:r_ii_e])-np.min(r_lon[r_jj_s:r_jj_e,r_ii_s:r_ii_e]))/2),
    llcrnrlon=np.min(r_lon[r_jj_s:r_jj_e,r_ii_s:r_ii_e]),llcrnrlat=np.min(r_lat[r_jj_s:r_jj_e,r_ii_s:r_ii_e]),
    urcrnrlon=np.max(r_lon[r_jj_s:r_jj_e,r_ii_s:r_ii_e]),urcrnrlat=np.max(r_lat[r_jj_s:r_jj_e,r_ii_s:r_ii_e]))
map.drawcoastlines()
map.drawcountries()
map.drawstates()
# map.pcolormesh(r_lon[r_jj_s:r_jj_e,r_ii_s:r_ii_e],r_lat[r_jj_s:r_jj_e,r_ii_s:r_ii_e],\
#                xy_wave[:,:],cmap='jet',vmin=0,vmax=np.nanmax(xy_wave),latlon='true')
map.pcolormesh(r_lon[r_jj_s:r_jj_e,r_ii_s:r_ii_e],r_lat[r_jj_s:r_jj_e,r_ii_s:r_ii_e],\
               xy_wave[:,:],cmap='jet',vmin=0,vmax=10,latlon='true')
x,y = map(coords[1], coords[0])
map.plot(x,y,'y*',markersize=16)
map.colorbar(location='bottom')
plt.title(('Wave Height (ft)'),fontsize=24,family='Helvetica')

plt.subplot2grid((7, 4), (0, 2), rowspan=2)
map = Basemap(projection='merc',
    resolution='c',lat_0=((np.max(r_lat[r_jj_s:r_jj_e,r_ii_s:r_ii_e])-np.min(r_lat[r_jj_s:r_jj_e,r_ii_s:r_ii_e]))/2),
    lon_0=((np.max(r_lon[r_jj_s:r_jj_e,r_ii_s:r_ii_e])-np.min(r_lon[r_jj_s:r_jj_e,r_ii_s:r_ii_e]))/2),
    llcrnrlon=np.min(r_lon[r_jj_s:r_jj_e,r_ii_s:r_ii_e]),llcrnrlat=np.min(r_lat[r_jj_s:r_jj_e,r_ii_s:r_ii_e]),
    urcrnrlon=np.max(r_lon[r_jj_s:r_jj_e,r_ii_s:r_ii_e]),urcrnrlat=np.max(r_lat[r_jj_s:r_jj_e,r_ii_s:r_ii_e]))
map.drawcoastlines()
map.drawcountries()
map.drawstates()
# map.pcolormesh(r_lon[r_jj_s:r_jj_e,r_ii_s:r_ii_e],r_lat[r_jj_s:r_jj_e,r_ii_s:r_ii_e],\
#                xy_temp[:,:],cmap='jet',vmin=np.nanmin(xy_temp),vmax=np.nanmax(xy_temp),latlon='true')
map.pcolormesh(r_lon[r_jj_s:r_jj_e,r_ii_s:r_ii_e],r_lat[r_jj_s:r_jj_e,r_ii_s:r_ii_e],\
               xy_temp[:,:],cmap='jet',vmin=80,vmax=90,latlon='true')
x,y = map(coords[1], coords[0])
map.plot(x,y,'y*',markersize=16)
map.colorbar(location='bottom')
plt.title(('SST ($^\circ$F)'),fontsize=24,family='Helvetica')

plt.subplot2grid((7, 4), (0, 3), rowspan=2)
map = Basemap(projection='merc',
    resolution='c',lat_0=((np.max(w_lat[w_jj_s:w_jj_e,w_ii_s:w_ii_e])-np.min(w_lat[w_jj_s:w_jj_e,w_ii_s:w_ii_e]))/2),
    lon_0=((np.max(w_lon[w_jj_s:w_jj_e,w_ii_s:w_ii_e])-np.min(w_lon[w_jj_s:w_jj_e,w_ii_s:w_ii_e]))/2),
    llcrnrlon=np.min(w_lon[w_jj_s:w_jj_e,w_ii_s:w_ii_e]),llcrnrlat=np.min(w_lat[w_jj_s:w_jj_e,w_ii_s:w_ii_e]),
    urcrnrlon=np.max(w_lon[w_jj_s:w_jj_e,w_ii_s:w_ii_e]),urcrnrlat=np.max(w_lat[w_jj_s:w_jj_e,w_ii_s:w_ii_e]))
map.drawcoastlines()
map.drawcountries()
map.drawstates()
# map.pcolormesh(w_lon[w_jj_s:w_jj_e,w_ii_s:w_ii_e],w_lat[w_jj_s:w_jj_e,w_ii_s:w_ii_e],xy_mag_wind[:,:],cmap='jet',\
#                vmin=0,vmax=np.nanmax(xy_mag_wind),latlon='true')
map.pcolormesh(w_lon[w_jj_s:w_jj_e,w_ii_s:w_ii_e],w_lat[w_jj_s:w_jj_e,w_ii_s:w_ii_e],xy_mag_wind[:,:],cmap='jet',\
               vmin=0,vmax=25,latlon='true')
#Normalized vectors
# map.quiver(w_lon[w_jj_s:w_jj_e:qv_space,w_ii_s:w_ii_e:qv_space],\
#            w_lat[w_jj_s:w_jj_e:qv_space,w_ii_s:w_ii_e:qv_space],\
#            xy_u_wind[::qv_space,::qv_space]/(xy_u_wind[::qv_space,::qv_space]**2+xy_v_wind[::qv_space,::qv_space]**2) ** 0.5,\
#            xy_v_wind[::qv_space,::qv_space]/(xy_u_wind[::qv_space,::qv_space]**2+xy_v_wind[::qv_space,::qv_space]**2) ** 0.5,\
#            latlon='true',width=0.005,scale=1 / 0.1)
# Magnitude vectors
map.quiver(w_lon[w_jj_s:w_jj_e:qv_space,w_ii_s:w_ii_e:qv_space],\
           w_lat[w_jj_s:w_jj_e:qv_space,w_ii_s:w_ii_e:qv_space],\
           xy_u_wind[::qv_space,::qv_space],\
           xy_v_wind[::qv_space,::qv_space],\
           latlon='true',width=0.005,scale=1 / 0.005)
x,y = map(coords[1], coords[0])
map.plot(x,y,'y*',markersize=16)
map.colorbar(location='bottom')
plt.title(('Wind Speed (kt) + Direction'),fontsize=24,family='Helvetica')

plt.subplot2grid((7, 4), (2, 0), colspan=4)
ylim([0,ceil(np.max(t_wave))])
xlim(min(r_time_ar),max(r_time_ar))
ylabel('Wave Height (ft)')
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %HZ'))
plt.gca().axes.get_xaxis().set_ticklabels([])
plot(r_time_ar,t_wave)

plt.subplot2grid((7, 4), (3, 0), colspan=4)
ylim([0,ceil(np.max(t_mag_wind))])
xlim(min(r_time_ar),max(r_time_ar))
ylabel('Wind Speed (kt)')
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %HZ'))
plt.gca().axes.get_xaxis().set_ticklabels([])
plot(w_time_ar,t_mag_wind)

plt.subplot2grid((7, 4), (4, 0), colspan=4)
ylim([floor(np.min(t_zeta)),ceil(np.max(t_zeta))])
xlim(min(r_time_ar),max(r_time_ar))
ylabel('Sea Level (ft)')
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %HZ'))
plt.gca().axes.get_xaxis().set_ticklabels([])
plot(r_time_ar,t_zeta)

plt.subplot2grid((7, 4), (5, 0), colspan=4, rowspan=2)
pcolormesh(r_time_ar_tz,roms_z,np.transpose(np.fliplr(tz_temp)),vmin=floor(np.min(tz_temp))+32,\
           vmax=ceil(np.max(tz_temp))-1,shading='gouraud',cmap='jet')
ylim([-120,ceil(np.max(roms_z))])
ylabel('Depth (ft)')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %-I%p'))
ax = plt.colorbar(orientation="horizontal")
ax.set_label('Ocean Temperature ($^\circ$F)',rotation=0)

plt.savefig('divesite_' + start_date.strftime("%d%b%Y%H"+"Z"))

