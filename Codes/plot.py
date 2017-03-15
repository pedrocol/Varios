#Importing tools
from netCDF4 import Dataset

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import rc
import matplotlib.cm as cm
import matplotlib.pylab as pylab
import os
import data_tools as data_tools
import matplotlib.colors as colors
import cmapshift as cmapshift
import filters as filters
#3D plot/animation
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
#Basemap
#from mpl_toolkits.basemap import Basemap
#from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FuncFormatter
from scipy.optimize import curve_fit
#Seaborn
import seaborn as sns
labo="yes"
if labo == "no":
   #Cartopy
   import cartopy
   import cartopy.crs as ccrs
   from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
   from cartopy.feature import NaturalEarthFeature
   import cartopy.feature as cfeature


def plot_contour_data_sections(conf=None,experiment=None,section=None,year=None,variable=None,tracer=1, \
                               contour=None,x_1=None,x_2=None,y_1=None,y_2=None):
    """It makes a contour plot along 4 given points 
    """
    fs = 10
    root_gen   = "/media/extra/Analysis/experiment_"+str(experiment)+"/average/"
    root_dens   = "/media/extra/Analysis/experiment_"+str(experiment)+"/cdfsig0_region/"
    ncfile_grid ="/media/extra/GRIDS/DENST12-I/DENST12_mesh_zgr.nc"
    if variable == 'vozocrtx': grid='gridU'
    if variable == 'vomecrty': grid='gridV'
    if variable == 'votemper': grid='gridT'
    if variable == 'vosaline': grid='gridT'
    if variable == 'votkeavt': grid='gridW'
    if variable == 'vovecrtz': grid='gridW'
    if variable == 'tracer'  : grid='ptrcT'
    ncfile = root_gen+"/"+str(conf)+"/"+str(year)+ \
             "/ave_"+str(conf)+"_y00"+str(year)+"_"+str(grid)+".nc"
    ncfile_ptrc = root_gen+"/"+str(conf)+"/"+str(year)+ \
             "/ave_"+str(conf)+"_y00"+str(year)+"_ptrcT.nc"
    ncfile_dens = root_dens+"/"+str(conf)+"/"+str(year)+ \
                              "/ave_"+str(conf)+"_y00"+str(year)+".nc"
    if x_1 == None:
       x_1,x_2,y_1,y_2 = plot_sigma.data_sections()
       x_1,x_2,y_1,y_2 = x_1[section],x_2[section],y_1[section],y_2[section]
    xxx = [np.linspace(x_1,x_2,x_2-x_1+1)]
    yyy = [np.linspace(y_1,y_2,y_2-y_1+1)]
    #Get data
    data_plot = import_data3d(ncfile,variable)[:,yyy,xxx][:,0,:]
    if contour == "dens": data_contour = import_data3d(ncfile_dens,'vosigma0')[:,yyy,xxx][:,0,:]
    if contour == "ptrc": data_contour = import_data3d(ncfile_ptrc,'tracer_'+str(tracer))[:,yyy,xxx][:,0,:]
    #bathy = import_bathy("/scratch/cnt0024/hmg2840/colombo/DENST12/DENST12-I-sco/bathymetry_DENST12_V3.3.nc")
    #Get grid and data to plot
    grid = import_grid_z(ncfile_grid)
    #Find indexes
    lat,lon=grid['nav_lat'][yyy,xxx][0,:],grid['nav_lon'][yyy,xxx][0,:]
    dimx,dimy = lon.shape,lat.shape
    #Build grid
    if x_1 == x_2:   #Meridional
       xx = lat
       xlabel = "Latitude"
    elif y_1 == y_2: #Zonal
       xx = lon
       xlabel = "Longitude"
    ##s way
    #yy = grid['gdept_0'][:,yyy,xxx][:,0,:] #Vertical part of the grid is given by the coordinate
    #z_levels = yy.shape[0]
    #yi = np.linspace(0,5000,z_levels) #Dummy, just for have z_levels
    #xi, yyi = np.meshgrid(xx, yi) #Dummy, we repeat the horizontal part of the grid by the number of levels
    #XX = xi
    #YY = yy
    ##z way
    deptht = import_dim(ncfile=ncfile_grid,variable="gdept_1d")
    z_levels = deptht.shape[1]
    grid = np.meshgrid(xx,deptht)
    XX = grid[0]
    yy = grid[1]
    #Select data
    masked_data_vel = np.ma.getmask(data_plot)
    zz = data_plot
    #Plot setup
    vmin = zz.min()
    vmax = zz.max()
    nb_contours = 50.
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    contourstep = (vmax-vmin)/nb_contours
    contours = np.arange(vmin,vmax+contourstep,contourstep)
    my_cmap = cm.jet_r
    #Plot routine
    plt.figure()
    ax = plt.gca()
    #contf_masked_vel = plt.contourf(xi,yy,masked_data_vel,colors='gray')
    CS = plt.contourf(XX,yy,zz,contours,cmap = my_cmap,norm=norm,\
                      vmin=vmin,vmax=vmax,extend="both",corner_mask=False)
    cbar=plt.colorbar(CS,format='%1.2g')
    if contour == "dens": cont_sigma = plt.contour(xi,yy,data_contour,(27.60,27.80,27.85),colors=('r','k','g'))
    if contour == "ptrc": cont_sigma = plt.contour(xi,yy,data_contour,(0.2,0.3,0.4))#,colors=('r','k','g'))
    if contour != "none": dens_labels      = plt.clabel(cont_sigma,inline=1,fontsize=fs,fmt='%1.4g')
    #plt.plot(xx,bathy,color='r',lw=3.)
    #Plot grid
    plot_grid = "no"
    if plot_grid == "yes":
      cont = 0
      while cont < z_levels-1:
            plt.plot(xx,yy[cont,:],color='#808080')
            cont += 1
      cont_x = 0
      while cont_x < xx.shape[0]:
            plt.plot([xx[cont_x],xx[cont_x]],[yy[0,cont_x],yy[z_levels-2,cont_x]],color='#808080')
            cont_x += 1
    #Plot functions
    plt.title(str(conf))
    plt.xlabel(xlabel)
    plt.ylabel("Depth")
    #plt.gca().invert_yaxis()
    plt.autoscale(True,'x',True)
    plt.ylim(ymax=0,ymin=2500)
    #plt.autoscale(True,'y',True)
    plt.show(block=False)

