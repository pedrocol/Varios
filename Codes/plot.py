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


def import_data3d(ncfile=None,var=None):
    fh = Dataset(ncfile, mode='r')
    #Get data
    data = fh.variables[str(var)][0,:,:,:]
    fh.close() #Close file
    return data

def import_grid_z(ncfile=None):
    fh = Dataset(ncfile, mode='r')
    #Get data
    grid = {}
    grid['mbathy' ] = fh.variables['mbathy' ][0,:,:]
    grid['e3t_0'    ] = fh.variables['e3t_0'    ][0,:,:,:]
    grid['gdept_0'  ] = fh.variables['gdept_0'  ][0,:,:,:]
    grid['nav_lon'] = fh.variables['nav_lon'][:,:]
    grid['nav_lat'] = fh.variables['nav_lat'][:,:]
    fh.close() #Close file
    return grid

def import_dim(ncfile=None,variable=None,T=None):
    fh = Dataset(ncfile, mode='r')
    #Get data
    data = fh.variables[str(variable)][:]
    fh.close() #Close file
    return data

def import_bathy(ncfile=None):
    fh = Dataset(ncfile, mode='r')
    #Get data
    grid = {}
    grid['Bathymetry'] = fh.variables['Bathymetry'][:,:]
    grid['nav_lon'] = fh.variables['nav_lon'][:,:]
    grid['nav_lat'] = fh.variables['nav_lat'][:,:]
    fh.close() #Close file
    return grid


def plot_data_contour_average(conf="NATL60",exp=4,section=4,year=76,variable='vomecrty',T=0,contour='tracer',coord="s",plt_cen="yes"):
    """Makes a contourf plot of the xtrac sections of the averaged data and 
       put the contours of different densities
    """
    #Seaborn
    sns.set()
    sns.set_context("paper",font_scale=1.8,rc={"lines.linewidth": 2,'grid.linewidth': 0.05})
    #Directories
    root_xtrac  = "/media/extra/Analysis/experiment_"+str(exp)+"/cdf_xtrac_brokenline/"
    root_sig0   = "/media/extra/Analysis/experiment_"+str(exp)+"/cdfsig0_broken/"
    root_sigtrp = "/media/extra/Analysis/experiment_"+str(exp)+"/cdfsigtrp_broken/"
    ncfile_vel  = root_xtrac+"/section_"+str(section)+"/"+str(conf)+"/"+str(year)+ \
                              "/ave_"+str(conf)+"_y00"+str(year)+".nc"
    ncfile_dens = root_sig0+"/section_"+str(section)+"/"+str(conf)+"/"+str(year)+ \
                              "/ave_"+str(conf)+"_y00"+str(year)+".nc"
    ncfile_sigtrp = root_sigtrp+"/section_"+str(section)+"/"+str(conf)+"/"+str(year)+ \
                              "/ave_"+str(conf)+"_y00"+str(year)+".nc"
    root_gen   = "/media/extra/Analysis/experiment_"+str(exp)+"/average/"
    if variable == 'votkeavt': grid,vmax,vmin='gridW',10,1e-5
    if variable == 'vovecrtz': grid,vmax,vmin='gridW',0.001,-0.001
    ncfile_data = root_gen+"/"+str(conf)+"/"+str(year)+ \
             "/ave_"+str(conf)+"_y00"+str(year)+"_gridW.nc"
    ncfile_ptrc = root_gen+"/"+str(conf)+"/"+str(year)+ \
             "/ave_"+str(conf)+"_y00"+str(year)+"_ptrcT.nc"
    #Font sizes
    plt.figure(figsize=(10,7))
    fs = 10
    dire = "ver"
    if section == 34:  dire="hor"
    if dire == "ver":    var_nav,xlab = 'nav_lat',"Latitude"
    if dire == "hor":    var_nav,xlab = 'nav_lon',"Longitude"
    #Mask and bathy
    #bathy = import_data3d(ncfile=ncfile_vel,variable="Bathymetry")
    #bathy = bathy["Bathymetry"][0,:-1]
    isec = import_data2d(ncfile=ncfile_vel,variable="isec")[0,:]
    jsec = import_data2d(ncfile=ncfile_vel,variable="jsec")[0,:]
    isec,jsec = isec.astype(int)[:-1],jsec.astype(int)[:-1]
    if variable in ["vnorm","votemper","vosaline"]:
       data_xtrac = import_dataT(ncfile=ncfile_vel,variable=str(variable),T=T)
       data_vel = data_xtrac[str(variable)][:,0,:-1]
    if variable in ["votkeavt","vovecrtz"]:
       data_vel = import_data3d(ncfile=ncfile_data,var=variable)[:,jsec,isec]
    if variable in ["tracer_1","tracer_7"]:
       data_vel = import_data3d(ncfile=ncfile_ptrc,var=variable)[:,jsec,isec]
    ##Contour
    if contour == "dens":
       data_sig0 = import_dataT(ncfile=ncfile_dens,variable="vosigma0",T=T)
       data_cont = data_sig0['vosigma0'][:,0,:-1]
       levels,colors=[27.6,27.8,27.85],['r','k','g','b']
    if contour != "dens" and contour != None:
       data_cont = import_data3d(ncfile=ncfile_ptrc,var=contour)[:,jsec,isec]
       levels,colors=[0.1,0.2,0.3,0.4],['r','k','g','b']
       if contour == "tracer_7": levels,colors=[0.2,0.3,0.4,0.6],['r','k','g','b']
    #Build grid
    if coord == "s":
      e3v_0 = import_data2d(ncfile_vel,variable="e3v_0")
      e3v_0 = e3v_0.data[:,0,:-1]
      gdept_0 = np.zeros_like(e3v_0)
      npi,npk = e3v_0.shape[1],e3v_0.shape[0]
      for ji in range(0,npi):
         gdept_0[0,ji] = e3v_0[0,ji]
         for jk in range(1,npk):
            gdept_0[jk,ji] = gdept_0[jk-1,ji] + e3v_0[jk,ji]
      #yy = grid['gdept_0'][:,index_y,index_x] #Vertical part of the grid is given by the coordinate
      xx = import_nav(ncfile=ncfile_vel,variable=str(var_nav) )
      xx = xx[0,:-1]
      yy = gdept_0[:,:]
      z_levels = yy.shape[0]
      yi = np.linspace(0,5000,z_levels) #Dummy, just for have z_levels
      xi, yyi = np.meshgrid(xx, yi) #We repeat the horizontal part of the grid by the number of zlevels
      X,Y = xi,yy
    if coord == "z":
      deptht = import_dim(ncfile=ncfile_vel,variable="deptht")
      xx = import_nav(ncfile=ncfile_vel,variable=str(var_nav))
      xx = xx[0,:-1]
      grid = np.meshgrid(xx,deptht)
      X = grid[0]
      Y = grid[1]
    #Plot setup
    vmax = data_vel.max()
    vmin = data_vel.min()
    nb_contours = 50.
    contourstep = np.ceil((vmax-vmin)/nb_contours)
    adjust = "yes"
    if adjust == "yes":
       if variable == "vnorm" or variable == "vomecrty":
          if section == 1 : vmax,vmin,contourstep = 0.6,-0.3,0.05
          if section == 15: vmax,vmin,contourstep = 0.6,-0.1,0.05
          if section == 19: vmax,vmin,contourstep = 0.7,-0.1,0.05
          if section == 22: vmax,vmin,contourstep = 0.4,-0.1,0.05
          if section == 27: vmax,vmin,contourstep = 0.5,-0.1,0.05
          shifted_cmap = cmapshift.remap_cmap(cmap=cm.coolwarm_r,midpoint=(abs(vmin)/(abs(vmax)+abs(vmin))))
       if variable == "vosaline":
          #vmax,vmin,contourstep = 35.5,34.5,0.01
          vmax,vmin,contourstep = 35.2,34.5,0.01
          shifted_cmap = cmapshift.remap_cmap(cmap=cm.jet,midpoint=0.5)
       if variable == "votemper":
          vmax,vmin,contourstep = 8.,3.,0.1
          if section == 1 : vmax,vmin,contourstep = 8.,0.,0.1
          shifted_cmap = cmapshift.remap_cmap(cmap=cm.coolwarm,midpoint=0.5)
       if variable == "vovecrtz":
          vmax,vmin,contourstep = 0.001,-0.001,0.0001
          shifted_cmap = cmapshift.remap_cmap(cmap=cm.coolwarm,midpoint=0.5)
       if variable == "votkeavt":
          vmax,vmin = 10,1e-5
          shifted_cmap = cmapshift.remap_cmap(cmap=cm.coolwarm,midpoint=0.5)
       if variable in ["tracer_1","tracer_7"]:
          vmax,vmin,contourstep = 0.7,0,0.05
          shifted_cmap = cmapshift.remap_cmap(cmap=cm.coolwarm,midpoint=0.5)
    cmap=shifted_cmap
    nb_contours = np.ceil((vmax-vmin)/contourstep)
    #contours = np.arange(vmin,vmax,contourstep)
    contours = [round(x , 4) for x in np.linspace(vmin,vmax,nb_contours+1)]
    norm = mpl.colors.Normalize(vmin=vmin,vmax=vmax)
    #Plot routines 1
    ax = plt.gca()
    contf_masked_vel = plt.contourf(X,Y,data_vel.mask,colors='gray',corner_mask=False)
    if variable == 'votkeavt':
       levels  = np.logspace(-5,1,500)
       norm = colorsLog.LogNorm(vmin=vmin,vmax=vmax)
       CS = plt.contourf(X,Y,data_vel,cmap = cmap,norm=norm,levels=levels,\
                         corner_mask=False,interpolation="nearest")
       cbar=plt.colorbar(CS,ticks = [1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1])
       cbar.set_label('log')
    else:
       formats='%1.2f'
       if variable == "vovecrtz": formats=ticker.ScalarFormatter(useMathText=True);formats.set_powerlimits((0, 0))
       norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
       CS = plt.contourf(X,Y,data_vel,contours,vmin=vmin,vmax=vmax,\
                         cmap = cmap,extend='both',corner_mask=False)
       cbar=plt.colorbar(CS,format=formats)
    #batthyplot       = plt.plot(xx,bathy,color='r',lw=3.)
    if contour != None:
       cont_contour = plt.contour(X,Y,data_cont,levels=levels,colors=colors,corner_mask=False)
       cont_labels  = plt.clabel(cont_contour,inline=1,fontsize=fs,fmt='%1.4g')
    if variable == 'vnorm':  cont_vel0        = plt.contour(X,Y,data_vel,(0,),colors='w',linestyle='-',corner_mask=False)
    #Plot center of mass
    if plt_cen == "yes":
       #Get vein position
       X_ov = import_data0d(ncfile=ncfile_sigtrp,variable="cent_ve278X_0",T=T)
       Z_ov = import_data0d(ncfile=ncfile_sigtrp,variable="cent_ve278Z_0",T=T)
       plt.plot(xx[int(X_ov)],Z_ov,marker='D',color='w',label="vel > 0")
       X_ov = import_data0d(ncfile=ncfile_sigtrp,variable="cent_ve278X_01",T=T)
       Z_ov = import_data0d(ncfile=ncfile_sigtrp,variable="cent_ve278Z_01",T=T)
       plt.plot(xx[int(X_ov)],Z_ov,marker='D',color='k',label="vel > 0.1")
       X_ov = import_data0d(ncfile=ncfile_sigtrp,variable="cent_ve278X_02",T=T)
       Z_ov = import_data0d(ncfile=ncfile_sigtrp,variable="cent_ve278Z_02",T=T)
       plt.plot(xx[int(X_ov)],Z_ov,marker='D',color='m',label="vel > 0.2")
       plt.legend(loc="lower left")
    #Plot grid
    plot_grid="no"
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
    #Plot cosmetics
    title=str(variable)+" - "+str(conf)+" - Section "+str(section)+" - year="+str(year)
    if contour != None:
       title=str(variable)+"+"+str(contour)+" - "+str(conf)+" - Section "+str(section)+" - year="+str(year)
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel("Depth (m)")
    if section <= 28: plt.gca().invert_xaxis()
    plt.autoscale(True,'x',True)
    ylim = 3200
    if section == 1 : ylim = 700
    plt.ylim(ymax=ylim,ymin=0)
    plt.gca().invert_yaxis()
    #plt.autoscale(True,'y',True)
    #plt.savefig( str(dirs)+"/"+"{0:03d}".format(T)+"_section_"+"{0:02d}".format(section)+"_"+"y00"+str(year)+".png")
    #plt.close()
    plt.show(block=False)

    #if section == 34 and vel_var == "votemper":
    #   contours,vmin,vmax = [1,1.5,2,2.5,2.7,2.9,3,3.1,3.2,3.3,3.4,3.5,3.6,3.8,4,4.5,5,5.5,6,7,8,9,11],1,11
    #   cmap = cm.jet
    #if section == 34 and vel_var == "vosaline":
    #   contours,vmin,vmax = [33,34.5,34.7,34.75,34.8,34.82,34.83,34.84,34.85,34.86,34.87,34.88,34.89,34.9,34.91,34.92,34.94,34.97,35,35.05,35.1,35.15,35.2],33,35.2
    #   cmap = cm.jet
    #if section == 34 and vel_var == "vomecrty":
    #   contours,vmin,vmax = [-0.40,-0.30,-0.20,-0.15,-0.10,-0.08,-0.06,-0.04,-0.02,0,0.02,0.04,0.06,0.08,0.10,0.15,0.20,0.30,0.40],-0.40,0.40
    #   cmap = cm.jet_r


def import_grid_z(ncfile=None):
    fh = Dataset(ncfile, mode='r')
    #Get data
    grid = {}
    grid['mbathy' ] = fh.variables['mbathy' ][0,:,:]
    grid['e3t_0'    ] = fh.variables['e3t_0'    ][0,:,:,:]
    grid['gdept_0'  ] = fh.variables['gdept_0'  ][0,:,:,:]
    grid['nav_lon'] = fh.variables['nav_lon'][:,:]
    grid['nav_lat'] = fh.variables['nav_lat'][:,:]
    fh.close() #Close file
    return grid

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

