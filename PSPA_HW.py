# -*- coding: utf-8 -*-
'''
Created on 27 Jan 2017

@author: jmc
'''

from ctypes import sizeof
from multiprocessing.sharedctypes import Value
import os
from tkinter import E
from turtle import position
from typing import List
from collections import defaultdict
from mpl_toolkits.axes_grid1 import make_axes_locatable

from numpy.lib.index_tricks import unravel_index
from numpy.linalg.linalg import eig, eigvals
from scipy.sparse import coo
os.environ['PROJ_LIB'] = 'C:/Users/sergi/miniconda3/Library/share/basemap'
os.environ['PROJ_LIB'] = 'C:/Users/sergi/miniconda3/Library/share/basemap/lib/mpl_toolkits/basemap'

import csv
import numpy as np
import netCDF4 as nc4
import xarray as xr
import array

from eofs.standard import Eof
from datetime import datetime, date, time, timedelta
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
import pandas as pd

from mpl_toolkits.basemap import Basemap
from ecmwfapi import ECMWFDataServer
from sklearn import preprocessing
from numpy import array, dot, ones, diag, tile, sqrt, eye
import numpy.linalg as linalg

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.signal import argrelextrema
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from cdo import *


import os

import collections

import warnings
warnings.simplefilter("ignore", DeprecationWarning)


def llegeix_grib(episodis):
    #days= [x.strftime("%Y%m%d") for x in episodis]
    values={}
    lat=[]
    lon=[]
    
    NC1 = nc4.Dataset('WRF_slp_hist_corrected_hw.nc')
    NC2 = nc4.Dataset('WRF_z500_hist_corrected_hw.nc')
    NC3 = nc4.Dataset('WRF_tmax_hist_corrected_hw.nc')
    
    # iterate dates    
    lat,lon,psl,zg500,tmax = NC1.variables['lat'],NC1.variables['lon'],NC1.variables['msl'],NC2.variables['z'],NC3.variables['mx2t']
    for idx,date in enumerate(episodis):
    #for idx,date in episodis:
        date_str=date.strftime("%Y%m%d")
        values[date_str]={}
        psl=np.ma.filled(psl[:],float('nan'))
        zg500=np.ma.filled(zg500[:],float('nan'))
        tmax=np.ma.filled(tmax[:],float('nan'))
        values[date_str]['msl'] = np.concatenate((psl[idx].flatten(),psl[idx-1].flatten(),psl[idx-2].flatten()))
        values[date_str]['z'] = np.concatenate((zg500[idx].flatten(),zg500[idx-1].flatten(),zg500[idx-2].flatten()))
        values[date_str]['mx2t'] = np.concatenate((tmax[idx].flatten(),tmax[idx-1].flatten(),tmax[idx-2].flatten()))

    print("printem values") 
    print(values)
    return np.array(lat),np.array(lon),values


    pass


def llegeix_episodis(fitxer_dies):
    # Llegim els episodis que farem
    dies=[]
    with open(fitxer_dies, 'r') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            dia=datetime.strptime(row[0],'%d/%m/%Y')
            #dia=datetime.strptime(row[0],'%Y%m%d')
            dies.append(dia)
    return dies         
    

def converteix_matriu(values,variable):
    
    matriu=[]
    for dia in values.keys():
        fila=values[dia][variable]
        matriu.append(fila)

    return np.vstack(matriu)
     

def escriu_dades(data,nom):
    
    #Escrivim PCA
    fitxer=nom+'.csv'
    np.savetxt(fitxer, data, delimiter=";")       


def calcula_EOF(data,lat):
    print ("Transposem per fer T-mode...")
    transposed_data=np.transpose(data)
    data=transposed_data
    #escriu_dades(transposed_data,'transpoting')
    # Si fem S-mode no cal tranposar
    
    #Square-root of cosine of latitude weights are applied before the computation of EOFs.
    #coslat = np.cos(np.deg2rad(lat)).clip(0., 1.)
    #weights = np.sqrt(coslat)[..., np.newaxis]
    #print wgts
    solver = Eof(data)
    print ("Fem scree test per saber el numero de components a retenir...")
    eigenvalues = solver.eigenvalues()
    variance = solver.varianceFraction()
    # Fem scree test per saber nombre PC
    num_pcs=scree_test(eigenvalues)
    # Per calcular el nombre del PC de manera automatica agafem 
    # el criteri de eigenvalues<1
    #num_pcs=np.argmax(eigenvalues<1)
    print ("El numero de components que retenim és:", num_pcs)
    pcs = solver.pcs(pcscaling=0,npcs=num_pcs)
    eof1 = solver.eofs(eofscaling=0,neofs=num_pcs)
    #eigenvalues=solver.eigenvalues()
    projectat=solver.projectField(data,neofs=num_pcs)
    #escriu_dades(pcs,'pca')
    #escriu_dades(eof1,'eof')
    #escriu_dades(eigenvalues,'lambda')
    
    #print(pcs,'pca')
    #print(eof1,'eof')
    #print(eigenvalues,'eigenvalues')
    print('Variança: ', variance*100)
    print('Eigenvalues: ', eigenvalues)

    return eof1,pcs,projectat,num_pcs


def scree_test(vaps):
    fig = plt.figure(figsize=(8,5))
    num_vars=vaps.shape[0]
    sing_vals = np.arange(num_vars) + 1
    plt.plot(sing_vals[0:30], vaps[0:30], 'ro-', linewidth=2)
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue')
    #I don't like the default legend so I typically make mine like below, e.g.
    #with smaller fonts and a bit transparent so I do not cover up data, and make
    #it moveable by the viewer in case upper-right is a bad place for it 
    leg = plt.legend(['Eigenvalues from SVD'], loc='best', borderpad=0.3, 
                     shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                     markerscale=0.4)
    leg.get_frame().set_alpha(0.4)
    leg.set_draggable(state=True)
    plt.show() 
    plt.close('all')
    num_pcs = input("Posa el numero de components que vols retenir... ")
    while not num_pcs.isdigit():
        num_pcs = input("Ha de ser un numero...")
    return int(num_pcs)

def varimax( x, normalize=False, positive=False, tol=1e-10, max_it=1000): 
    # Les columnes  han de ser els vectors propis en entrada (matriu x)
    if len(x.shape)!=2:
        raise ValueError ('AMAT must be 2-dimensional')
    elif len(x.shape)==2:     
        p,nc= x.shape
        if normalize:
            rl = tile(sqrt(diag(dot(x,x.T))), (nc,1) ).T#; % By rows.l = repmat( sqrt(diag( x*x' )), [1,nc] ); % By rows.
            x = x/rl
    TT = eye(nc)
    d=0
    
    for i in range(max_it):
        z = dot(x,TT)
        c0=dot(ones((1,p)),(z**2))
        c1=diag(c0.squeeze())/p
        c3= z**3 - dot(z,c1)
        B= dot(x.T,c3)
        U,S,V= linalg.svd(B,full_matrices=True)
        
        TT = dot(U,V)
        
        d2 = d
        d= sum(S)
        if d< d2*(1+tol):
            print ("varimax done in ", i , " iteration")
            break

    x= dot(x,TT)
    
    if positive:
        for i,item in enumerate(x.T):                      #I taker each column and I test if is sum positive
            if sum(item) < 0:                              #if not i multiplie it for -1
                x[:,i] =  -1*item
                TT[:,i] = -1*TT[:,i]

    return x




def plot_dades(eof,lats,lons):
    
    # set up orthographic map projection with
    # perspective of satellite looking down at 50N, 100W.
    # use low resolution coastlines.
    # Passem lats i lons a list per eleminar duplicats
    # fem inverse de lats perque tingui ordre original
    
    lat_list=lats.tolist()
    lon_list=lons.tolist()
    lats=list(reversed(sorted(list(set(lat_list)))))
    lons=sorted(list(set(lon_list)))
    
    mapa = Basemap(llcrnrlon =lons[0], llcrnrlat = lats[-1], urcrnrlon = lons[-1], urcrnrlat = lats[0], resolution = 'i')
    
    plt.figure(figsize=(12,5))
    # draw coastlines, country boundaries, fill continents.
    
    #mapa.fillcontinents(color='coral',lake_color='aqua')
    # draw the edge of the map projection region (the projection limb)
    #mapa.drawmapboundary(fill_color='aqua')
    # draw lat/lon grid lines every 30 degrees.

    num_variables=3
    num_dies=3
    
    # compute native map projection coordinates of lat/lon grid.
    x, y = np.meshgrid(lons, lats)
    # contour data over the map.
    
    #eof_variables=np.split_array(eof,num_variables)

    for i in range(num_variables):
        for j in range(num_dies):

            mapa.drawcoastlines(linewidth=0.5)
            mapa.drawcountries(linewidth=0.5)
            mapa.drawmeridians(np.arange(0,360,15))
            mapa.drawparallels(np.arange(-90,90,15))

            plt.subplot(3, 3, 3*i+j+1)
            #plt.title("Time n-2")
            eof_mapa=eof[j*4941+(4941*3*i):4941*(j+1)+(4941*i*3)]
            dataMesh = eof_mapa.reshape(len(lats), len(lons))
            divnorm = colors.Normalize(vmin=-8,vmax=8)
            levels = np.linspace(-8,8,17)
            cs = mapa.contour(x,y,dataMesh,colors='k',norm=divnorm, levels=levels, linewidths=0.3)
            cs2 = mapa.contourf(x,y,dataMesh, cmap='RdYlBu_r',norm=divnorm, levels=levels, extend="both")
    
            plt.clim(-8,8)
            plt.clabel(cs, fontsize=10, colors='k', inline=1,fmt='%.1f')
    plt.show()
    plt.close("all")

   
def clusters_ward(X):
    Z=linkage(X)
    # calculate full dendrogram
  
    #plt.figure(figsize=(25, 10))
    #plt.title('Hierarchical Clustering Dendrogram')
    #plt.xlabel('sample index')
    #plt.ylabel('distance')
    #dendrogram(
    #    Z,
    #    leaf_rotation=90.,  # rotates the x axis labels
    #    leaf_font_size=8.,  # font size for the x axis labels
    #)
    #plt.show()
  
    #Apliquem un criteri d'Elbow per decidir el nombre de clusters
    # Calculem les distàncies per cada iteració
    last = Z[-50:, 2] 
    last_rev = last[::-1]
    idxs = np.arange(1, len(last) + 1)
    plt.plot(idxs, last_rev)
    
    acceleration = np.diff(last, 2)  # 2nd derivative of the distances
    acceleration_rev = acceleration[::-1]
    plt.plot(idxs[:-2] + 1, acceleration_rev)
    # for local maxima
    maxims=argrelextrema(acceleration_rev , np.greater)
    k = acceleration_rev.argmax() + 2  # if idx 0 is the max of this we want 2 clusters
    print ("El calcul surt a num clusters="), k
    print ("Si no hi estas d'acord entre el nombre de clusters...")
    print ("Els màxims que hi ha per triar són: ") 
    print (", ".join('%02d'%x for x in maxims[0].tolist()))
    plt.show()
    num_clusters = input("Si hi estas d'acord posa [Y] ")
    if not num_clusters.isdigit():
        num_clusters=k
    print ("El numero de clusters que usarem serà:", num_clusters)  
    
    return int(num_clusters) 

def calcula_clusters(pcs):
    print ("Primer apliquem el criteri de Ward per obtenir numero de clusters a retenir....")
    # generate the linkage matrix
    num_clusters=clusters_ward(pcs)
    print ("Obtingut el nombre de clusters farem un K-means...")
    CLASI=KMeans(n_clusters=num_clusters).fit(pcs)
    print ("Per cada PC calculem quin tipus pertany i ho guardem en un array....")
    labels=CLASI.predict(pcs)
    #print (labels)
    #centroids = np.array(CLASI.cluster_centers_)
    #closest, _ = pairwise_distances_argmin_min(centroids, pcs)
    return np.array(labels)

def analisi_discriminant(pcs,labels,num_pcs):
    
    lda = LinearDiscriminantAnalysis().fit(pcs[:,0:num_pcs], labels)
    new_labels=lda.predict(pcs[:,0:num_pcs])
    
    dies=collections.defaultdict(list)
    tipus={}
    for date,tip in zip(episodis,new_labels):
        dia=date.strftime('%Y%m%d')
        dies[tip].append(dia)
        tipus[dia]=tip
    tipus_ordenat = collections.OrderedDict(sorted(tipus.items(), key=lambda t: t[0]))
    return tipus_ordenat,dies


def busca_maxim(EOF,num):

    keys=[str(i+1) for i in range(num)]

    Dict = dict(zip(keys, np.array(EOF[0:])))
    #print("Diccionari EOFS:   ",Dict)

    pandas_dict = pd.DataFrame(Dict)
    EOFmax = pandas_dict.idxmax(axis=1)
    Llista_EOFS = EOFmax.values.tolist()
    #print(Llista_EOFS)

    days_file="Episodes_CORDEX.csv"
    episodis=llegeix_episodis(days_file)
    #days= [x.strftime("%Y%m%d") for x in episodis]
    EOFS_dies = pd.DataFrame(list(zip(Llista_EOFS,episodis)))
    EOFS_dies.columns = ["EOF","DAY"]
    #print(EOFS_dies)

    PCA={}

    for index in keys:
        PCA[index] = EOFS_dies.loc[EOFS_dies['EOF']==index].iloc[:,1]
  

    print("Dies corresponents a PC1:",PCA['1'].shape, " PC2:", PCA['2'].shape, " PC3:",PCA['3'].shape, " PC4:", PCA['4'].shape)
     
    return EOFS_dies, PCA
    

def plot_composites(pca,values,lats,lons):
    
    # set up orthographic map projection with
    # perspective of satellite looking down at 50N, 100W.
    # use low resolution coastlines.
    # Passem lats i lons a list per eleminar duplicats
    # fem inverse de lats perque tingui ordre original
    
    inverted = collections.defaultdict(dict)

    # Canviem ordre DIES/VARIABLE per deixar preparada la malla...

    for key,subd in values.items():
        for k,v in subd.items():  # no inspiration for key/value names...
            inverted[k][key] = v

    # Preparem la malla per cada PC ...

    data_composite={}
    for pc in pca.keys():
        data_composite[pc]={}
        dies=pca[pc]
        for variable in inverted.keys():
            data=[]
            for dia in dies:
                dia_str=datetime.strftime(dia,"%Y%m%d")
                data.append(inverted[variable][dia_str])
            data_composite[pc][variable]=np.mean(data, axis=0)

    # Preparem els composites

    lat_list=lats.tolist()
    lon_list=lons.tolist()
    lats=list(reversed(sorted(list(set(lat_list)))))
    lons=sorted(list(set(lon_list)))
    
    mapa = Basemap(llcrnrlon =lons[0], llcrnrlat = lats[-1], urcrnrlon = lons[-1], urcrnrlat = lats[0], resolution = 'i')

    # draw coastlines, country boundaries, fill continents.
    #mapa.fillcontinents(color='coral',lake_color='aqua')
    # draw the edge of the map projection region (the projection limb)
    #mapa.drawmapboundary(fill_color='aqua')
    # draw lat/lon grid lines every 30 degrees.

    
    
    # compute native map projection coordinates of lat/lon grid.
    x, y = np.meshgrid(lons, lats)
    # contour data over the map.
    
    #eof_variables=np.split_array(eof,num_variables)

    for key,value in data_composite['4'].items():
        
        def get_vals(data_composite, key_list):
            for i,j in data_composite['4'].items():
                if i in key_list:
                    yield (i,j)
                yield from [] if not isinstance(j,dict) else get_vals(j, key_list)

        key_list= ['msl','z','mx2t']
        res = dict(get_vals(data_composite,key_list))

        a = np.array_split(np.array(res['msl']),3)

        x, y = np.meshgrid(lons, lats)
        plt.figure(figsize=(15,5))
        plt.subplot(1, 3, 1)
        mapa.drawcoastlines(linewidth=0.25)
        # draw lat/lon grid lines every 30 degrees.
        mapa.drawmeridians(np.arange(0,360,15))
        mapa.drawparallels(np.arange(-90,90,15))
        # contour data over the map.
        dataMesh = a[2].reshape(len(lats), len(lons))
        dataMesh2 = dataMesh/100
        #map[tipus]['psl']=dataMesh
        #cs = mapa.contour(x,y,dataMesh/100,10,colors='k')
        divnorm = colors.Normalize(vmin=1000,vmax=1026)
        levels = np.linspace(1000,1026,14)
        cs = mapa.contour(x,y,dataMesh2,colors='k',norm=divnorm, levels=levels, linewidths=0.4)
        cs2 = mapa.contourf(x,y,dataMesh2,14,cmap='RdYlBu_r',norm=divnorm, levels=levels)

        plt.clabel(cs, fontsize=12, colors='k', inline=1,fmt='%.1f')
        
        plt.subplot(1, 3, 2)
        mapa.drawcoastlines(linewidth=0.25)
        # draw lat/lon grid lines every 30 degrees.
        mapa.drawmeridians(np.arange(0,360,15))
        mapa.drawparallels(np.arange(-90,90,15))
        # contour data over the map.
        dataMesh = a[1].reshape(len(lats), len(lons))
        dataMesh2 = dataMesh/100
        #map[tipus]['psl']=dataMesh
        #cs = mapa.contour(x,y,dataMesh/100,10,colors='k')
        divnorm = colors.Normalize(vmin=1000,vmax=1026)
        levels = np.linspace(1000,1026,14)
        cs = mapa.contour(x,y,dataMesh2,colors='k',norm=divnorm, levels=levels, linewidths=0.4)
        cs2 = mapa.contourf(x,y,dataMesh2,14,cmap='RdYlBu_r',norm=divnorm, levels=levels)
        
        plt.clabel(cs, fontsize=12, colors='k', inline=1,fmt='%.1f')

       
        plt.subplot(1, 3, 3)
        mapa.drawcoastlines(linewidth=0.25)
        # draw lat/lon grid lines every 30 degrees.
        mapa.drawmeridians(np.arange(0,360,15))
        mapa.drawparallels(np.arange(-90,90,15))
        # contour data over the map.
        dataMesh = a[0].reshape(len(lats), len(lons))
        dataMesh2 = dataMesh/100
        #map[tipus]['psl']=dataMesh
        #cs = mapa.contour(x,y,dataMesh/100,10,colors='k')
        divnorm = colors.Normalize(vmin=1000,vmax=1026)
        levels = np.linspace(1000,1026,14)
        cs = mapa.contour(x,y,dataMesh2,colors='k',norm=divnorm, levels=levels, linewidths=0.4)
        cs2 = mapa.contourf(x,y,dataMesh2,14,cmap='RdYlBu_r',norm=divnorm, levels=levels)
        #plt.colorbar(cs)
        #plt.clim(20,40)
        
        plt.clabel(cs, fontsize=12, colors='k', inline=1,fmt='%.1f')
        
        plt.tight_layout()
        
        plt.show() 


        b = np.array_split(np.array(res['z']),3)

        x, y = np.meshgrid(lons, lats)
        plt.figure(figsize=(15,5))
        plt.subplot(1, 3, 1)
        mapa.drawcoastlines(linewidth=0.25)
        # draw lat/lon grid lines every 30 degrees.
        mapa.drawmeridians(np.arange(0,360,15))
        mapa.drawparallels(np.arange(-90,90,15))
        # contour data over the map.
        dataMesh = b[2].reshape(len(lats), len(lons))
        dataMesh2 = dataMesh/10
        #map[tipus]['psl']=dataMesh
        #cs = mapa.contour(x,y,dataMesh/100,10,colors='k')
        divnorm = colors.Normalize(vmin=5400,vmax=5900)
        levels = np.linspace(5400,5900,21)
        cs = mapa.contour(x,y,dataMesh2,colors='k',norm=divnorm, levels=levels, linewidths=0.4)
        cs2 = mapa.contourf(x,y,dataMesh2, cmap='RdYlBu_r',norm=divnorm, levels=levels, extend="both")

        plt.clabel(cs, fontsize=12, inline=1,fmt='%d')
        
        plt.subplot(1, 3, 2)
        mapa.drawcoastlines(linewidth=0.25)
        # draw lat/lon grid lines every 30 degrees.
        mapa.drawmeridians(np.arange(0,360,15))
        mapa.drawparallels(np.arange(-90,90,15))
        # contour data over the map.
        dataMesh = b[1].reshape(len(lats), len(lons))
        dataMesh2 = dataMesh/10
        #map[tipus]['psl']=dataMesh
        #cs = mapa.contour(x,y,dataMesh/100,10,colors='k')
        divnorm = colors.Normalize(vmin=5400,vmax=5900)
        levels = np.linspace(5400,5900,21)
        cs = mapa.contour(x,y,dataMesh2,colors='k',norm=divnorm, levels=levels, linewidths=0.4)
        cs2 = mapa.contourf(x,y,dataMesh2, cmap='RdYlBu_r',norm=divnorm, levels=levels, extend="both")
        
        plt.clabel(cs, fontsize=12, inline=1,fmt='%d')

       
        plt.subplot(1, 3, 3)
        mapa.drawcoastlines(linewidth=0.25)
        # draw lat/lon grid lines every 30 degrees.
        mapa.drawmeridians(np.arange(0,360,15))
        mapa.drawparallels(np.arange(-90,90,15))
        # contour data over the map.
        dataMesh = b[0].reshape(len(lats), len(lons))
        dataMesh2 = dataMesh/10
        #map[tipus]['psl']=dataMesh
        #cs = mapa.contour(x,y,dataMesh/100,10,colors='k')
        divnorm = colors.Normalize(vmin=5400,vmax=5900)
        levels = np.linspace(5400,5900,21)
        cs = mapa.contour(x,y,dataMesh2,colors='k',norm=divnorm, levels=levels, linewidths=0.4)
        cs2 = mapa.contourf(x,y,dataMesh2, cmap='RdYlBu_r',norm=divnorm, levels=levels, extend="both")
        #plt.colorbar(cs)
        #plt.clim(20,40)
        
        plt.clabel(cs, fontsize=12, inline=1,fmt='%d')
        
        plt.tight_layout()
        
        plt.show() 

        c = np.array_split(np.array(res['mx2t']),3)

        x, y = np.meshgrid(lons, lats)
        plt.figure(figsize=(15,5))
        plt.subplot(1, 3, 1)
        mapa.drawcoastlines(linewidth=0.25)
        # draw lat/lon grid lines every 30 degrees.
        mapa.drawmeridians(np.arange(0,360,15))
        mapa.drawparallels(np.arange(-90,90,15))
        # contour data over the map.
        dataMesh = c[2].reshape(len(lats), len(lons))
        dataMesh2 = dataMesh-273.15
        #map[tipus]['psl']=dataMesh
        #cs = mapa.contour(x,y,dataMesh/100,10,colors='k')
        divnorm = colors.Normalize(vmin=10,vmax=40)
        levels = np.linspace(10,40,31)
        #levels2 = np.linspace(10,40,6)
        #cs = mapa.contour(x,y,dataMesh2,colors='k',norm=divnorm, levels=levels2, linewidths=0.05)
        cs2 = mapa.contourf(x,y,dataMesh2,20,cmap='coolwarm',norm=divnorm, levels=levels)

        plt.clabel(cs, fontsize=12, inline=1,fmt='%d')
        
        plt.subplot(1, 3, 2)
        mapa.drawcoastlines(linewidth=0.25)
        # draw lat/lon grid lines every 30 degrees.
        mapa.drawmeridians(np.arange(0,360,15))
        mapa.drawparallels(np.arange(-90,90,15))
        # contour data over the map.
        dataMesh = c[1].reshape(len(lats), len(lons))
        dataMesh2 = dataMesh-273.15
        #map[tipus]['psl']=dataMesh
        #cs = mapa.contour(x,y,dataMesh/100,10,colors='k')
        divnorm = colors.Normalize(vmin=10,vmax=40)
        levels = np.linspace(10,40,31)
        #levels2 = np.linspace(10,40,6)
        #cs = mapa.contour(x,y,dataMesh2,colors='k',norm=divnorm, levels=levels2, linewidths=0.05)
        cs2 = mapa.contourf(x,y,dataMesh2,20,cmap='coolwarm',norm=divnorm, levels=levels)

        plt.clabel(cs, fontsize=12, inline=1,fmt='%d')

       
        plt.subplot(1, 3, 3)
        mapa.drawcoastlines(linewidth=0.25)
        # draw lat/lon grid lines every 30 degrees.
        mapa.drawmeridians(np.arange(0,360,15))
        mapa.drawparallels(np.arange(-90,90,15))
        # contour data over the map.
        dataMesh = c[0].reshape(len(lats), len(lons))
        dataMesh2 = dataMesh-273.15
        #map[tipus]['psl']=dataMesh
        #cs = mapa.contour(x,y,dataMesh/100,10,colors='k')
        divnorm = colors.Normalize(vmin=10,vmax=40)
        levels = np.linspace(10,40,31)
        #levels2 = np.linspace(10,40,6)
        #cs = mapa.contour(x,y,dataMesh2,colors='k',norm=divnorm, levels=levels2, linewidths=0.05)
        cs2 = mapa.contourf(x,y,dataMesh2,20,cmap='coolwarm',norm=divnorm, levels=levels)

        
        #plt.colorbar(cs2, fraction=0.046,pad=0.04)
        plt.clim(10,40)
        plt.clabel(cs, fontsize=12, inline=1,fmt='%d')
        plt.tight_layout()
        
        plt.show() 

        """
        values=[]
        for key,value in res.items():
            values.append(value)
        print(values)
        #values= np.concatenate(values)   
        num_dies=3
        
        a = np.array_split(np.array(values[0]),3)
        print(a)
        #for j in range(num_dies):
            #def split(a,n):
                #k,m =divmod(len(a),n)
                #return (a[i * k + min(i,m):(i+1) * k + min(i+1,m)] for i in range(n))
        """  


    return map,np.array(lats),np.array(lons)


def escriu_csv(values):
    if os.path.exists("t_850.csv"):
        os.remove("t_850.csv")
    if os.path.exists("z_500.csv"):    
        os.remove("z_500.csv")  
    if os.path.exists("rh_700.csv"):
        os.remove("rh_700.csv")
    if os.path.exists("slp.csv"):
        os.remove("slp.csv")    

    for dies in values.keys():
        for variables in values[dies].keys():
            with open(variables+'.csv', 'a') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(values[dies][variables])     

def find_nearest(valors, value):
    valors = np.asarray(valors)
    id = (np.abs(valors - value)).argmin()
    
    return id


def llegeix_grib_punts(input,punts):
    
    grid=[]
        

    # Llegim els punts de la graella 
    points=[]
    fp = open(punts, 'Ur')
    for line in fp:
        points.append(tuple(line.strip().split(' ')))

    for lat,lon in points:
        idx=find_nearest(LATS,float(lat))
        idy=find_nearest(LONS,float(lon))
        #print(idx,idy,input[idx,idy])
        grid.append(input[idx,idy])
            
    return np.array(grid)



def calcula_centre_graella(punts):
    
    graella=[]
    fp = open(punts, 'Ur')
    for line in fp:
        graella.append(line.strip().split(' '))

    lat_nord=graella[0][0]
    lat_sud=graella[-1][0]
    mig= (float(lat_sud)+float(lat_nord))/2
        
    return mig



def jenkinson_collison_sfc(grid_slp,cenlat):
     
    pi=4*np.arctan(1.)
    deg2rad=pi/180.
           
#   Definim constants de correcio d'escala que dependran de la latitud
#   sobre la qual tenim centrada la graella: cenlat

#      Constant que relaciona la longitut del meridia amb el paralel
#      el 2 apareix perque amb Jenkinson collison tal com l'hem definit,
#      fem anar 2 arcs de paralel per cada arc de meridia pel calcul
#      de cada una de les respectives direccions del vent 

    sfconstant=1/(2*np.cos(deg2rad*cenlat))
    
#     La mateixa constant que abans pero per la vorticitat amb la qual 
#     usem el mateix numero de meridians que de paralels pes calcular
#     les dues direccions

    zsconstant=1/(np.cos(deg2rad*cenlat))
#
#    La vorticitat zonal va multiplicada per uns factors que depenen
#    del sinus de la latitud. Se suposa que aixo es fa per compensar que
#    com mes prop del pol, l'area que hi ha entre dos meridians es menor
#    a mesura que ens acostem al pol i l'hem de compensar. El 2.5 es posa
#    perque es la meitat de la distancia en que estan separats els 
#    paralels
#    
    zwconst1=np.sin(deg2rad*cenlat)/np.sin(deg2rad*(cenlat-2.5))
    zwconst2=np.sin(deg2rad*cenlat)/np.sin(deg2rad*(cenlat+2.5))

# Calcularem el Jenkinson-Collison per cada un dels dies i ho posarem en un diccionari

    classificacio={}

   
        # Treballem amb hPa
    pres=np.array(grid_slp)/100.
        
#     Calculem el vent a partir de les diferencies de pressio
    w=(1./4.)*(pres[6]+2*pres[7]+pres[8])-(1./4.)*(pres[0]+2*pres[1]+pres[2])
    s=sfconstant*((1./4.)*(pres[2]+2*pres[5]+pres[8])-(1./4.)*(pres[0]+2*pres[3]+pres[6])) 

#     Calculem l'angle i la força del flux        
    d=np.arctan(w/s)*(180/pi)
    f=np.sqrt(pow(w,2)+pow(s,2)) 
        
#    Calcul de la vorticitat de cisalladura a partir de les diferencies  
#    de vent al llarg de la vertical
        
    zw=zwconst1*((pres[6]+2*pres[7]+pres[8])-(pres[3]+2*pres[4]+pres[5]))-  \
        zwconst2*((pres[3]+2*pres[4]+pres[5])-(pres[0]+2*pres[1]+pres[2]))       
        
    zs=zsconstant*(0.25*(pres[2]+2*pres[5]+pres[8])-0.25*(pres[1]+2*pres[4]+pres[7])- \
                    0.25*(pres[1]+2*pres[4]+pres[7])+0.25*(pres[0]+2*pres[3]+pres[6]))
        
    z=zw+zs
        
        # Fluxos zonals
        
    if (d>=-22.5 and d<22.5):
        if (s>0):
            direccio='S'
        else:
            direccio='N'
    if (d>=22.5 and d<67.5):
        if (w>0):
            direccio='SW'
        else:
            direccio='NE'
    if (d>=67.5 and d<=112.5):
        if (w>0):
            direccio='W'
        else:
            direccio='E'
    if (d>=-67.5 and d<-22.5):
        if (w>0):
            direccio='NW'
        else:
            direccio='SE'
    if (d>=-112.5 and d<-67.5):
        if (w>0):
            direccio='W'
        else:
            direccio='E'                
    
    if (np.abs(z)<f):
        tipus=direccio
    
    if (np.abs(z)>2*f):
        if (z>0):
            tipus='C'
        else:
            tipus='A'
    if (np.abs(z)>f and np.abs(z)<2*f):
        if (z>0):
            tipus='C'+direccio
        else:
            tipus='A'+direccio
    if (f<6 and np.abs(z)<6):
        tipus='U'        
        
    classificacio=tipus

    return classificacio  

def jenkinson_collison_500(grid_slp,cenlat):
     
    pi=4*np.arctan(1.)
    deg2rad=pi/180.
           
#   Definim constants de correcio d'escala que dependran de la latitud
#   sobre la qual tenim centrada la graella: cenlat

#      Constant que relaciona la longitut del meridia amb el paralel
#      el 2 apareix perque amb Jenkinson collison tal com l'hem definit,
#      fem anar 2 arcs de paralel per cada arc de meridia pel calcul
#      de cada una de les respectives direccions del vent 

    sfconstant=1/(2*np.cos(deg2rad*cenlat))
    
#     La mateixa constant que abans pero per la vorticitat amb la qual 
#     usem el mateix numero de meridians que de paralels pes calcular
#     les dues direccions

    zsconstant=1/(np.cos(deg2rad*cenlat))
#
#    La vorticitat zonal va multiplicada per uns factors que depenen
#    del sinus de la latitud. Se suposa que aixo es fa per compensar que
#    com mes prop del pol, l'area que hi ha entre dos meridians es menor
#    a mesura que ens acostem al pol i l'hem de compensar. El 2.5 es posa
#    perque es la meitat de la distancia en que estan separats els 
#    paralels
#    
    zwconst1=np.sin(deg2rad*cenlat)/np.sin(deg2rad*(cenlat-2.5))
    zwconst2=np.sin(deg2rad*cenlat)/np.sin(deg2rad*(cenlat+2.5))

# Calcularem el Jenkinson-Collison per cada un dels dies i ho posarem en un diccionari

    classificacio={}

    
    # Treballem amb dam
    pres=np.array(grid_slp)/10.
    
#     Calculem el vent a partir de les diferencies de pressio
    w=(1./4.)*(pres[6]+2*pres[7]+pres[8])-(1./4.)*(pres[0]+2*pres[1]+pres[2])
    s=sfconstant*((1./4.)*(pres[2]+2*pres[5]+pres[8])-(1./4.)*(pres[0]+2*pres[3]+pres[6])) 

#     Calculem l'angle i la forsa del flux        
    d=np.arctan(w/s)*(180/pi)
    f=np.sqrt(pow(w,2)+pow(s,2)) 
    
#    Calcul de la vorticitat de cisalladura a partir de les diferencies  
#    de vent al llarg de la vertical
    
    zw=zwconst1*((pres[6]+2*pres[7]+pres[8])-(pres[3]+2*pres[4]+pres[5]))-  \
        zwconst2*((pres[3]+2*pres[4]+pres[5])-(pres[0]+2*pres[1]+pres[2]))       
    
    zs=zsconstant*(0.25*(pres[2]+2*pres[5]+pres[8])-0.25*(pres[1]+2*pres[4]+pres[7])- \
                    0.25*(pres[1]+2*pres[4]+pres[7])+0.25*(pres[0]+2*pres[3]+pres[6]))
    
    
    
    z=zw+zs
    
    # Fluxos zonals
    
    if (d>=-22.5 and d<22.5):
        if (s>0):
            direccio='S'
        else:
            direccio='N'
    if (d>=22.5 and d<67.5):
        if (w>0):
            direccio='SW'
        else:
            direccio='NE'
    if (d>=67.5 and d<=112.5):
        if (w>0):
            direccio='W'
        else:
            direccio='E'
    if (d>=-67.5 and d<-22.5):
        if (w>0):
            direccio='NW'
        else:
            direccio='SE'
    if (d>=-112.5 and d<-67.5):
        if (w>0):
            direccio='W'
        else:
            direccio='E'                
#
#  El llindar per cassos anticiclonis i cassos ciclonics
#  el fem diferent, en el cas ciclonic som menys retrictius
#  i amb poca corbatura ja ho considerem ciclonic. Amb 
#  anticiclo agafem el mateix criteri que J&C original.
#

    # Cas ciclonic
    if (z>0):
        if (np.abs(z)<=(3./8.)*f):
            tipus=direccio
        tipus=direccio

    # Cas anticiclonic
    if (z<=0):
        if (np.abs(z)<=(4./3.)*f):
            tipus=direccio

    
    if (np.abs(z)>6*f):
        if (z>0):
            tipus='C'
        else:
            tipus='A'

    # Cas ciclonic
    if (z>0):
        if (np.abs(z)>(3./8.)*f and np.abs(z)<6*f):
            tipus='C'+direccio
    #Cas anticiclonic
    if (z<=0):
        if (np.abs(z)>(4./3.)*f and np.abs(z)<6*f):
            tipus='A'+direccio  
        
    #Indeterminat
    if (f<6 and np.abs(z)<6):
        tipus='U'        
        
    classificacio=tipus

    return classificacio    


def combinacio(tipus_sfc,tipus_500):
    
#
#  En aquesta subrutina segons el resultat de J&C en sfc i en 500 mb elegirem un dels tipus
#  de Martin Vide, tal com he descrit en la memoria del 2014 del doctorat. Respecte a la classificacio 
#  del doctorat he refet alguns tipus: 
#    TIP11 (Gota Freda al SW) ha estat assimiiat a TIP09 (Adveccio del SW)
#    TIP13 (Baixa Termia) i TIP14 han estat assimilats a un sol tipus
#    TIP16 (Anticiclo Termic) ha estat assimilat al anticiclo normal (TIP13)
# Aixo ha suposat una reduccio i posterior reordenament dels tipus. Ara hi haura 13 tipus. 
#
    tipus='kk'
    classificacio={}
  
    tipus1=tipus_sfc
    tipus2=tipus_500
    #Cas Anticiclo en superficie
    if (tipus1=='A'):
        if (tipus2[0]=='C'):
            if (tipus2 == 'C'):
                # Seria Anticiclo termic pero ho assimilarem a Anticiclo TIP15
                tipus='TIP13'
            elif (tipus2=='CSW' or tipus2=='CW' or tipus2=='CNW'):
                # Adveccio del W anticiclonica
                tipus='TIP02'
            else:
                # Anticiclo
                tipus='TIP13'
        elif (tipus2[0] == 'A'):
            if (tipus2 == 'A'):
                # Anticiclo
                tipus='TIP13'
            elif (tipus2=='ASW' or tipus2=='AW' or tipus2=='ANW'):
                # Adveccio del W anticiclonica
                tipus='TIP02'
            else:
                # Anticiclo
                tipus='TIP13'
        else:
            if (tipus2=='SW' or tipus2=='W' or tipus2=='nw'):
                # Adveccio del W anticiclonica
                tipus='TIP02'
            else:
                # Anticiclo
                tipus='TIP13'      
                
    # Cas anticiclo del E o be del SE (AE o ASE)        
    
    elif (tipus1=='AE' or tipus1=='ASE'):
        if (tipus2[0]=='C'):
            # Adveccio del E amb gota freda
            tipus='TIP07'
        else:
            # Adveccio del E
            tipus='TIP06'

    # Cas anticiclo del S (AS)
    
    elif (tipus1 == 'AS'):
        # Adveccio del S
        tipus='TIP08'
        
    # Cas anticiclo del SW (ASW)
    
    elif (tipus1 == 'ASW'):
        # Adveccio del SW
        tipus='TIP09'
    
    # Cas anticiclo del W (AW)
    
    elif (tipus1 == 'AW'):
        # Adveccio del W
        tipus='TIP02'
        
    # Cas anticiclo del NW
    
    elif (tipus1 == 'ANW'):
        # Adveccio del NW
        tipus = 'TIP03'
    
    # Cas anticiclo del N
    
    elif (tipus1 == 'AN'):
        # Adveccio del N
        tipus = 'TIP04'       
            
    # Cas anticiclo del NE
    
    elif (tipus1 == 'ANE'):
        # Adveccio del NE
        tipus = 'TIP05'  
    
    # Cas ciclo (C)
    
    elif (tipus1 == 'C'):
        if (tipus2[0]=='C'):
            # Ciclo
            tipus='TIP11'
        elif (tipus2[0]=='A'):
            if (tipus2 == 'AW' or tipus2 == 'ASW' or tipus2 == 'AE' or tipus2 == 'ASE'):
                # Tipus indeterminat: panta barometric o baixa superficial
                tipus='TIP12'
            else:
                # Ciclo
                tipus='TIP11'
        elif (tipus2 == 'U'):
            # Tipus indeterminat: panta barometric o baixa superficial
            tipus='TIP12'
        else:
            if (tipus2 == 'W' or tipus2 == 'SW' or tipus2 == 'E' or tipus2 == 'SE'):
                # Tipus indeterminat: panta barometric o baixa superficial
                tipus='TIP12'
            else:
                # Ciclo
                tipus='TIP11'
        
    # Cas ciclo del Est i del Sud-Est (CE i CSE)
    
    elif (tipus1 == 'CE' or tipus1 == 'CSE'):
        if (tipus2[0] == 'C'):
            # Adveccio del E amb gota freda
            tipus='TIP07'
        else:
            # Adveccio del E
            tipus='TIP06'
    
    # Cas ciclo del Sud (CS)
    
    elif (tipus1 == 'CS'):
        # Adveccion del S
        tipus ='TIP08'

    # Cas ciclo del SE (CSW)
    
    elif (tipus1 == 'CSW'):
        # Adveccio del SW
        tipus='TIP09'
        
    # Cas ciclo del W (CW)
    
    elif (tipus1 == 'CW'):
        if (tipus2[0] == 'C' ):
            # Solc
            tipus='TIP10'
        else:
            # Adveccio W
            tipus='TIP01'
                
    # Cas cilo del NW (CNW)  
    
    elif (tipus1 == 'CNW'):
        # Adveccio del NW
        tipus='TIP03'
        
    # Cas ciclo del N
        
    elif (tipus1 == 'CN'):
        #Adveccio N
        tipus='TIP04'
        
    # Cas ciclo del N
        
    elif (tipus1 == 'CNE'):
        # Adveccio NE
        tipus='TIP05'

    # Cas advectiu del E i del SE

    elif (tipus1 == 'E' or tipus1 == 'SE'):
        if (tipus2[0] == 'C'):
            # Adveccio de E amb gota freda
            tipus='TIP07'
        else:
            # Adveccio de E
            tipus='TIP06'
            
    # Cas advectiu del S 
    
    elif (tipus1 == 'S'):
        tipus='TIP08'
        
    # Cas advectiu del SW
    
    elif (tipus1 == 'SW'):
        tipus='TIP09'
        
    # Cas advectiu del W       
    
    elif (tipus1 == 'W'):
        tipus='TIP01'
    
    # Cas advectiu del NW
    
    elif (tipus1 == 'NW'):
        tipus='TIP03'
        
    # Cas advectiu del N
    
    elif (tipus1 == 'N'):
        tipus='TIP04'
    
    # Cas advectiu del NE
    
    elif (tipus1 == 'NE'):
        tipus='TIP05'
        
    # Cas indeterminat
    
    elif (tipus1 == 'U'):
        if (tipus2[0] == 'C'):
            if (tipus2 == 'CS' or tipus2 == 'CSW' or tipus2 == 'CSE'):
                # Cas de gota freda al SW, l'assimilarem al tipus Adveccio del SW: TIP09
                tipus='TIP09'
            else:
                # Solc
                tipus='TIP10'
        else:
            # Tipus indeterminat: panta barometric o baixa superficial
            tipus='TIP12'
    
    classificacio=tipus    

    return classificacio




if __name__ == '__main__':
    
    days_file="Episodes_ERA5.csv"
    # ERA_40, Interim, Operacional
    TYPE_GRD='Interim'
    
    #Llegim els episodis i els posem en una llista de datetime
    episodis=llegeix_episodis(days_file)
    print ("Numero d'episodis:", len(episodis))
    
    # Preparem les dades necessaries (Era, Interim,.... etc)
    #read_grib(episodis)
        
    # Llegim les variables dels fitxer grib generats
    print ("Llegim les dades dels fitxers corresponents...")
    lat,lon,values=llegeix_grib(episodis)
    print ("Escrivim les dades en format csv per ser usades...")
    #escriu_csv(values)

    # Construim un numpy array amb el qual calcularem les EOF de les 
    # variables que seleccionem: rh_700,z_500,slp,t_850 poden ser mes d'una
    #VARIABLES=["slp","z_500","t_850"]
    VARIABLES=["msl","z","mx2t"]
    NUM_POINTS = None
    PCS_TOTAL=np.array([])
    llista_entrada=[]
    for VAR in VARIABLES:
        matriu_dades=converteix_matriu(values,VAR)
        #print (matriu_dades.shape)
        #print (matriu_dades)

        print ("Fent EOF....")
        # normalize the data attributes
        scaler = preprocessing.StandardScaler()
        scaler_var = scaler.fit(matriu_dades)
        matriu_norm=scaler_var.transform(matriu_dades)
        ''''
        if NUM_POINTS is None:
            NUM_POINTS= matriu_dades.shape[1]
        print ("El numero de punts de grid es: ",NUM_POINTS)
        '''
        llista_entrada.append(matriu_norm)
    matriu_entrada=np.concatenate(llista_entrada,axis=1)
    
    pass
        #print PCS.shape,EOF.shape

    #print ("Calculant varimax....")
    # Trasposem per tenir les columnes com a "vectors propis"  
    
    #rotated_EOF=varimax(EOF[0:NUM_PCS,:].T)
    #print rotated_EOF.shape, EOF[0:NUM_PCS,:].shape
    
    print ("Dibuixant les EOF corresponent...")
    EOF,PCS,PROJECTAT,NUM_PCS=calcula_EOF(matriu_entrada,lat)
    #plot_dades(NUM_PCS[:,0],lat,lon)
    plot_dades(EOF[0:NUM_PCS,:],lat,lon)
    #plot_dades(PCS[:,0],lat,lon)
    #plot_dades(PCS[:,1],lat,lon)
    #plot_dades(PCS[:,2],lat,lon)
    #plot_dades(PCS[:,3],lat,lon)

    print ("Dibuixant els composites dels dies amb més senyal per cada PCA")
    #MAPES,LATS,LONS=plot_tipus(CLASS_DAYS,values,lat,lon,VARIABLES)

    EOFS_DIES, PCS = busca_maxim(EOF, num=NUM_PCS)
    #print(EOFS_DIES)

    #print(MAXIMS.shape)

    MAPES,LATS,LONS=plot_composites(pca=PCS,values=values,lats=lat,lons=lon)

    
        
    #PCS_TOTAL=np.concatenate((PCS_TOTAL,PCS),axis=1)  if PCS_TOTAL.size else PCS
    

    #Calculant els clusters
    #print ("Fem clusters als scores obtinguts....")
    #LABELS=calcula_clusters(PCS_TOTAL)
    #print (LABELS.shape, PCS_TOTAL.shape)
    
    #Analisi discriminant
    #print ("Fem anàlisi discriminant per reclassificar...")
    #NUM_PCS=NUM_PCS*2
    #CLASSIF,CLASS_DAYS=analisi_discriminant(PCS_TOTAL,LABELS,NUM_PCS)

    # Fem els composites dels diferents grups
    #print ("Fem els composites per cada un dels grups...")
    #MAPES,LATS,LONS=plot_tipus(CLASS_DAYS,values,lat,lon,VARIABLES)

    '''
    print('KEYS :',MAPES.keys())

    for TYPE in MAPES.keys():

        # CALCULANT JENKINSON AND COLLISON PER CADA UN DE LES PCA
        np.seterr(divide='ignore')
        PUNTS1 = './data/list_points_mslp'
        PUNTS2 = './data/list_points_500mb'
        # Calculem la latitud central que despres s'usara per J&C
        CENLAT= calcula_centre_graella(PUNTS1)
        slp=MAPES[TYPE]['psl']
        GRID_SLP = llegeix_grib_punts(slp,PUNTS1)
        z500=MAPES[TYPE]['z_500']
        GRID_500 = llegeix_grib_punts(z500,PUNTS2)

        TIPUS_SFC = jenkinson_collison_sfc(GRID_SLP,CENLAT)
        TIPUS_500 = jenkinson_collison_500(GRID_500,CENLAT)
        CLASSIFICACIO= combinacio(TIPUS_SFC,TIPUS_500)
        #print ("%s;%s;%s;%s" %(TIPUS_SFC,TIPUS_500,CLASSIFICACIO))
        print (TIPUS_SFC,TIPUS_500,CLASSIFICACIO)
    '''
        
        
     
def troba_coordenades(psl):
    slp = np.asarray(psl)
    idx = (np.abs(array - value)).argmin()
    return array[idx]    
pass
