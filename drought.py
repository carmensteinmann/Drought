"""
Define Drought class.
"""
import time
from datetime import datetime



import logging
import numpy as np
import xarray as xr
import pandas as pd
from scipy import sparse
#from numpy import array

from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh
import matplotlib.pyplot as plt
#
#import cartopy
#import cartopy.crs as ccrs
#import matplotlib.pyplot as plt
#from matplotlib.axes import Axes
#from cartopy.mpl.geoaxes import GeoAxes
#GeoAxes._pcolormesh_patched = Axes.pcolormesh


from climada.hazard.base import Hazard
from climada.hazard.centroids.base import Centroids

from climada.hazard.tag import Tag as TagHazard
#from climada.util.files_handler import get_file_names, to_list

LOGGER = logging.getLogger(__name__)

HAZ_TYPE = 'DR'
""" Hazard type acronym Drought """


class Drought(Hazard):
    """Contains drought events.

    Attributes:
        SPEI (float): Standardize Precipitation Evapotraspiration Index


    """
    intensity_thres = 2
    """ intensity threshold for storage """

    vars_opt = Hazard.vars_opt.union({'spei'})
    """Name of the variables that aren't need to compute the impact."""

    def __init__(self):
        """Empty constructor. """
        Hazard.__init__(self, HAZ_TYPE)
        self.spei = np.array([], int)

    def setArea(self, latmin, lonmin, latmax, lonmax):
        self.latmin = latmin
        self.lonmin = lonmin
        self.latmax = latmax
        self.lonmax = lonmax


    def setPath(self, path):
        self.path = path



    def __readIndicesSpei(self, dataset):

        #self.__getLatLonTimeVector(dataset)

        lat_total = dataset.lat.data
        lon_total = dataset.lon.data
        index_lon = np.where(np.logical_and(lon_total >= self.lonmin, lon_total <= self.lonmax))[0]
        index_lat = np.where(np.logical_and(lat_total >= self.latmin, lat_total <= self.latmax))[0]

        #spei_matrix = dataset.spei[:, index_lat[0]:index_lat[len(index_lat)-1],index_lon[0]:index_lon[len(index_lon)-1]].data
        lat_vector = dataset.lat[index_lat[0]:index_lat[len(index_lat)-1]].data
        lon_vector = dataset.lon[index_lon[0]:index_lon[len(index_lon)-1]].data
        self.time_vector = dataset.time.data
        self.lat_vector = lat_vector
        self.lon_vector = lon_vector

        spei_matrix = dataset.spei[:, index_lat[0]:index_lat[len(index_lat)-1],index_lon[0]:index_lon[len(index_lon)-1]].data


        #time_vector = dataset.time[latmin:latmax,lonmin:lonmax].data
#        """Zeitvektor umbauen"""
#        time_to_plot = 200308
#        time_vector = dataset.time.data
#        t = pd.to_datetime(time_vector)
#        year = t.year
#        month = t.month
#        timenew = year*100+month
#        timenp = array(timenew.tolist())
#        index_time = np.where(timenp == time_to_plot)[0]
#
#        time_vector = dataset.time.data
#        ax = plt.axes(projection=ccrs.PlateCarree())
#        ax.coastlines()
#        ax.set_extent([lon_vector[0], lon_vector[len(lon_vector)-1], lat_vector[0], lat_vector[len(lat_vector)-1]],ccrs.PlateCarree())
#        plt.contourf(lon_vector, lat_vector, spei_matrix[index_time[0],:,:],60,transform=ccrs.PlateCarree(),cmap='Spectral')
#        ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
#        cbar = plt.colorbar()
#        cbar.set_label('SPEI')
#        #matplotlib.pyplot.title(time_to_plot)
#        plt.show()
#
        return spei_matrix


    def setup(self):

        dataset = xr.open_dataset(self.path)
        #spei_3D = self.__readIndicesSpei(dataset)
        #spei_2D = self._TraslateMatrix(spei_3D)

        spei_3D = self.__readIndicesSpei(dataset)
        spei_2D = self.__traslateMatrix(spei_3D)

        #spei_2D_sparse = sparse.csr_matrix(spei_2D)

        #intensity_matrix_max, start_date_matrix, length_matrix = self.__getIntensityFrom2D(spei_2D)
        intensity_matrix_min = self.__getIntensityFrom2D(spei_2D, 1)
        self.hazard_def(intensity_matrix_min)
        #intensity_matrix_max_sparse = sparse.csr_matrix(intensity_matrix_max)


        #self._event_plot(self, 4, intensity_matrix_max_sparse)
        #self.plot_intensity(5)



        return self



    def __traslateMatrix(self, spei_3D):
        """ return hazard intensity as a simple threshold on the SPEI values
        Parameters: see read_indices_spei, just call before
        Returns: matrix
        sparse.csr_matrix
        """

        intensity_thres = -1

        n_centroids = spei_3D.shape[1] * spei_3D.shape[2]
        n_timesteps = spei_3D.shape[0]
        spei_2D = np.zeros((n_timesteps, n_centroids))


        for i in range(n_timesteps):

            one_event_1D = spei_3D[i, :, :]

              # get rid of nan's
            nan_pos = np.isnan(one_event_1D)
            one_event_1D[nan_pos] = 0

            # apply threshold
            non_drought_pos = np.where(one_event_1D > intensity_thres)
            one_event_1D[non_drought_pos] = 0



            #  n_times = spei_matrix.shape[0]


            one_event_array = one_event_1D.reshape(n_centroids)

            spei_2D[i,:] = one_event_array
        #spei_2D.transpose()
        #intensity_matrix = intensity_max(spei_2D)

        return spei_2D

    #@staticmethod_ no new hazard
    def hazard_def(self, intensity_matrix):
        """ return hazard set
        Parameters: see intensity_from_spei
        Returns:
            Drought, full hazard set
            check using new_haz.check()
        """
        #new_haz = Drought()
        self.tag = TagHazard(HAZ_TYPE, 'TEST')

        self.intensity = sparse.csr_matrix(intensity_matrix)
        #new_haz.intensity=new_haz.intensity_mean(spei_matrix, time_vector,lat_vector, lon_vector)

        self.units = 'SPEI'

        # fill centroids th bad way (there must be a code like grid...)
        lat_2d = np.zeros([self.lat_vector.shape[0], self.lon_vector.shape[0]])
        lon_2d = np.zeros([self.lat_vector.shape[0], self.lon_vector.shape[0]])
        n_centroids = self.lat_vector.shape[0] * self.lon_vector.shape[0]
        for lat_i in range(0, self.lat_vector.shape[0]):
            for lon_i in range(0, self.lon_vector.shape[0]):
                lat_2d[lat_i, lon_i] = self.lat_vector[lat_i]
                lon_2d[lat_i, lon_i] = self.lon_vector[lon_i]

        lon_1d = lon_2d.reshape(n_centroids,)
        lat_1d = lat_2d.reshape(n_centroids,)

        self.centroids.coord = np.zeros((n_centroids, 2))
        self.centroids.coord[:, 0] = lat_1d
        self.centroids.coord[:, 1] = lon_1d
        self.centroids.id = np.arange(n_centroids)

        self.event_id = np.arange(1, self.n_years+1,1)
        # frequency set when all eventsavailable
        #self.frequency = np.array([1])
        #per default equal to event_id
        name_list = []
        time = pd.to_datetime(self.time_vector)

        for i in range(13, len(time), 12):
            name_list.append(str(time[i].year))
        self.event_name = name_list
        self.frequency = np.ones(self.n_years)/self.n_years

        self.fraction = self.intensity.copy()
        self.fraction = self.intensity.copy().tocsr()
        self.fraction.data.fill(1)
        # store date of start
#        new_haz.date = np.array([dt.datetime(
#            track.time.dt.year[0], track.time.dt.month[0],
#            track.time.dt.day[0]).toordinal()])

        #new_haz.date = _datetime64_toordinal(self.time_vector)
        self.date = np.arange(1, self.n_years+1,1)
        #new_haz.orig =

        self.check()
        return self


    def __getIntensityFrom2D(self, spei_2D, intensity_definition):
        """Parameters: the 2D matrix called 'spei_2D' defined in
        intensity_from_spei, which containes every time and spacial resolution
        pixel with either the SPEI value or zero if the pixel value doesn't
        reach the threshold value.
        Returns: matrix
        The matrix with the intensity of every event (maximum one per year).
        The intensity is simply the maximum value for
        the event."""

        #time_steps = spei_2D.size([0])
        #n_years = self.time_vector[0].year - self.time_vector[len(self.time_vector)-1].year
        n_centroids = spei_2D.shape[1]



        #first_year = time_vector[0].year
        time = pd.to_datetime(self.time_vector)
        first_year = time[0].year + 1

        first_month = time[0].month

        #index_offset to get index of january of first year considered
        index_offset = 12 - first_month + 1


        if time[0].month > 10:
            first_year += 1
            index_offset += 12


        last_year = time[len(time)-1].year

        if time[len(time)-1].month < 9:
            last_year -= 1




#        if (time.year == first_year && time.month == 8) in time:
#            first_year = time[1].year

        n_years = last_year - first_year + 1 # the first year not counted because of hydrological years
        years_vector = np.arange(first_year, last_year)
        self.n_years = n_years

        intensity_min_matrix = np.zeros((n_years, n_centroids))
        intensity_sum_matrix = np.zeros((n_years, n_centroids))

        #date_start_matrix = np.zeros((n_years, n_centroids))
        #length_array = np.zeros((n_years, n_centroids))


        time = time[index_offset - 3: index_offset + 12*n_years - 3]

        list_events_info = list() #save start end of the event, minimum SPEI value and sum



        for pixel in range(n_centroids):

            array_time_in_centroid = spei_2D[index_offset - 3: index_offset + 12*n_years - 3, pixel]

#            idx_oct = index_offset + 12*(year - first_year)
#            idx_sept = idx_oct + 11

            #min_1D_array = np.min(spei_2D[idx_oct: idx_sept, pixel]
            event = 0
            min_spei = 0
            sum_spei = 0
            year_offset = 0
            min_spei_offset = 0
            list_events_info.clear()

            #create a list with every event exeeding the threshold
            for time_idx in range(len(array_time_in_centroid)):



                if array_time_in_centroid[time_idx] == 0:
                    if event:
                        event = 0
                        list_events_info.append([start_time, end_time, min_spei, sum_spei])
                        min_spei = 0
                        sum_spei = 0



                else:
                    if event:
                        end_time = time[time_idx]
                        sum_spei += array_time_in_centroid[time_idx]
                        if array_time_in_centroid[time_idx] < min_spei:
                            min_spei = array_time_in_centroid[time_idx]

                    else:
                        start_time = time[time_idx]
                        end_time = time[time_idx]
                        min_spei = array_time_in_centroid[time_idx]
                        sum_spei = array_time_in_centroid[time_idx]

                        event = 1


            #intensity_min = __getIntensityFromlist(array_events_info, n_years, n_centroids, time, pixel):
            # from list get one hazard per jear (would be better to use a separate method)
            intensity_min_array = np.zeros((n_years))
            intensity_sum_array = np.zeros((n_years))
            date_start_array = np.zeros((n_years))
            date_end_array = np.zeros((n_years))

            year_offset = first_year
            min_spei_offset = 0


            for idx_event in range(0, len(list_events_info)):



                min_spei = list_events_info[idx_event][2]
                sum_spei = list_events_info[idx_event][3]


                year_start = list_events_info[idx_event][0].year
                month_start = list_events_info[idx_event][0].month

                if month_start > 10:
                    year_start +=1

                idx_year =  np.where(years_vector == year_start)

                year_end = list_events_info[idx_event][1].year
                month_end = list_events_info[idx_event][1].month

                if year_offset == year_start:
                    if min_spei < min_spei_offset:
                        intensity_min_array[idx_year] = min_spei
                        intensity_sum_array[idx_year] = sum_spei

                        min_spei_offset = min_spei

                else:
                    intensity_min_array[idx_year] = min_spei
                    intensity_sum_array[idx_year] = sum_spei

                    min_spei_offset = min_spei
                #date_start_array[idx_year] =  int(time.mktime(self.time_vector[idx_event].now().timetuple()))
#                date_end_array[idx_year] =  array_events_info[idx_event][1]
                #date_start_array[idx_year] = time.mktime(array_events_info[idx_event][0].timetuple())






                year_offset = year_start





            intensity_min_matrix[:, pixel] =  intensity_min_array
            intensity_sum_matrix[:, pixel] =  intensity_sum_array
            #date_start_matrix[:, pixel] =  date_start_array
#            date_end_array[:, pixel] =  date_end_array


        if intensity_definition == 1:
            return intensity_min_matrix
        
        return intensity_sum_matrix
    
    
       #def __getEventsFromList(list, years_vector): 
       #output: intensity_min_array
    












def _datetime64_toordinal(datetime):
    """ Converts from a numpy datetime64 object to an ordinal date.
        See https://stackoverflow.com/a/21916253 for the horrible details. """
    return pd.to_datetime(datetime.tolist()).toordinal()