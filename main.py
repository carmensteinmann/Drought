
" Main to check code"

import sys

# run drought.py before

""" set path dynamic """
#sys.path.append(r'/Users/carmensteinmann/Desktop/ETH/Climada Project/climada_python-master')
sys.path.append(r'/Users/carmensteinmann/Desktop/ETH/Climada Project/climada_python-master')

latmin = -45
latmax = -10
lonmin = 110
lonmax = 160
#timetoplot = 200308 #warning: first three timesteps are empty """yyyymm""" 
# if intensity definition = 1, then in min, else it is the sum of the events
intensity_definition = 1




#file_path_spei = r'/Users/carmensteinmann/Desktop/ETH/Climada Project/Drought_Data/spei06.nc'
file_path_spei =  r'/Users/carmensteinmann/Desktop/ETH/Climada Project/Drought_Data/spei06.nc'


d = Drought()
d.setArea(latmin, lonmin, latmax, lonmax)
d.setPath(file_path_spei)
#d.setTimeInterval(interval)

new_haz = d.setup()


new_haz.plot_intensity(event='2003')

#plt.ylim((25,250))


