import pandas as pd
import geopandas as gpd
from apps.business_expansion.module.processing import *

def calculate_pop_all_catchment(catch_name):
    list_distance = []
    list_population = []
    dis = {500:'500m',1000:'1 km',2000:'2 km',3000:'3 km',5000:'5km'}
    for distance in [500, 1000, 2000, 3000, 5000]:
        huff = pd.read_parquet(r'.\apps\business_expansion\data\huff\huff_{}.parquet'.format(distance)).rename(columns = {'id':'index'})
        pop = get_data('Population with Hex')[0].rename(columns = {"population_index":'Population'})
        pop_i = pd.merge(huff[['index',catch_name]], pop, on = 'index', how = 'left')
        pop_i['Total Population'] = round(pop_i['Population']*pop_i[catch_name])
        list_distance.append(str(dis[distance]))
        list_population.append(pop_i['Total Population'].sum())

    df_pop = pd.DataFrame({'Distance':list_distance,
                                'Total Population':list_population
                               })
    return df_pop

