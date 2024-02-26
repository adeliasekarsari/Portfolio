import streamlit as st
from streamlit_option_menu import option_menu
from apps.business_expansion.module.processing import *
from apps.business_expansion.module.huff_model import *
from hydralit import HydraHeadApp
import pandas as pd
import geopandas as gpd
from PIL import Image
import plotly.express as px
import numpy as np
import folium
from streamlit_folium import st_folium


class BEApp(HydraHeadApp):

    def __init__(self, title = '', **kwargs):
        self.__dict__.update(kwargs)
        self.title = title
        
    def run(self):
        with st.sidebar:
            choose = option_menu("Business Expansion", ["Project Description",
                                                    "Data Visualization",
                                                    "Huff Analysis",
                                                    "Business Expansion"],
                                icons=['book','pin-map-fill','person lines fill','book'],
                                menu_icon="app-indicator", default_index=0,
                                styles={
                "container": {"padding": "5!important", "background-color": "black"},
                "icon": {"color": "White", "font-size": "25px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "#6BA1D1"},
            }
            
            )
        if choose == "Project Description":
            st.markdown("""
            <style>
            .big-font {
                font-size:60px !important;
            }
            .medium-font {
                font-family:sans-serif;
                font-size:18px !important;
                text-align: justify color:White;
            }.medium1-font {
                font-family:sans-serif;
                font-size:80px !important;
                text-align: justify color:White;
            }.small-font {
                font-family:sans-serif;
                font-size:18px !important;
                text-align: justify color:White;
            }
            </style>
            """, unsafe_allow_html=True)


            # Title

            st.markdown("<p class='big-font'>Leverage Your Business! </p>", unsafe_allow_html=True)
            

            st.title("")
            text = """
            In today's competitive business landscape, companies are constantly seeking innovative 
            strategies to expand their operations and reach new markets. Geospatial data analysis 
            offers a powerful tool for businesses to gain insights into market dynamics, consumer 
            behavior, and optimal locations for expansion. This project aims to develop a web application 
            that harnesses the power of geospatial data to facilitate informed decision-making 
            and drive business growth.
            """
            st.markdown(f"<p class='medium-font'>{text}</p>", unsafe_allow_html=True)

        elif choose == 'Data Visualization':
            with st.container():
                col1, col2 = st.columns([0.7,0.3])
                with col1:
                    type_data = st.selectbox('Data :',('Population with Hex','Population in Admin',
                                                    'Building in Hex','POI in Hex','POI by Category'
                                                    ))
                    data, df = get_data(type_data)
                    df = df.fillna(0)
                with col2:
                    if type_data == 'POI in Hex':
                        poi_cat = st.selectbox('Category :',tuple(data.drop(columns = 'index').columns.tolist()))
                    elif type_data == 'POI by Category':
                        poi_cat = st.selectbox('Category :',tuple(data.category.unique().tolist()))
                    elif type_data == 'Population in Admin':
                        poi_cat = st.selectbox('Category :',('Population','Density'))
                    elif type_data == 'Population with Hex':
                        poi_cat = st.selectbox('Category :',('density_index','population_index'))
                    else:
                        poi_cat = None
            
            with st.container():
                col1, col2 = st.columns([0.4,0.6])
                with col1:
                    description = get_title_text(type_data)
                    st.markdown(f"<p class='medium1-font'>{description[0]}</p>", unsafe_allow_html=True)
                    for i in description[1].keys():
                        st.markdown(f"<p class='medium-font'>{i}</p>", unsafe_allow_html=True)
                        st.markdown(f"<p class='small-font'>{description[1][i]}</p>", unsafe_allow_html=True)
                        st.text("")
                    
    
                with col2:
                    base_map = get_map(df, type_data, poi_cat)
                    st_data = st_folium(base_map, 
                                width = 1300,
                                height =600
                                )
                    st.markdown(f"<p class='medium1-font'>Table of {type_data}</p>", unsafe_allow_html=True)
                    st.dataframe(df.drop(columns = 'geometry'), width=1000, height=200)
                    


        elif choose == 'Huff Analysis':
            text = """
            Employing the Huff Model for Gravity Analysis, this study 
                    investigates the attractiveness and accessibility 
                    of malls within Bandung City. By considering factors 
                    such as distance, population size, and the attractiveness of competing malls, 
                    the analysis aims to identify key determinants influencing consumer behavior 
                    and mall patronage patterns. Through this method, planners and stakeholders 
                    can gain insights into the spatial distribution of consumer demand and make 
                    informed decisions regarding retail development and market positioning within the city.

            """
            st.markdown(f"<p class='medium-font'>{text}", unsafe_allow_html=True)
            catchment = gpd.read_parquet(r'./apps\business_expansion/data/catchment_mall.parquet')
            poi = gpd.read_parquet(r'./apps/business_expansion/data/point_mall.parquet')

            # Catchment data
            building_catch = pd.read_parquet(r'./apps\business_expansion/data/data_catchment/building_catchment.parquet')
            poi_catch = pd.read_parquet(r'./apps/business_expansion/data\data_catchment/poi_catchment.parquet')
            pop_catch = pd.read_parquet(r'./apps/business_expansion/data/data_catchment/pop_catchment.parquet')
            poi_category = get_data('POI in Hex')[0]

            # hex data
            hex_ = gpd.read_parquet(r'./apps/business_expansion/data/hex.parquet')
            method = st.selectbox('Anaysis :',('Data','Huff Analysis'))
            col1, col2 = st.columns([0.6,0.4])
            with col1:
                distance = st.select_slider('Select Distance',
                                        options=[500, 1000, 2000, 3000, 5000])
            with col2:
                catch_name = st.selectbox('Catchment :',tuple(['All Catchment']+poi['name_location'].tolist()))
            
            # get data by name
            building_i = building_catch[building_catch['id']==catch_name].reset_index()
            poi_i = poi_catch[poi_catch['id']==catch_name].reset_index()
            pop_i = pop_catch[pop_catch['id']==catch_name].reset_index()
            
            if method == 'Data':
                with st.container():
                    col1, col2 = st.columns([0.6,0.4])
                    with col1:
                        if catch_name == 'All Catchment':
                            catch_i = catchment[catchment['dt']==distance]
                        else:
                            catch_i = catchment[(catchment['id']==catch_name)&(catchment['dt']==distance)]
                        maps = catch_i.explore(tiles = 'cartodb darkmatter', color = 'lightblue')
                        points1 = list(map(tuple, zip(poi.geometry.y, poi.geometry.x)))
                        train_group = folium.FeatureGroup(name="Mall Location").add_to(maps)
                        i = 0
                        for tuple_ in points1:
                            tooltip ="{}".format(poi['name_location'][i]) 
                            icon=folium.Icon(color='lightblue', icon='fa-shopping-cart', icon_color="white", prefix='fa')
                            train_group.add_child(folium.Marker(tuple_, icon=icon,tooltip = tooltip))
                            i+=1
                        folium.LayerControl().add_to(maps)
                        st_data = st_folium(maps, 
                                    width = 1300,
                                    height =600
                                    )
                    with col2:
                        st.markdown("""
                                <style>
                                .medium1-font {
                                    font-family:sans-serif;
                                    font-size:19px !important;
                                    text-align: center color:White;
                                }.medium2-font {
                                    font-family:sans-serif;
                                    font-size:28px !important;
                                    text-align: center color:White;
                                }.small-font {
                                    font-family:sans-serif;
                                    font-size:18px !important;
                                    text-align: justify color:White;
                                }
                                </style>
                                """, unsafe_allow_html=True)
                        st.text("")
                        st.text("")
                        with st.container():
                            colA, colB, colC = st.columns([0.5,0.5,0.5])
                            with colA:
                                st.markdown(f"<p class='medium1-font'>Population in {catch_name}</p>", unsafe_allow_html=True)
                                st.markdown(f"<p class='medium2-font'>{'N/A' if catch_name == 'All Catchment' else pop_i[f'Population_{distance}'][0]} </p>", unsafe_allow_html=True)
                            with colB:
                                st.markdown(f"<p class='medium1-font'>Mean Population</p>", unsafe_allow_html=True)
                                st.markdown(f"<p class='medium2-font'>{round(pop_catch[f'Population_{distance}'].mean())} </p>", unsafe_allow_html=True)
                            with colC:
                                st.markdown(f"<p class='medium1-font'>Max Population</p>", unsafe_allow_html=True)
                                st.markdown(f"<p class='medium2-font'>{round(pop_catch[f'Population_{distance}'].max())} </p>", unsafe_allow_html=True)
                                
                        st.text("")
                        st.text("")
                        with st.container():
                            colA, colB, colC = st.columns([0.5,0.5,0.5])
                            with colA:
                                st.markdown(f"<p class='medium1-font'>Total POI in {catch_name}</p>", unsafe_allow_html=True)
                                st.markdown(f"<p class='medium2-font'>{'N/A' if catch_name == 'All Catchment' else poi_i[f'total poi {distance}'][0]} </p>", unsafe_allow_html=True)
                            with colB:
                                st.markdown(f"<p class='medium1-font'>Mean Total POI</p>", unsafe_allow_html=True)
                                st.markdown(f"<p class='medium2-font'>{round(poi_catch[f'total poi {distance}'].mean())} </p>", unsafe_allow_html=True)
                            with colC:
                                st.markdown(f"<p class='medium1-font'>Max POI</p>", unsafe_allow_html=True)
                                st.markdown(f"<p class='medium2-font'>{round(poi_catch[f'total poi {distance}'].max())} </p>", unsafe_allow_html=True)
                        st.text("")
                        st.text("")        
                        with st.container():
                            colA, colB, colC = st.columns([0.5,0.5,0.5])
                            with colA:
                                st.markdown(f"<p class='medium1-font'>Total Building in {catch_name}</p>", unsafe_allow_html=True)
                                st.markdown(f"<p class='medium2-font'>{'N/A' if catch_name == 'All Catchment' else building_i[f'total_building_{distance}'][0]} </p>", unsafe_allow_html=True)
                            with colB:
                                st.markdown(f"<p class='medium1-font'>Mean Total Building</p>", unsafe_allow_html=True)
                                st.markdown(f"<p class='medium2-font'>{round(building_catch[f'total_building_{distance}'].mean())} </p>", unsafe_allow_html=True)
                            with colC:
                                st.markdown(f"<p class='medium1-font'>Max Building</p>", unsafe_allow_html=True)
                                st.markdown(f"<p class='medium2-font'>{round(building_catch[f'total_building_{distance}'].max())} </p>", unsafe_allow_html=True)

            elif method == 'Huff Analysis':
                color = {'Braga CityWalk': 'Purples',
                        'Bandung Indah Plazaz': 'Blues',
                        'Istana Plaza': 'Greens',
                        'Festival Citylink': 'Oranges',
                        '23 Paskal Shopping Center': 'Reds',
                        'Cihampelas Walk': 'Greys',
                        'Trans Studio Mall Bandung': 'PuRd',
                        "D'Botanica Bandung Mall": 'BuPu',
                        'Miko Mall': 'YlGn',
                        'The Kings Shopping Center': 'BuGn',
                        'Paris Van Java Supermall': 'YlOrBr',
                        'Click Square': 'PuBuGn'}
                huff = pd.read_parquet(r'./apps/business_expansion/data/huff/huff_{}.parquet'.format(distance)).rename(columns = {'id':'index'})
                with st.container():

                    df_huff = pd.merge(hex_, 
                                    huff, 
                                    on = 'index', 
                                    how = 'inner'
                                    ).fillna(0)
                    if catch_name == 'All Catchment':
                        m = hex_.boundary.explore(color = 'grey',
                                                    tiles = 'Cartodb darkmatter',
                                                    style_kwds = {'opacity':0.2},
                                                    )
                        for i in color.keys():
                            df_huff[df_huff['Significant']==i].explore(m = m,
                                            column = i,
                                            cmap = color[i],
                                            style_kwds = {'opacity':0.5},
                                            legend = False
                                            )
                        points1 = list(map(tuple, zip(poi.geometry.y, poi.geometry.x)))
                        train_group = folium.FeatureGroup(name="Mall Location").add_to(m)
                        i = 0
                        for tuple_ in points1:
                            tooltip ="{}".format(poi['name_location'][i]) 
                            icon=folium.Icon(color='darkblue', icon='fa-shopping-cart', icon_color="white", prefix='fa')
                            train_group.add_child(folium.Marker(tuple_, icon=icon,tooltip = tooltip))
                            i+=1
                        folium.LayerControl().add_to(m)
                            
                        st_data = st_folium(m, 
                                    width = 1800,
                                    height =600
                                    )
                    else:
                        m = hex_.boundary.explore(color = 'grey',
                                                    tiles = 'Cartodb darkmatter',
                                                    style_kwds = {'opacity':0.2},
                                                    )
                        df_huff.explore(m=m,
                                        column = catch_name,
                                        cmap = color[catch_name]
                                        )
                        poi_i = poi[poi['name_location']==catch_name].reset_index()
                        points1 = list(map(tuple, zip(poi_i.geometry.y, poi_i.geometry.x)))
                        train_group = folium.FeatureGroup(name="Mall Location").add_to(m)
                        i = 0
                        for tuple_ in points1:
                            tooltip ="{}".format(poi_i['name_location'][i]) 
                            icon=folium.Icon(color='darkblue', icon='fa-shopping-cart', icon_color="white", prefix='fa')
                            train_group.add_child(folium.Marker(tuple_, icon=icon,tooltip = tooltip))
                            i+=1
                        folium.LayerControl().add_to(m)
                        st_data = st_folium(m, 
                                    width = 1800,
                                    height =600
                                    )
                if catch_name == "All Catchment":
                        pass
                else:
                    with st.container():
                    
                        # Calculate population in catchment
                        pop = get_data('Population with Hex')[0].rename(columns = {"population_index":'Population'})
                        pop_i = pd.merge(huff[['index',catch_name]], pop, on = 'index', how = 'left')
                        pop_i['Total Population'] = round(pop_i['Population']*pop_i[catch_name])
                        st.markdown("""
                            <style>
                            .medium1-font {
                                font-family:sans-serif;
                                font-size:19px !important;
                                text-align: center color:White;
                            }.medium2-font {
                                font-family:sans-serif;
                                font-size:28px !important;
                                text-align: center color:White;
                            }.small-font {
                                font-family:sans-serif;
                                font-size:18px !important;
                                text-align: justify color:White;
                            }
                            </style>
                                    
                            """, unsafe_allow_html=True)
                        colA, colB = st.columns([0.4,0.7])
                        with colB:
                            st.markdown(f"<p class='medium2-font'>Probabilities per Hex in Catchment by Huff Model</p>", unsafe_allow_html=True)
                        col0, colA, colB, colC = st.columns([0.5,0.5,0.5,0.5])
                        with colA:
                            st.markdown(f"<p class='medium1-font'>Min. </p>", unsafe_allow_html=True)
                            st.markdown(f"<p class='medium1-font'>{round(huff[catch_name].min(),2)}% </p>", unsafe_allow_html=True)
                        with colB:
                            st.markdown(f"<p class='medium1-font'>Max. </p>", unsafe_allow_html=True)
                            st.markdown(f"<p class='medium1-font'>{round(huff[catch_name].max(),2)}%</p>", unsafe_allow_html=True)
                        with colC:
                            st.markdown(f"<p class='medium1-font'>Median. </p>", unsafe_allow_html=True)
                            st.markdown(f"<p class='medium1-font'>{round(huff[catch_name].median(),2)}%</p>", unsafe_allow_html=True)

                    
                    with st.container():
                        col1, col2 = st.columns([0.5,0.5])
                        with col2:
                            # st.markdown(f"<p class='medium1-font'>Total Population in Catchment by Huff Model</p>", unsafe_allow_html=True)
                            total_pop = calculate_pop_all_catchment(catch_name)
                            colors = ['rgb(158,202,225)',] * 5
                            dis = [500, 1000, 2000, 3000, 5000]
                            colors[dis.index(distance)] = 'crimson'
                            fig = px.bar(total_pop, x='Distance', y='Total Population')
                            fig.update_traces(marker_color=colors, marker_line_color='rgb(8,48,107)',
                                            marker_line_width=1.5, opacity=0.6)
                            fig.update_layout(title_text='Total Population in Catchment')
                            st.plotly_chart(fig, use_container_width=True)
                        with col1:
                                    
                            building_i = building_catch[building_catch['id']==catch_name].set_index('id').T.reset_index()
                            building_i.columns = ['Distance','Total Building']
                            building_i['Distance'] = ['500m','1 km','2 km','3 km','5 km']
                            colors = ['rgb(158,202,225)',] * 5
                            dis = [500, 1000, 2000, 3000, 5000]
                            colors[dis.index(distance)] = 'crimson'
                            fig = px.bar(building_i, x='Distance', y='Total Building')
                            fig.update_traces(marker_color=colors, marker_line_color='rgb(8,48,107)',
                                            marker_line_width=1.5, opacity=0.6)
                            fig.update_layout(title_text='Total Building in Catchment')
                            st.plotly_chart(fig, use_container_width=True)

                    with st.container():
                        df_huff = pd.merge(hex_, 
                            huff, 
                            on = 'index', 
                            how = 'inner'
                            ).fillna(0)
                        poi_i = poi_category[poi_category['index'].isin(df_huff[df_huff['Significant']==catch_name]['index'].tolist())].drop(columns = ['index','total poi']).sum().reset_index().fillna(0)                          
                        poi_i.columns = ['Category','Total POI']
                        poi_i['Total POI'] = poi_i['Total POI'].astype(float)
                        poi_i = poi_i.sort_values('Total POI')
                        poi_i = poi_i[poi_i['Total POI']!=0]
                        fig = px.bar(poi_i, x='Category', y='Total POI')
                        fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                                        marker_line_width=1.5, opacity=0.6)
                        fig.update_layout(title_text='Total POI by Category')
                        st.plotly_chart(fig, use_container_width=True)                     

        elif choose == 'Business Expansion':
            st.markdown("""
                                <style>
                                .medium1-font {
                                    font-family:sans-serif;
                                    font-size:22px !important;
                                    text-align: center color:White;
                                }.medium2-font {
                                    font-family:sans-serif;
                                    font-size:28px !important;
                                    text-align: center color:White;
                                }.small-font {
                                    font-family:sans-serif;
                                    font-size:18px !important;
                                    text-align: justify color:White;
                                }.small1-font {
                                    font-family:sans-serif;
                                    font-size:15px !important;
                                    text-align: justify color:White;
                                }
                                </style>
                                """, unsafe_allow_html=True)
            grid1 = pd.read_parquet(r'./apps/business_expansion/data/poi_grid.parquet')
            grid2 = pd.read_parquet(r'./apps/business_expansion/data/griana_data.parquet')
            df_plot = pd.merge(grid1, grid2, on = 'index').drop(columns = 'geometry')
            grid = gpd.read_file(r'./apps/business_expansion/data/hex.geojson')
            with st.container():
                col1, col2 = st.columns([0.35,0.65])
                with col1:
                    st.markdown(f"<p class='medium1-font'>Data Filtering</p>", unsafe_allow_html=True)
                    with st.container():
                        cola, colb = st.columns([0.1,0.7])
                        with cola:
                            st.markdown(f"<p class='small-font'>POI</p>", unsafe_allow_html=True)
                        with colb:
                            poi_col = st.multiselect(' ', grid1.drop(columns = 'index').columns.tolist())
                    for i in poi_col:
                        data = st.slider(f'select {i}',grid1[i].min(), grid1[i].max(), (grid1[i].min(), grid1[i].max()), 1.0)
                        grid1[i] = np.where((grid1[i]>=data[0])& (grid1[i]<=data[1]), 1.0, 0)
                    data = st.slider('Select Total POI',grid1['total poi'].min(), grid1['total poi'].max(), (grid1['total poi'].min(), grid1['total poi'].max()), 1.0)
                    grid1['total poi'] = np.where((grid1['total poi']>=data[0])& (grid1['total poi']<=data[1]), 1.0, 0)

                    st.text("")
                    st.markdown(f"<p class='small-font'>Population</p>", unsafe_allow_html=True)
                    data = st.slider(f'select range',grid2['population_index'].min(), grid2['population_index'].max(), 
                                     (grid2['population_index'].min(), 
                                      grid2['population_index'].max()), 5.0)
                    grid2['population_index'] = np.where((grid2['population_index']>=data[0])& (grid2['population_index']<=data[1]), 
                                                         1, 0)
                    st.text("")
                    st.markdown(f"<p class='small-font'>Density</p>", unsafe_allow_html=True)
                    data = st.slider(f'select range',grid2['density_index'].min(), grid2['density_index'].max(), 
                                     (grid2['density_index'].min(), 
                                      grid2['density_index'].max()), 5.0)
                    grid2['density_index'] = np.where((grid2['density_index']>=data[0])& (grid2['density_index']<=data[1]), 
                                                         1, 0)
                    st.text("")
                    st.markdown(f"<p class='small-font'>Building</p>", unsafe_allow_html=True)
                    data = st.slider(f'select range',grid2['total building'].min(), grid2['total building'].max(), 
                                     (grid2['total building'].min(), 
                                      grid2['total building'].max()), 5.0)
                    grid2['total building'] = np.where((grid2['total building']>=data[0])& (grid2['total building']<=data[1]), 
                                                         1, 0)
                    griana_scoring = pd.merge(grid1[['index','total poi']+poi_col],
                                              grid2, on = 'index'
                                              ).drop(columns = 'geometry')
                    geo_scoring = pd.merge(grid, griana_scoring, on = 'index')

                    # calculate score
                    griana_scoring['score'] = griana_scoring.drop(columns = 'index').sum(axis = 1)
                    griana_scoring = griana_scoring.sort_values('score', ascending = True)
                    griana_scoring['Class'] = pd.cut(griana_scoring['score'],
                        bins=5,
                        labels=['Acceptable 5', 'Acceptable4','Acceptable 3','Acceptable 2','Acceptable 1'])
                    geo_scoring = pd.merge(geo_scoring, griana_scoring[['index','score','Class']], on = 'index')
                
                with col2:
                    st.markdown(f"<p class='medium1-font'>Data Visualization</p>", unsafe_allow_html=True)
                    dataframe = pd.merge(df_plot[griana_scoring.drop(columns = ['score','Class']).columns.tolist()],
                                          griana_scoring[['index','score','Class']], on = 'index'
                                          )
                    # if st.button('Generate'):
                    m = geo_scoring.explore(column = 'Class',
                                    cmap = 'GnBu',
                                    tiles = 'cartodb darkmatter'
                                    )
                    
                    grid_select = st.selectbox('Grid Selected',['All','Acceptable 1', 'Acceptable 2','Acceptable 3','Acceptable 4','Acceptable 5'])
                    if grid_select == 'All':
                        st_data = st_folium(m, 
                                width = 1800,
                                height =600
                                )
                        col11, col12, col13, col14 = st.columns([0.5,0.5,0.5,0.5])
                        with col11:
                            st.markdown(f"<p class='medium1-font'>Avg. Population</p>", unsafe_allow_html=True)
                            st.markdown(f"<p class='small-font'>{round(dataframe['population_index'].mean())}</p>", unsafe_allow_html=True)
                        with col12:
                            st.markdown(f"<p class='medium1-font'>Avg. Density</p>", unsafe_allow_html=True)
                            st.markdown(f"<p class='small-font'>{round(dataframe['density_index'].mean())}</p>", unsafe_allow_html=True)
                        with col13:
                            st.markdown(f"<p class='medium1-font'>Avg. Building</p>", unsafe_allow_html=True)
                            st.markdown(f"<p class='small-font'>{round(dataframe['total building'].mean())}</p>", unsafe_allow_html=True)
                        with col14:
                            st.markdown(f"<p class='medium1-font'>Avg. POI</p>", unsafe_allow_html=True)
                            st.markdown(f"<p class='small-font'>{round(dataframe['total poi'].mean())}</p>", unsafe_allow_html=True)
                        st.dataframe(dataframe, width=1500, height=200)
                    else:
                        geo_scoring[geo_scoring['Class']==grid_select].boundary.explore(m=m,
                                                                               color = 'red'
                                                                               )
                        st_data = st_folium(m, 
                                width = 1800,
                                height =600
                                )
                        dataframe1 = dataframe[dataframe['Class']==grid_select]
                        col11, col12, col13, col14 = st.columns([0.5,0.5,0.5,0.5])
                        with col11:
                            st.markdown(f"<p class='medium1-font'>Avg. Population</p>", unsafe_allow_html=True)
                            st.markdown(f"<p class='small-font'>{round(dataframe1['population_index'].mean())}</p>", unsafe_allow_html=True)
                        with col12:
                            st.markdown(f"<p class='medium1-font'>Avg. Density</p>", unsafe_allow_html=True)
                            st.markdown(f"<p class='small-font'>{round(dataframe1['density_index'].mean())}</p>", unsafe_allow_html=True)
                        with col13:
                            st.markdown(f"<p class='medium1-font'>Avg. Building</p>", unsafe_allow_html=True)
                            st.markdown(f"<p class='small-font'>{round(dataframe1['total building'].mean())}</p>", unsafe_allow_html=True)
                        with col14:
                            st.markdown(f"<p class='medium1-font'>Avg. POI</p>", unsafe_allow_html=True)
                            st.markdown(f"<p class='small-font'>{round(dataframe1['total poi'].mean())}</p>", unsafe_allow_html=True)
                        st.dataframe(dataframe[dataframe['Class']==grid_select], width=1500, height=200)


                    




        
        