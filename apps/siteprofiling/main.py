import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from  PIL import Image
import numpy as np
import pandas as pd
from st_aggrid import AgGrid
import plotly.express as px
import folium
from streamlit_folium import st_folium
import geopandas as gpd
from folium import plugins
from folium.plugins import HeatMap
import plotly.graph_objects as go
import apps.siteprofiling.module as viz
from hydralit import HydraHeadApp


class LippoApp(HydraHeadApp):

    def __init__(self, title = '', **kwargs):
        self.__dict__.update(kwargs)
        self.title = title
        
    def run(self):
        if st.button("Back to Projects"):
                # Set session state to navigate back to the project app
                st.session_state.selected_project = None
        loc_url = 'apps/siteprofiling/data/data_tematic'
        loc_url2 = 'apps/siteprofiling/data/data_mw'

        name = 'Cibubur Junction'
        gdf_all = gpd.read_file("apps/siteprofiling/data/lippo_building.geojson")
        data_ages = pd.read_parquet(f'{loc_url}/demog_ages_{name}.parquet')
        data_ses = pd.read_parquet(f'{loc_url}/ses_{name}.parquet')
        group_poi = pd.read_parquet(f'{loc_url}/poi_{name}.parquet')
        catch_overlay = gpd.read_file(f'{loc_url}/catch_overlay_{name}.geojson')
        dt = gpd.read_file(f'{loc_url}/dt_{name}.geojson')
        gdf_admin = gpd.read_file(f'{loc_url2}/gdf_admin_{name}.geojson')
        distance = gpd.read_file(f'{loc_url2}/distance_{name}.geojson')
        mw_in = pd.read_parquet(f'{loc_url2}/mw_in_{name}.parquet')
        segmen_sum = pd.read_parquet(f'{loc_url2}/segment_{name}.parquet')

        ages_list = ['age_1824','age_2534','age_3544','age_4554','age_55']
        income_list = ['income_average','income_high','income_low']
        gender_list = ['mi_female','mi_male']

        gdf = gdf_all[gdf_all['name']==name]
        gdf['lat'] = gdf.centroid.geometry.y
        gdf['lon'] = gdf.centroid.geometry.x

        with st.sidebar:
            choose = option_menu("Site Profiling", ["Site Profiling", "Mobility Analysis"],
                                icons=['book','pin-map-fill','person lines fill'],
                                menu_icon="app-indicator", default_index=0,
                                styles={
                "container": {"padding": "5!important", "background-color": "black"},
                "icon": {"color": "White", "font-size": "25px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "lightblue"},
            }
            )

        if choose == "Site Profiling":
            st.markdown("""
            <style>
            .big-font {
                font-size:54px !important;
            }
            </style>
            """, unsafe_allow_html=True)

            st.markdown('<p class="big-font">Site Profiling</p>', unsafe_allow_html=True)
            
            #get data
            col1, col2 = st.columns( [0.35, 0.65])
            with col1:
                #plot demog by ages
                color1 = '#0D2C5B'
                color2 = '#FDD212'
                color = viz.get_color_gradient(color1, color2, data_ages.Ages.nunique())
                fig = px.bar(data_ages, 
                            x='Ages', 
                            y='population',
                            hover_data=['population','Ages'], 
                            color = 'Ages', 
                            color_discrete_sequence=color, 
                            height=450,
                            animation_frame="dt", 
                            )

                for step in fig.layout.sliders[0].steps:
                    step["args"][1]["frame"]["redraw"] = True
                fig.update_xaxes(type='category')
                fig.update_layout(yaxis_range=[0,round(data_ages['population'].max()/1000)*1000],
                                title = 'Total Population category by Ages'
                                )
                st.plotly_chart(fig, use_container_width=True)

                #plot ses
                fig = px.bar(data_ses, 
                            x='nilai', 
                            y='ses',
                            hover_data=['nilai'], 
                            color = 'nilai', 
                            color_discrete_map={
                                                'high':'#0D2C5B',
                                                'medium-high':'#206DAB',
                                                'medium-low':'#FDD212',
                                                'low':'#FDF8B3'}, 
                            labels={
                                    "ses": "Percentage (%)",
                                    "nilai": "SocioEconomic Status",},
                            height=350,
                            animation_frame="dt", 
                            category_orders={'nilai':['high','medium-high','medium-low','low']}
                            )
                fig.update_xaxes(categoryorder='array', 
                                    categoryarray= ['high','medium-high','medium-low','low'])
                fig.update_layout(yaxis_range=[0,100],title = 'Sosio Economic Percentage (%)')
                for step in fig.layout.sliders[0].steps:
                    step["args"][1]["frame"]["redraw"] = True
                fig.update_xaxes(type='category')
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                #basemap
                geom = gpd.GeoSeries(catch_overlay.set_index('distance')['geometry']).to_json()
                distance_reverse = [5000, 3000, 2000, 1000, 500]
                base_map = folium.Map(
                    location = (dt.dissolve().centroid.y[0],dt.dissolve().centroid.x[0]),
                    control_scale = True,
                    zoom_start = 13.5,
                    tiles = None
                )
                tile_layer = folium.TileLayer(
                    tiles="https://{s}.basemaps.cartocdn.com/rastertiles/dark_all/{z}/{x}/{y}.png",
                    attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
                    max_zoom=19,
                    name='darkmatter',
                    control=False,
                    opacity=0.7
                )

                tile_layer.add_to(base_map)

                #cathment visualization
                color1 = '#0D2C5B'
                color2 = '#FDD212'
                color = viz.get_color_gradient(color1, 
                                        color2, 
                                        7)
                #icon plot
                points1 = list(map(tuple, zip([gdf['lat'].mean()], [gdf['lon'].mean()])))
                train_group = folium.FeatureGroup(name="Lokasi {}".format(gdf.name.unique()[0])).add_to(base_map)
                for tuple_ in points1:
                        icon=folium.Icon(color='darkblue', icon='fa-shopping-cart ', icon_color="white", prefix='fa')
                        train_group.add_child(folium.Marker(tuple_, icon=icon))
                folium.Choropleth(
                    geo_data = geom,
                    name = 'Choropleth',
                    data = catch_overlay,
                    columns = ['distance','dt'],
                    key_on = 'feature.id',
                    fill_color = 'Blues',
                    fill_opacity = 0.5,
                    line_opacity = 0,
                    legend_name = 'Unemployment (%)',
                    smooth_factor=  0
                ).add_to(base_map)
                index = 0
                #color = ['darkblue','darkred','darkyellow','blue','red','yellow','white']
                for i in distance_reverse:
                    dt_catch = dt[dt['dt']==i].copy()
                    style = {
                        'fillColor': 'white',
                        'color': color1,
                        'weight': 2,
                        }
                    range_group = folium.FeatureGroup(name="Range Area {}".format(i)).add_to(base_map)
                    range_group.add_child(folium.GeoJson(
                        catch_overlay[catch_overlay['range_area']==i].geometry,style_function=lambda feature: style))
                folium.LayerControl().add_to(base_map)
                st_data = st_folium(base_map, 
                                    width = 1100, 
                                    height = 800)
            color = viz.get_color_gradient(color1, 
                                color2, 
                                group_poi.groupby('range_area')['nama_kategori'].nunique().max())

            fig = px.bar(group_poi.sort_values(['range_area','id_merchant']), 
                        y='id_merchant', 
                        x='nama_kategori',
                        hover_data=['id_merchant'], 
                        color = 'nama_kategori', 
                        color_discrete_sequence=color, 
                        labels={
                                "id_merchant": "Total POI",
                                "nama_kategori": "Category Names",},
                        height=500,
                        width = 900,
                        animation_frame="range_area", 
                        )
            fig.update_xaxes(type='category')
            fig.update_layout(yaxis_range=[0,round(group_poi['id_merchant'].max())], xaxis = None,
                            title = 'Total POI based on Category')
            for step in fig.layout.sliders[0].steps:
                step["args"][1]["frame"]["redraw"] = True
            fig.update_xaxes(type='category')
            st.plotly_chart(fig, use_container_width=True)

        elif choose == 'Mobility Analysis':
            name = 'Cibubur Junction'
            color1 = '#0D2C5B'
            color2 = '#FDD212'
            color = viz.get_color_gradient(color1, 
                                        color2, 
                                        7)
            tab1, tab2 = st.tabs(["Exploratory Data Analysis", "Home Location"])
            with tab1:
                st.header("EDA Telco Data")
                col1, col2, col3 = st.columns( [0.4, 0.05, 0.55])
                with col1:
                    fig1 = viz.mw_hour(mw_in)
                    st.plotly_chart(fig1, use_container_width=True)
                    fig2 = viz.mw_chart_weekdays(mw_in)
                    st.plotly_chart(fig2, use_container_width=True)
                with col2:
                    st.write("")
                with col3:
                    base_map = folium.Map(
                        location = ([mw_in['lat'].mean(), mw_in['lon'].mean()]),
                        control_scale = True,
                        zoom_start = 16,
                        tiles = None
                    )
                    tile_layer = folium.TileLayer(
                        tiles="https://{s}.basemaps.cartocdn.com/rastertiles/dark_all/{z}/{x}/{y}.png",
                        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
                        max_zoom=25,
                        name='darkmatter',
                        control=False,
                        opacity=0.6
                    )
                    tile_layer.add_to(base_map)
                    #icon plot
                    points1 = list(map(tuple, zip([mw_in['lat'].mean()], [mw_in['lon'].mean()])))
                    train_group = folium.FeatureGroup(name="Lokasi {}".format(gdf.name.unique()[0])).add_to(base_map)
                    for tuple_ in points1:
                            icon=folium.Icon(color='darkblue', icon='fa-shopping-cart ', icon_color="white", prefix='fa')
                            train_group.add_child(folium.Marker(tuple_, icon=icon))
                    # plot heatmap
                    lat_long = list(map(list, zip(mw_in.lat, mw_in.lon)))
                    HeatMap(lat_long).add_to(base_map)
                    st_data = st_folium(base_map, 
                                    width = 800, 
                                    height = 700)
                fig4 = viz.mw_chart_heatmap(mw_in)
                st.plotly_chart(fig4, use_container_width=True)

            with tab2:
                st.header("Home Location Analysis")
                col1, col2 = st.columns( [0.5, 0.55])
                with col1:
                    base_map = folium.Map(
                        location = ([mw_in['lat'].mean(), mw_in['lon'].mean()]),
                        control_scale = True,
                        zoom_start = 10,
                        tiles = None
                    )
                    tile_layer = folium.TileLayer(
                        tiles="https://{s}.basemaps.cartocdn.com/rastertiles/dark_all/{z}/{x}/{y}.png",
                        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
                        max_zoom=19,
                        name='darkmatter',
                        control=False,
                        opacity=0.6
                    )
                    tile_layer.add_to(base_map)
                    #icon plot
                    points1 = list(map(tuple, zip([mw_in['lat'].mean()], [mw_in['lon'].mean()])))
                    train_group = folium.FeatureGroup(name="Lokasi {}".format(gdf.name.unique()[0])).add_to(base_map)
                    for tuple_ in points1:
                            icon=folium.Icon(color='darkblue', icon='fa-shopping-cart ', icon_color="white", prefix='fa')
                            train_group.add_child(folium.Marker(tuple_, icon=icon))

                    #catchment 1 km
                    catch1000 = dt[dt['dt']==1000].copy()
                    style = {
                        'fillColor': 'white',
                        'color': color1,
                        'weight': 3,
                        }
                    range_group = folium.FeatureGroup(name="Catchment 1 KM").add_to(base_map)
                    range_group.add_child(folium.GeoJson(
                        catch1000.geometry,style_function=lambda feature: style))


                            
                    #choropleth desa
                    gdf_admin['kode_kecamatan'] = gdf_admin['kode_kecamatan'].astype(str)
                    geom = gpd.GeoSeries(gdf_admin.set_index('kode_kecamatan')['geometry']).to_json()
                    folium.Choropleth(
                        geo_data = geom,
                        name = 'Home Location',
                        data = gdf_admin,
                        columns = ['kode_kecamatan','total_device'],
                        key_on = 'feature.id',
                        fill_color = 'Blues',
                        fill_opacity = 0.5,
                        line_opacity = 0.5,
                        legend_name = 'Total Device',
                        smooth_factor=  0
                    ).add_to(base_map)

                    folium.LayerControl().add_to(base_map)
                    st_data = st_folium(base_map, 
                                        width = 800, 
                                        height = 700)

                with col2:
                    #plot chart
                    fig = go.Figure()
                    trace1 = gdf_admin.groupby('nama_kecamatan')['total_device'].sum().reset_index().sort_values('total_device', ascending = False)
                    
                    fig = px.bar(trace1[:10], 
                        y='total_device', 
                        x='nama_kecamatan',
                        orientation = 'v',
                        hover_data=['total_device'], 
                        height=400, 
                        labels={
                                "nama_kecamatan": "District",
                                "total_device": "Total Devices",},
                        )

                    fig.update_xaxes(type='category')
                    fig.update_traces(marker_color='lightblue')
                    fig.update_layout(font = dict(
                        size=21),
                        title = 'Top 10 District with based on total device')
                    st.plotly_chart(fig, use_container_width=True)


                    
                    fig = go.Figure()
                    trace2 = gdf_admin.groupby('nama_kota')['total_device'].sum().reset_index().sort_values('total_device', ascending = False)
                    
                    fig = px.bar(trace2[:10], 
                        y='total_device', 
                        x='nama_kota',
                        hover_data=['total_device'], 
                        height=400, 
                        labels={
                                "nama_kota": "City",
                                "total_device": "Total Devices",},
                        )

                    fig.update_xaxes(type='category')
                    fig.update_traces(marker_color='lightblue')
                    fig.update_layout(font = dict(
                        size=21),
                        title = 'Top 10 City with based on total device')
                    st.plotly_chart(fig, use_container_width=True)
                col1, col2, col3 = st.columns([0.5,0.25,0.25])
                color1 = '#0D2C5B'
                color2 = '#ADD8E6'
                with col1:
                    ages = segmen_sum[segmen_sum['description'].isin(ages_list)]
                    color = viz.get_color_gradient(color1, 
                                            color2, 
                                            5)
                    fig = px.bar(ages, 
                                y='description', 
                                x='total_pop',
                                hover_data=['total_pop'], 
                                color = 'description', 
                                color_discrete_sequence=color, 
                                labels={
                                        "description": "Descriptions",
                                        "total_pop": "Total Populations",},
                                height=500,
                                width = 700,
                                title = "Total Population Ages segmentation based on Telco Data"
                                )
                    fig.update_yaxes(type='category')
                    st.plotly_chart(fig, use_container_width=True)
                with col3:
                    color = viz.get_color_gradient(color1, 
                                            color2, 
                                            3)
                    income = segmen_sum[segmen_sum['description'].isin(income_list)]
                    fig = px.pie(income, 
                                values='total_pop', 
                                names='description', 
                                title='Total Population by Income',
                                color_discrete_sequence=color, )
                    fig.update_yaxes(type='category')
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    color2 = '#941266'
                    color1 = '#ADD8E6'
                    color = viz.get_color_gradient(color1, 
                                color2, 
                                2)
                    gender = segmen_sum[segmen_sum['description'].isin(gender_list)]
                    fig = px.pie(gender, 
                                values='total_pop', 
                                names='description', 
                                title='Total Population by Gender',
                                color_discrete_sequence=color, )
                    fig.update_yaxes(type='category')
                    st.plotly_chart(fig, use_container_width=True)