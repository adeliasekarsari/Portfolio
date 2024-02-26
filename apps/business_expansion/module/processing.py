import pandas as pd
import geopandas as gpd

def get_data(data_type):
    grid = gpd.read_file(r'.\apps\business_expansion\data\hex.geojson')
    admin = gpd.read_parquet(r'.\apps\business_expansion\data\admin.parquet')
    if data_type == 'Population with Hex':
        data = pd.read_parquet(r'.\apps\business_expansion\data\hex_w_pop.parquet')
    elif data_type == 'Population in Admin':
        data = gpd.read_parquet(r'.\apps\business_expansion\data\admin_w_pop.parquet') #.drop(columns = 'geometry')
    elif data_type == 'POI in Hex':
        data = pd.read_parquet(r'.\apps\business_expansion\data\hex_w_poi.parquet')
    elif data_type == 'Building in Hex':
        data = pd.read_parquet(r'.\apps\business_expansion\data\hex_w_building.parquet')
    elif data_type == 'POI by Category':
        data = gpd.read_parquet(r'.\apps\business_expansion\data\poi_all.parquet')
    
    if data_type == 'Population in Admin':
        df = data.copy()#pd.merge(admin, data, on = 'nama_kecamatan')
    elif data_type == 'POI by Category':
        df = data.copy()
    else:
        df = pd.merge(grid, data, how = 'left', on = 'index')
    return data, df

def get_map(df, data_type, classify=None):
    admin = gpd.read_parquet(r'.\apps\business_expansion\data\admin.parquet')
    if  data_type == 'Building in Hex':
        maps = df.explore(column = 'total building',
                    cmap = 'GnBu',
                    tiles = 'cartodb darkmatter')
        return maps
    
    elif data_type == 'POI in Hex' or data_type == 'Population in Admin' or data_type == 'Population with Hex':
        maps = df.explore(column = classify,
                    cmap = 'GnBu',
                    tiles = 'cartodb darkmatter')
        return maps

    elif data_type == 'POI by Category':
        df = df[df['category']==classify]
        maps = admin.explore(tiles = 'cartodb darkmatter', color = 'lightblue')
        df.explore(m = maps)
        return maps

def get_title_text(data_type):
    if data_type == 'Population with Hex':
        title = "Hexagonal Population Visualization for Bandung City (2022)"
        text = {"Description":"""
                Explore the population landscape of Bandung City 
                in 2022 through an interactive hexagonal visualization generated 
                from data provided by the Badan Pusat Statistika (BPS). This 
                visualization offers a comprehensive overview of both total population 
                counts and population density across different geographic areas within Bandung City.""",
                "Total Population":"""
                Discover the distribution of total population across hexagonal units, 
                each representing a specific area within Bandung City. The visualization 
                showcases areas with higher concentrations of population, providing 
                insights into densely populated neighborhoods, urban centers, 
                and suburban regions.
                """,
                "Population Density":"""
                Dive deeper into the population density of Bandung City with the density 
                layer of the hexagonal visualization. By overlaying population density 
                data onto the hexagonal grid, this feature enables users to identify areas 
                with the highest population densities, facilitating analyses of urban density 
                patterns and population distribution.
                """,
                "Interactive Exploration":"""
                Interact with the hexagonal visualization to zoom in on specific neighborhoods,
                toggle between total population and density views, and access detailed
                population statistics for individual hexagons within Bandung City. 
                Gain a nuanced understanding of population dynamics by exploring 
                the spatial distribution of both total population and population 
                density in Bandung City in 2022.
                """
                }
    elif data_type == 'Population in Admin':
        title = "Administrative-Level Population Visualization for Bandung City (2022)"
        text = {"Description":"""
                Explore the population landscape of Bandung City in 
                2022 through an interactive visualization based on administrative 
                boundaries provided by the Badan Informasi Geospatial (BIG). This 
                visualization offers insights into total population counts and 
                population density across different administrative levels within Bandung City.""",
                "Administrative Boundaries":"""
                Discover the administrative hierarchy of Bandung 
                City, including districts, sub-districts, and villages. Each administrative level 
                is represented on the map, allowing users to navigate through the hierarchical 
                structure and explore population data at various levels of granularity.""",
                "Total Population":"""
                Gain insights into the distribution of total population across administrative 
                boundaries. The visualization highlights areas with larger populations, 
                providing a clear understanding of population concentrations within districts, 
                sub-districts, and villages throughout Bandung City.
                """,
                "Population Density":"""
                Examine population density across administrative boundaries to identify areas 
                with high population density. By visualizing population density data, users 
                can discern densely populated regions and areas with lower population density, 
                enabling analyses of urbanization patterns and population distribution at 
                different administrative levels.
                """,
                "Interactive Exploration":"""
                Interact with the visualization to explore population data at different 
                administrative levels within Bandung City. Zoom in to view detailed 
                population statistics for specific districts, sub-districts, or villages, 
                and toggle between total population counts and population density views 
                for deeper insights.
                """
                }
    elif data_type == 'POI in Hex':
        title = "Hexagonal Point of Interest (POI) Visualization for Bandung City"
        text = {"Description":"""
                Embark on a journey to explore the diverse array 
                of Points of Interest (POIs) scattered across Bandung City through an 
                interactive hexagonal visualization powered by data sourced from OpenStreetMap. 
                This innovative visualization offers a comprehensive overview of key landmarks, 
                attractions, amenities, and other points of interest at the hexagon level within 
                Bandung City""",
                "POI Diversity":"""
                Discover the rich tapestry of POIs that characterize Bandung City's urban landscape. 
                From iconic landmarks and cultural attractions to bustling markets, restaurants, 
                parks, and recreational facilities, the hexagonal visualization provides insights 
                into the diverse range of amenities and attractions available to residents and 
                visitors alike.
                """,
                "Interactive Exploration":"""
                Interact with the hexagonal visualization to zoom in on specific areas of interest, 
                revealing detailed POI information within individual hexagons. Toggle between 
                different categories of POIs, such as restaurants, hotels, landmarks, and more, 
                to tailor your exploration based on your interests and preferences.
                """,
                "Data Sourced from OpenStreetMap":"""
                Harness the power of crowdsourced data from OpenStreetMap to access up-to-date 
                and comprehensive POI information for Bandung City. Leveraging the collaborative 
                efforts of contributors worldwide, the visualization provides a reliable and 
                dynamic platform for discovering and exploring POIs across the city.
                """,
                }
    elif data_type == 'Building in Hex':
        title = "Hexagonal Building Visualization for Bandung City"
        text = {"Description":"""
                Embark on an immersive journey to explore the urban fabric of 
                Bandung City through an interactive hexagonal visualization showcasing building
                data sourced from OpenStreetMap. This innovative visualization offers a unique
                perspective on the architectural diversity and spatial distribution of buildings 
                at the hexagon level within Bandung City.""",
                "Building Diversity":"""
                Discover the architectural richness of Bandung City's built environment as you
                explore the hexagonal visualization. From historic landmarks and modern 
                skyscrapers to residential complexes and commercial developments, the 
                visualization provides insights into the diverse range of building types
                and styles that define the city's urban landscape.
                """,
                "Interactive Exploration":"""
                Interact with the hexagonal visualization to zoom in on specific areas of 
                interest, revealing detailed building information within individual hexagons. 
                Toggle between different categories of buildings, such as residential, 
                commercial, educational, and cultural, to tailor your exploration based 
                on your interests and preferences.
                """,
                "Data Sourced from OpenStreetMap":"""
                Access up-to-date and comprehensive building data for Bandung City 
                sourced from OpenStreetMap, a collaborative mapping platform powered by 
                contributors worldwide. Leveraging this crowdsourced data, the visualization 
                provides a reliable and dynamic platform for discovering and exploring 
                buildings across the city.
                """,
                }
    elif data_type == 'POI by Category':
        title = "Exploring Points of Interest (POIs) in Bandung City by Category"
        text = {"Description":"""
                Embark on an enlightening exploration of Bandung City's 
                vibrant Points of Interest (POIs) categorized by their unique attributes 
                and characteristics. Utilizing data sourced from OpenStreetMap (OSM), this
                visualization offers an immersive experience, allowing users to discover 
                and engage with diverse POIs across various categories that define the city's 
                cultural and social landscape.""",
                "Comprehensive POI Dataset":"""
                Delve into a rich dataset of POIs meticulously curated from OpenStreetMap, 
                a collaborative mapping platform driven by contributions from a global 
                community of mappers. From cultural landmarks and recreational facilities
                to dining establishments and shopping destinations, the visualization 
                encompasses a broad spectrum of POI categories that cater to diverse 
                interests and preferences.
                """,
                "Categorized Visualization":"""
                Experience Bandung City's POIs through a categorized visualization that 
                organizes POIs into distinct categories based on their functionalities 
                and purposes. Navigate through an intuitive interface that allows users 
                to explore POIs within specific categories such as landmarks, restaurants, 
                parks, museums, shopping centers, and more
                """,
                "Interactive Discovery":"""
                Immerse yourself in an interactive journey as you navigate through Bandung 
                City's neighborhoods and districts, each brimming with unique POIs waiting 
                to be explored. Engage with the visualization to pinpoint POIs of interest, 
                access detailed information about each location, and visualize their spatial
                  distribution within the city.
                """,
                "Data Sourced from OpenStreetMap":"""
                Harness the power of crowdsourced data from OpenStreetMap to access up-to-date 
                and comprehensive POI information for Bandung City. Leveraging the collaborative 
                efforts of contributors worldwide, the visualization provides a reliable and 
                dynamic platform for discovering and exploring POIs across the city.
                """,
                }
    return title, text

