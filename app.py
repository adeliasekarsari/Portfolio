from hydralit import HydraApp
import hydralit_components as hc
import apps
import streamlit as st

#Only need to set these here as we are add controls outside of Hydralit, to customise a run Hydralit!
st.set_page_config(page_title="Adelia's Portfolio",page_icon="👋🏻",layout='wide',initial_sidebar_state='auto',)

if __name__ == '__main__':

    over_theme = {'txc_inactive': '#FFFFFF'}
    #this is the host application, we add children to it and that's it!
    app = HydraApp(
        title="Adelia's Portfolio",
        favicon="👋🏻",
        banner_spacing=[5,30,60,30,5],
        navbar_theme=over_theme
    )

    #Home button will be in the middle of the nav list now
    app.add_app("Home",  app=apps.HomeApp(title='Home'),is_home=True)

    #add all your application classes here
    app.add_app("Exploratory Data Analytics",  app=apps.eda_automation.main.EdaApp(title="Exploratory Data Analytics"))
    app.add_app("Business Expansion Analytics", app=apps.business_expansion.main.BEApp(title="Business Expansion Analytics"))
    app.add_app("Sentiment Analysis", app=apps.sentiment_analysis.main.SentimentApp(title="Sentiment Analysis"))
    app.add_app("Flood Analysis",  app=apps.flood_analysis.main.FloodApp(title="Flood Analysis"))

    app.add_app("About Me",  app=apps.AboutMe(title="About Me"))
    
 
    #---------------------------------------------------------------------

    complex_nav = {
            'Home': ['Home'],
            'Study Cases': ['Exploratory Data Analytics','Business Expansion Analytics',"Sentiment Analysis","Flood Analysis"],
            'About Me': ['About Me']
        }

    

  
    #and finally just the entire app and all the children.
    app.run(complex_nav)
