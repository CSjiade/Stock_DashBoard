import pandas as pd
import streamlit as st
import base64




# function to your download dataframes to a csv file
def download_link(object_to_download, download_filename, download_link_text):

    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    b64 = base64.b64encode(object_to_download.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'


def download_interface(stock_data,stock_ticker):
    tmp_download_link = download_link(stock_data, stock_ticker + '.csv', 'Click here to download the Stock data')
    st.markdown(tmp_download_link, unsafe_allow_html=True)
