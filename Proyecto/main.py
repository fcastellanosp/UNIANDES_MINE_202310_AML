#!/usr/bin/env python3

import Definitions

import altair as alt
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
# import plotly.graph_objects as go
import seaborn as sns

st.set_page_config(page_title=Definitions.PAGE_TITLE, page_icon="🌡️")

df = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

st.map(df)