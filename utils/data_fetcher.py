import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

@st.cache_data
def fetch_x_data(hashtag, days=7):
    dates = [datetime.now() - timedelta(days=x) for x in range(days)]
    frequencies = np.random.randint(100, 1000, size=len(dates))
    data = pd.DataFrame({"date": dates, "frequency": frequencies})
    data["date"] = pd.to_datetime(data["date"])
    return data.sort_values("date")