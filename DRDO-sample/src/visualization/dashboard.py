"""
Visualization Dashboard using Streamlit.
"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def main():
    st.title("Spatio-Temporal Anomaly Detection Dashboard")

    st.sidebar.header("Settings")
    # Placeholder for inputs

    st.header("Anomaly Heatmap")
    # Placeholder heatmap
    fig, ax = plt.subplots()
    ax.imshow(np.random.rand(100, 100), cmap='hot')
    st.pyplot(fig)

    st.header("Object-Level Anomalies")
    # Placeholder table
    st.table({"Object": ["Vehicle 1", "Ship 2"], "Anomaly Score": [0.8, 0.2]})

if __name__ == "__main__":
    main()