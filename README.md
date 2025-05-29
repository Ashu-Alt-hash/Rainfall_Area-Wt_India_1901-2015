# 🌧️ Rainfall Analysis of India (1901–2015) With Python

> A data science project analyzing over a century of rainfall data in India to uncover patterns, anomalies, and climate change indicators using visualization, statistics, and forecasting models.

>  **Note:** You can find or download the dataset from [data.gov.in](https://data.gov.in/) or https://statso.io/rainfall-trends-in-india-case-study/ or other open government data sources.

> I would like to express my sincere gratitude to Aman Kharwal Sir for his invaluable guidance, continuous support, and encouragement throughout the course of this project. His expert advice, insightful suggestions, and constructive feedback were essential to the successful completion of this work.


## 📌 Overview

This project involves the exploration and forecasting of India's rainfall trends using historical data from 1901 to 2015. It includes:

- Trend analysis of annual, monthly, and seasonal rainfall
- Rolling average insights to assess long-term climate variability
- Detection of anomalies and extreme events (drought/flood years)
- Correlation study across seasonal patterns
- K-Means clustering to classify rainfall years as Dry, Normal, or Wet
- Future rainfall prediction using Facebook Prophet

## 🧪 Technologies Used

- Python 🐍
- [Pandas] for data manipulation
- [Plotly] for interactive visualizations
- [Scikit-learn] for machine learning (anomaly detection, clustering)
- [Prophet] for time series forecasting
- [SciPy] for statistical analysis

## 📊 Visualizations & Analysis Performed

- 📈 Annual Rainfall Trend (with mean)
- 📅 Monthly Rainfall Pattern
- 🌦️ Seasonal Rainfall Distribution
- 🔍 Rolling Average (10-Year) Trend
- 🚨 Anomaly Detection using Isolation Forest
- 🔗 Correlation Between Seasons
- 🎯 Clustering (Dry, Normal, Wet Years)
- 🔮 Forecasting with Prophet

## 📁 Dataset

The dataset used is:
- 📄 **rainfall_area-wt_India_1901-2015.csv**
- Contains monthly, seasonal, and annual rainfall data across India (area-weighted averages)
  
📝 Results Summary

🌧️ Monsoon (June–September) is the dominant contributor to total annual rainfall.

⚠️ Detected multiple drought and extreme rainfall years.

📉 Slight declining trend in rainfall — indicating potential long-term climate change.

🧠 Clustering reveals shift toward drier years in recent decades.

🔮 Forecasting suggests the need for water resource planning and adaptation strategies.

