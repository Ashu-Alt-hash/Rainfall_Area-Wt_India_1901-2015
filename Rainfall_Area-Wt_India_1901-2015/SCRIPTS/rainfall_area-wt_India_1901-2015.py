import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import pearsonr
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
from prophet.plot import plot_plotly

# Load dataset
data_path = r"C:\Users\91877\Downloads\rainfall_area-wt_India_1901-2015.csv"
rainfall_data = pd.read_csv(data_path)

# Preview data
print(rainfall_data.head())

# =========================
# Annual Rainfall Trend
# =========================
annual_rainfall = rainfall_data[['YEAR', 'ANNUAL']]

fig_annual = go.Figure()
fig_annual.add_trace(go.Scatter(
    x=annual_rainfall['YEAR'], y=annual_rainfall['ANNUAL'],
    mode='lines', name='Annual Rainfall',
    line=dict(color='blue', width=2), opacity=0.7
))
fig_annual.add_trace(go.Scatter(
    x=annual_rainfall['YEAR'], y=[annual_rainfall['ANNUAL'].mean()] * len(annual_rainfall),
    mode='lines', name='Mean Rainfall',
    line=dict(color='red', dash='dash')
))
fig_annual.update_layout(
    title='Trend in Annual Rainfall in India (1901–2015)',
    xaxis_title='Year', yaxis_title='Rainfall (mm)', template='plotly_white', height=500
)
fig_annual.show()

# =========================
# Monthly Rainfall Analysis
# =========================
monthly_columns = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
monthly_avg = rainfall_data[monthly_columns].mean()
highest_rainfall_month = monthly_avg.idxmax()
lowest_rainfall_month = monthly_avg.idxmin()

fig_monthly = px.bar(
    x=monthly_avg.index, y=monthly_avg.values,
    labels={'x': 'Month', 'y': 'Rainfall (mm)'},
    title='Average Monthly Rainfall in India (1901–2015)',
    text=monthly_avg.values
)
fig_monthly.add_hline(
    y=monthly_avg.mean(), line_dash="dash", line_color="red",
    annotation_text="Mean Rainfall", annotation_position="top right"
)
fig_monthly.update_traces(marker_color='skyblue', marker_line_color='black', marker_line_width=1)
fig_monthly.update_layout(template='plotly_white', height=500)
fig_monthly.show()

# =========================
# Seasonal Rainfall Distribution
# =========================
seasonal_columns = ['Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec']
seasonal_avg = rainfall_data[seasonal_columns].mean()

fig_seasonal = px.bar(
    x=seasonal_avg.index, y=seasonal_avg.values,
    labels={'x': 'Season', 'y': 'Rainfall (mm)'},
    title='Seasonal Rainfall Distribution in India (1901–2015)',
    text=seasonal_avg.values,
    color=seasonal_avg.values, color_continuous_scale=['gold', 'skyblue', 'green', 'orange']
)
fig_seasonal.update_traces(marker_line_color='black', marker_line_width=1)
fig_seasonal.update_layout(template='plotly_white', height=500)
fig_seasonal.show()

# =========================
# Rolling Average Analysis
# =========================
rainfall_data['10-Year Rolling Avg'] = rainfall_data['ANNUAL'].rolling(window=10).mean()

fig_climate_change = go.Figure()
fig_climate_change.add_trace(go.Scatter(
    x=rainfall_data['YEAR'], y=rainfall_data['ANNUAL'],
    mode='lines', name='Annual Rainfall', line=dict(color='blue'), opacity=0.6
))
fig_climate_change.add_trace(go.Scatter(
    x=rainfall_data['YEAR'], y=rainfall_data['10-Year Rolling Avg'],
    mode='lines', name='10-Year Rolling Avg', line=dict(color='red', width=3)
))
fig_climate_change.update_layout(
    title='Impact of Climate Change on Rainfall Patterns',
    xaxis_title='Year', yaxis_title='Rainfall (mm)',
    template='plotly_white', height=500
)
fig_climate_change.show()

# =========================
# Drought and Extreme Rainfall Years
# =========================
mean_rainfall = rainfall_data['ANNUAL'].mean()
std_dev_rainfall = rainfall_data['ANNUAL'].std()

drought_years = rainfall_data[rainfall_data['ANNUAL'] < (mean_rainfall - 1.5 * std_dev_rainfall)]
extreme_years = rainfall_data[rainfall_data['ANNUAL'] > (mean_rainfall + 1.5 * std_dev_rainfall)]

# Correlation of seasonal rainfall with annual rainfall
seasonal_correlations = {
    season: pearsonr(rainfall_data[season], rainfall_data['ANNUAL'])[0] for season in seasonal_columns
}
seasonal_corr_df = pd.DataFrame.from_dict(seasonal_correlations, orient='index', columns=['Correlation'])

print("Drought Years:")
print(drought_years[['YEAR', 'ANNUAL']])
print("\nExtreme Rainfall Years:")
print(extreme_years[['YEAR', 'ANNUAL']])
print("\nSeasonal Correlations:")
print(seasonal_corr_df)

# =========================
# Anomaly Detection (Isolation Forest)
# =========================
model = IsolationForest(contamination=0.05, random_state=42)
rainfall_data['Annual_Anomaly'] = model.fit_predict(rainfall_data[['ANNUAL']])

monthly_data = rainfall_data[monthly_columns]
rainfall_data['Monthly_Anomaly'] = model.fit_predict(monthly_data)

annual_anomalies = rainfall_data[rainfall_data['Annual_Anomaly'] == -1]
monthly_anomalies_df = rainfall_data[rainfall_data['Monthly_Anomaly'] == -1][['YEAR'] + monthly_columns]

# Plot annual anomalies
fig_annual_anomalies = go.Figure()
fig_annual_anomalies.add_trace(go.Scatter(
    x=rainfall_data['YEAR'], y=rainfall_data['ANNUAL'],
    mode='lines', name='Annual Rainfall', line=dict(color='blue')
))
fig_annual_anomalies.add_trace(go.Scatter(
    x=annual_anomalies['YEAR'], y=annual_anomalies['ANNUAL'],
    mode='markers', name='Anomalies', marker=dict(color='red', size=8)
))
fig_annual_anomalies.add_hline(y=mean_rainfall, line_dash='dash', line_color='green', annotation_text='Mean Rainfall')
fig_annual_anomalies.update_layout(
    title='Annual Rainfall Anomalies in India (1901–2015)', template='plotly_white', height=500
)
fig_annual_anomalies.show()

# Monthly anomaly points
monthly_anomalies = []
for col in monthly_columns:
    for _, row in monthly_anomalies_df.iterrows():
        monthly_anomalies.append({'Year': row['YEAR'], 'Month': col, 'Rainfall': row[col]})
monthly_anomalies_long = pd.DataFrame(monthly_anomalies)

# Monthly trends and anomalies
fig_monthly_anomalies = px.line(
    rainfall_data, x='YEAR', y=monthly_columns,
    labels={'value': 'Rainfall (mm)', 'variable': 'Month'}, title='Monthly Rainfall Anomalies',
    color_discrete_sequence=px.colors.qualitative.Set3
)
fig_monthly_anomalies.add_trace(go.Scatter(
    x=monthly_anomalies_long['Year'], y=monthly_anomalies_long['Rainfall'],
    mode='markers', name='Anomalous Months', marker=dict(color='red', size=5)
))
fig_monthly_anomalies.update_layout(template='plotly_white', height=500)
fig_monthly_anomalies.show()

# =========================
# Monsoon Correlation with Other Seasons
# =========================
monsoon_col = 'Jun-Sep'
monsoon_relationships = {
    season: pearsonr(rainfall_data[monsoon_col], rainfall_data[season])[0]
    for season in seasonal_columns if season != monsoon_col
}
corr_df = pd.DataFrame({'Season': monsoon_relationships.keys(), 'Correlation Coefficient': monsoon_relationships.values()})

fig_corr = px.bar(
    corr_df, x='Season', y='Correlation Coefficient',
    title='Correlation Between Monsoon and Other Seasons',
    color='Correlation Coefficient', color_continuous_scale='Blues',
    text='Correlation Coefficient'
)
fig_corr.add_hline(y=0, line_dash='dash', line_color='red', annotation_text='No Correlation')
fig_corr.update_traces(marker_line_color='black', marker_line_width=1, texttemplate='%{text:.2f}')
fig_corr.update_layout(template='plotly_white', height=500)
fig_corr.show()

# =========================
# K-Means Clustering
# =========================
features = rainfall_data[seasonal_columns + ['ANNUAL']]
scaled = StandardScaler().fit_transform(features)

kmeans = KMeans(n_clusters=3, random_state=42)
rainfall_data['Cluster'] = kmeans.fit_predict(scaled)
cluster_map = {0: 'Dry', 1: 'Normal', 2: 'Wet'}
rainfall_data['Category'] = rainfall_data['Cluster'].map(cluster_map)

fig_cluster = px.scatter(
    rainfall_data, x='YEAR', y='ANNUAL', color='Category',
    title='Rainfall Clustering by K-Means',
    color_discrete_sequence=px.colors.qualitative.Set2
)
fig_cluster.update_layout(template='plotly_white', height=500)
fig_cluster.show()

# =========================
# Time-Series Forecast (Prophet)
# =========================
rainfall_data['DATE'] = pd.to_datetime(rainfall_data['YEAR'], format='%Y')
prophet_data = rainfall_data[['DATE', 'ANNUAL']].rename(columns={'DATE': 'ds', 'ANNUAL': 'y'})

model = Prophet()
model.fit(prophet_data)

future = model.make_future_dataframe(periods=20, freq='Y')
forecast = model.predict(future)

fig_forecast = plot_plotly(model, forecast)
fig_forecast.update_layout(
    title='Forecast of Annual Rainfall (Next 20 Years)', xaxis_title='Year',
    yaxis_title='Rainfall (mm)', template='plotly_white', height=500
)
fig_forecast.show()

# =========================
# Conclusion
# =========================
print("""
Conclusion:
- The monsoon season (June–September) dominates India’s rainfall.
- Rolling averages and anomalies indicate increasing variability over time.
- Clustering analysis suggests a shift toward drier years in recent decades.
- Forecasting suggests potential decline in average rainfall—critical for future planning.
""")
