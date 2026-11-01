"""Uber Trips - Predictive Analytics Dashboard.
Run the notebook first (it creates data/ and models/), then:  streamlit run app.py
"""
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Uber Trips Analytics", page_icon="🚕", layout="wide")


@st.cache_data
def load_data():
    d = pd.read_csv("data/uber_clean.csv", parse_dates=["pickup_time"])
    d["date"] = d["pickup_time"].dt.date
    return d


@st.cache_data
def load_series():
    hourly = pd.read_csv("data/hourly_series.csv", index_col=0, parse_dates=True)
    fc = pd.read_csv("data/forecast.csv", index_col=0, parse_dates=True)
    return hourly, fc


@st.cache_resource
def load_models():
    return (joblib.load("models/fare_model.joblib"),
            joblib.load("models/demand_model.joblib"),
            joblib.load("models/status_model.joblib"))


df = load_data()
hourly, forecast = load_series()
fare_model, demand_model, status_model = load_models()

# ---------------- Sidebar filters ----------------
st.sidebar.header("Filters")
cities = st.sidebar.multiselect("City", sorted(df.city.unique()), default=sorted(df.city.unique()))
statuses = st.sidebar.multiselect("Status", sorted(df.status.unique()), default=sorted(df.status.unique()))
payments = st.sidebar.multiselect("Payment method", sorted(df.payment_method.unique()),
                                  default=sorted(df.payment_method.unique()))
dmin, dmax = df.date.min(), df.date.max()
date_range = st.sidebar.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
if len(date_range) != 2:
    st.info("Select both a start and an end date.")
    st.stop()

f = df[df.city.isin(cities) & df.status.isin(statuses) & df.payment_method.isin(payments)
       & (df.date >= date_range[0]) & (df.date <= date_range[1])]

st.title("🚕 Uber Trips: Predictive Analytics Dashboard")
if f.empty:
    st.warning("No trips match the current filters.")
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Correlations", "Fare predictor", "Demand & cancellation", "Forecast"])

# ---------------- Tab 1: Overview ----------------
with tab1:
    completed = f[f.status == "Completed"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Trips", f"{len(f):,}")
    c2.metric("Revenue (completed)", f"${completed.fare_amount.sum():,.0f}")
    c3.metric("Avg fare", f"${f.fare_amount.mean():.2f}")
    c4.metric("Avg distance", f"{f.distance_km.mean():.2f} km")
    c5.metric("Completion rate", f"{f.is_completed.mean():.1%}")

    a, b = st.columns(2)
    by_hour = f.groupby("hour").size().reset_index(name="trips")
    a.plotly_chart(px.line(by_hour, x="hour", y="trips", markers=True, title="Trips by hour"),
                   width="stretch")
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    by_day = f.groupby("day_name").size().reindex(order).reset_index(name="trips")
    b.plotly_chart(px.bar(by_day, x="day_name", y="trips", title="Trips by weekday"),
                   width="stretch")

    a, b = st.columns(2)
    city_tbl = f.groupby("city").agg(trips=("trip_id", "count"), avg_fare=("fare_amount", "mean"),
                                     completion=("is_completed", "mean")).reset_index()
    a.plotly_chart(px.bar(city_tbl, x="city", y="trips", color="avg_fare", title="Trips and avg fare by city"),
                   width="stretch")
    b.plotly_chart(px.pie(f, names="status", title="Trip status"), width="stretch")

    st.subheader("Pickup locations (sample)")
    st.map(f.sample(min(3000, len(f)), random_state=1).rename(columns={"pickup_lat": "lat", "pickup_lng": "lon"})[["lat", "lon"]])
    st.caption("Note: in this dataset the coordinates do not match the city names (they look randomly generated).")

# ---------------- Tab 2: Correlations ----------------
with tab2:
    cols = ["distance_km", "fare_amount", "duration_min", "fare_per_km", "speed_kmph",
            "hour", "day_of_week", "is_weekend", "is_completed"]
    corr = f[cols].corr()
    st.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                              zmin=-1, zmax=1, title="Correlation heatmap"), width="stretch")
    st.plotly_chart(px.scatter(f.sample(min(4000, len(f)), random_state=1), x="distance_km", y="fare_amount",
                               color="city", opacity=0.5, trendline="ols", trendline_scope="overall",
                               title="Fare vs distance"), width="stretch")

# ---------------- Tab 3: Fare predictor (regression) ----------------
with tab3:
    st.subheader("Predict the fare for a trip")
    c1, c2, c3 = st.columns(3)
    dist = c1.slider("Distance (km)", 0.5, 20.0, 7.0, 0.5)
    hr = c2.slider("Hour of day", 0, 23, 12)
    dow = c3.selectbox("Day of week", range(7), format_func=lambda i: order[i])
    c1, c2 = st.columns(2)
    city = c1.selectbox("City", sorted(df.city.unique()), key="fare_city")
    pay = c2.selectbox("Payment method", sorted(df.payment_method.unique()))
    X = pd.DataFrame([{"distance_km": dist, "hour": hr, "day_of_week": dow,
                       "is_weekend": int(dow >= 5), "city": city, "payment_method": pay}])
    st.metric("Predicted fare", f"${fare_model.predict(X)[0]:.2f}")
    st.caption("Model: best of Linear / Random Forest / Gradient Boosting from the notebook (R² ≈ 0.76).")

# ---------------- Tab 4: Classification ----------------
with tab4:
    st.subheader("Demand level")
    c1, c2, c3 = st.columns(3)
    d_city = c1.selectbox("City", sorted(df.city.unique()), key="d_city")
    d_dow = c2.selectbox("Day of week", range(7), format_func=lambda i: order[i], key="d_dow")
    d_hr = c3.slider("Hour", 0, 23, 18, key="d_hr")
    Xd = pd.DataFrame([{"city": d_city, "day_of_week": d_dow, "hour": d_hr}])
    proba = pd.Series(demand_model.predict_proba(Xd)[0], index=demand_model.classes_)
    st.metric("Predicted demand", demand_model.predict(Xd)[0])
    st.plotly_chart(px.bar(proba, labels={"value": "probability", "index": ""}, title="Class probabilities"),
                    width="stretch")

    st.subheader("Demand heatmap (actual data)")
    score = {"Low Demand": 0, "Medium Demand": 1, "High Demand": 2}
    heat = (f.assign(s=f.demand_level.map(score)).groupby(["day_of_week", "hour"]).s.mean().unstack())
    heat.index = [order[i] for i in heat.index]
    st.plotly_chart(px.imshow(heat, aspect="auto", color_continuous_scale="YlOrRd",
                              labels={"color": "avg demand (0-2)"}), width="stretch")

    st.subheader("Completion risk")
    st.caption("Honest note: cancellations are not predictable from the available columns, "
               "so treat this as a baseline-level demonstration.")
    c1, c2, c3, c4 = st.columns(4)
    s_city = c1.selectbox("City", sorted(df.city.unique()), key="s_city")
    s_pay = c2.selectbox("Payment", sorted(df.payment_method.unique()), key="s_pay")
    s_dist = c3.slider("Distance (km)", 0.5, 20.0, 7.0, 0.5, key="s_dist")
    s_hr = c4.slider("Hour", 0, 23, 12, key="s_hr")
    s_dow = 2
    Xs = pd.DataFrame([{"city": s_city, "payment_method": s_pay, "distance_km": s_dist,
                        "fare_amount": 2.3 * s_dist, "hour": s_hr, "day_of_week": s_dow,
                        "is_weekend": int(s_dow >= 5)}])
    p = pd.Series(status_model.predict_proba(Xs)[0], index=status_model.classes_)
    st.write(p.map("{:.1%}".format).to_frame("probability"))

# ---------------- Tab 5: Forecast ----------------
with tab5:
    st.subheader("Next 48 hours: forecast revenue (Holt-Winters)")
    hist = hourly["revenue"].iloc[-96:].rename("actual")
    plot_df = pd.concat([hist, forecast["forecast_revenue"].rename("forecast")], axis=1).reset_index()
    plot_df = plot_df.rename(columns={plot_df.columns[0]: "time"})
    st.plotly_chart(px.line(plot_df, x="time", y=["actual", "forecast"], title="Hourly revenue: last 4 days + forecast"),
                    width="stretch")
    st.caption("Hourly volume in this dataset is nearly constant, so forecasts are close to a flat line.")
    st.dataframe(forecast.round(2))
