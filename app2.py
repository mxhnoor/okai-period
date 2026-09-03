import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium

st.set_page_config(
    page_title="OKAI Period",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .stApp { background-color: #fff5f5; }

    .main-header {
        background: #ef4444;
        padding: 1.75rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.25rem;
    }
    .main-header h1 { margin: 0; font-size: 1.75rem; font-weight: 700; }
    .main-header p  { margin: 0.4rem 0 0; opacity: 0.85; font-size: 0.95rem; }

    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 2px 8px rgba(153,27,27,0.08);
        border-left: 4px solid #ef4444;
    }
    .metric-label {
        color: #6b7280;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .metric-value          { color: #1f2937; font-size: 1.9rem; font-weight: 700; line-height: 1.15; }
    .metric-value.danger   { color: #dc2626; }
    .metric-value.warning  { color: #d97706; }
    .metric-value.good     { color: #059669; }

    .action-card {
        background: white;
        border-radius: 10px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border-left: 3px solid #ef4444;
        height: 100%;
    }

    .legend-item { display: flex; align-items: center; gap: 0.5rem; margin: 0.35rem 0; font-size: 0.83rem; color: #374151; }
    .legend-dot  { width: 11px; height: 11px; border-radius: 50%; display: inline-block; flex-shrink: 0; }

    footer { visibility: hidden; }

    /* Dark mode settings
    @media (prefers-color-scheme: dark) {
        .stApp { background-colour: #1a1616; }
        .metric-card, .action-card {
            background: #2b2424;
            box-shadow: 0 2px 8px rgba(0,0,0,0.4);
            }
        .metric-label { color: #9ca3af; }
        .metric-value { color: #f3f4f6; }
        .legend-item { color: #d1d5db; }
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    return pd.read_csv("model results/neighbourhood_clusters.csv").dropna(subset=['centroid_lat', 'centroid_lon'])


@st.cache_data
def load_overlay_data(lat, lon, delta=0.03):
    """
    Loads real infrastructure datasets and extracts coordinates inside a localized 
    bounding box around the viewport center to keep map interactions fluid.
    """
    overlays = {}
    lat_min, lat_max = lat - delta, lat + delta
    lon_min, lon_max = lon - delta, lon + delta

    # Washrooms 
    try:
        washrooms = pd.read_csv("clean data/washrooms.csv").dropna(subset=['lat', 'lon'])
        overlays["washrooms"] = washrooms[
            (washrooms['lat'].between(lat_min, lat_max)) & 
            (washrooms['lon'].between(lon_min, lon_max))
        ]
    except (FileNotFoundError, KeyError):
        overlays["washrooms"] = pd.DataFrame(columns=['lat', 'lon', 'location'])

    # Community Centers & Parks 
    try:
        cc = pd.read_csv("clean data/okai_comm_recs_parks.csv").dropna(subset=['lat', 'lon'])
        overlays["comm_recs_parks"] = cc[
            (cc['lat'].between(lat_min, lat_max)) & 
            (cc['lon'].between(lon_min, lon_max))
        ]
    except (FileNotFoundError, KeyError):
        overlays["comm_recs_parks"] = pd.DataFrame(columns=['lat', 'lon', 'Name'])

    # Libraries
    try:
        libraries = pd.read_csv("clean data/okai_library.csv").dropna(subset=['lat', 'lon'])
        lib_delta = 0.18
        overlays["libraries"] = libraries[
            (libraries['lat'].between(lat - lib_delta, lat + lib_delta)) & 
            (libraries['lon'].between(lon - lib_delta, lon + lib_delta))
        ]
    except (FileNotFoundError, KeyError):
        overlays["libraries"] = pd.DataFrame(columns=['lat', 'lon', 'BranchName'])

    # TTC Stops
    try:
        transit = pd.read_csv("clean data/TTC_routes.csv").dropna(subset=['stop_lat', 'stop_lon'])
        overlays["transit"] = transit[
            (transit['stop_lat'].between(lat_min, lat_max)) & 
            (transit['stop_lon'].between(lon_min, lon_max))
        ]
    except (FileNotFoundError, KeyError):
        overlays["transit"] = pd.DataFrame(columns=['route_long_name', 'stop_name', 'stop_lat', 'stop_lon'])

    #st.sidebar.write(f"Debug - Loaded Libraries Count: {len(overlays['libraries'])}")
    return overlays


# Read main model data frame
df = load_data()

# Sidebar Setup
with st.sidebar:
    st.markdown("### Filters")


    neighborhood_list = sorted(df['neighbourhood_name'].dropna().unique().tolist())
    selected_hoods = st.multiselect(
        "Neighborhoods",
        options=neighborhood_list,
        default=neighborhood_list,
        help="Select which neighborhoods to display"
    )

    st.markdown("---")
    st.markdown("### Risks Score Filter")
    risk_range = st.slider(
        "Filter by Risk Score Range:",
        min_value=0.0, max_value=1.0, value=(0.0, 1.0), step=0.05,
        help="Only display grid sectors falling within this designated range spectrum"
    )

    st.markdown("---")
    st.markdown("### Amenities")
    show_washrooms = st.checkbox("Washrooms")
    show_comm_recs_parks = st.checkbox("Community Centers & Parks")
    show_libraries = st.checkbox("Libraries")
    show_transit = st.checkbox("Public Transit (TTC)")

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem;color:#6b7280;line-height:1.65'>"
        "<strong>About</strong><br>"
        "This tool utilizes AI to identify potential period product desert zones in Toronto by analyzing amenity displacement and accessibility factors."<br>
        "Period product desert zones are areas where access to menstrual products is limited due to poverty, geography, or the absence of nearby distribution points."<br>
        "<strong>Created by WOG <3</strong>"
        "</div>",
        unsafe_allow_html=True
    )

# Selection Validation Check
if not selected_hoods:
    st.warning("Please select at least one neighborhood to view data.")
    st.stop()

# Cascade Filters (Neighborhood bounds + Slider Range Selection)
map_data = df[df['neighbourhood_name'].isin(selected_hoods)]
map_data = map_data[map_data['risk_score'].between(risk_range[0], risk_range[1])]

if map_data.empty:
    st.info("ℹ️ No grid sectors match the current Risk Score slider range. Widen your slider filter values to render the map assets.")
    st.stop()

# Trigger Bounding-Box Overlay Cache Pipeline
default_lat = map_data['centroid_lat'].mean()
default_lon = map_data['centroid_lon'].mean()
overlays = load_overlay_data(default_lat, default_lon)

# Header Element
st.markdown("""
<div class="main-header">
    <h1>OKAI Period</h1>
    <p><strong>Optimizing</strong> for <strong>K</strong>nowledge using <strong>A</strong>rtificial <strong>I</strong>ntelligence</p>
</div>
""", unsafe_allow_html=True)

# Metrics Cards Display Grid
c1, c2, c3, c4 = st.columns(4)
desert_zones_count = len(map_data[map_data['cluster_label'].astype(str).str.lower().str.contains('desert zone', na=False)])

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Neighborhoods</div>
        <div class="metric-value">{map_data['neighbourhood_name'].nunique()}</div>
    </div>""", unsafe_allow_html=True)

with c2:
    # Applies red if any desert zone labels are found, otherwise green
    cls_zone = "danger" if desert_zones_count > 0 else "good"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Desert Zones</div>
        <div class="metric-value {cls_zone}">{desert_zones_count}</div>
    </div>""", unsafe_allow_html=True)

with c3:
    avg_dist = map_data['dist_to_product_km'].mean() if not map_data.empty else 0
    cls_dist = "danger" if avg_dist > 2.0 else "warning" if avg_dist > 1.0 else "good"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Avg. Dist to Products</div>
        <div class="metric-value {cls_dist}">{avg_dist:.2f} km</div>
    </div>""", unsafe_allow_html=True)

with c4:
    avg_risk_score = map_data['risk_score'].mean() if not map_data.empty else 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Avg. Risk Score</div>
        <div class="metric-value">{avg_risk_score:.2f}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

# Map & Legend Section Layout

def get_cluster_color(label):
    lbl = str(label).lower()
    if 'desert zone' in lbl:
        return "#dc2626"       # Red
    elif 'low service' in lbl:
        return "#f97316"       # Orange
    elif 'moderate service' in lbl:
        return "#eab308"       # Yellow
    elif 'well served' in lbl:
        return "#16a34a"       # Green
    return "#9ca3af"           # Muted Gray Fallback


# Full-Width Legend Banner Wrapper Element
st.markdown("""
<div style="display: flex; flex-wrap: wrap; gap: 1.5rem; background: white; padding: 0.8rem 1.25rem; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin-bottom: 0.75rem; font-size: 0.82rem; align-items: center;">
    <span style="color: #6b7280; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin-right: 0.5rem;">Sectors Legend:</span>
    <div style="display: flex; align-items: center; gap: 0.4rem;"><span style="width: 12px; height: 12px; background: #dc2626; border-radius: 50%; display: inline-block;"></span><span style="font-weight: 600; color: #374151;">Desert Zone</span></div>
    <div style="display: flex; align-items: center; gap: 0.4rem;"><span style="width: 12px; height: 12px; background: #f97316; border-radius: 50%; display: inline-block;"></span><span style="font-weight: 600; color: #374151;">Low Service</span></div>
    <div style="display: flex; align-items: center; gap: 0.4rem;"><span style="width: 12px; height: 12px; background: #eab308; border-radius: 50%; display: inline-block;"></span><span style="font-weight: 600; color: #374151;">Moderate Service</span></div>
    <div style="display: flex; align-items: center; gap: 0.4rem;"><span style="width: 12px; height: 12px; background: #16a34a; border-radius: 50%; display: inline-block;"></span><span style="font-weight: 600; color: #374151;">Well Served</span></div>
</div>
""", unsafe_allow_html=True)

#with map_col:
m = folium.Map(
    location=[map_data['centroid_lat'].mean(), map_data['centroid_lon'].mean()],
    zoom_start=13,
)

# Plot Base Risk Grid Markers
for _, row in map_data.iterrows():
    clr = get_cluster_color(row['cluster_label'])
    
    popup_html = f"""
    <div style="font-family: 'Arial', sans-serif; font-size: 12px; color: #333; min-width: 200px;">
        <h4 style="margin: 0 0 8px 0; color: #ef4444; border-bottom: 1px solid #ddd; padding-bottom: 4px;">
            {row['neighbourhood_name']}
        </h4>
        <table style="width: 100%; border-collapse: collapse;">
            <tr><td><b>Label:</b></td><td style="text-align: right;">{row['cluster_label']}</td></tr>
            <tr><td><b>Risk Score:</b></td><td style="text-align: right; color: #dc2626; font-weight: bold;">{row['risk_score']:.2f}</td></tr>
            <tr><td><b>Dist to Products:</b></td><td style="text-align: right;">{row['dist_to_product_km']:.2f} km</td></tr>
        </table>
    </div>
    """

    
    folium.CircleMarker(
        location=[row['centroid_lat'], row['centroid_lon']],
        radius=8,
        color=clr,
        fill=True,
        fill_color=clr,
        fill_opacity=0.65,
        #tooltip=f"{row['neighbourhood_name']} (Cluster {int(row['cluster'])})\nRisk Score: {row['risk_score']:.2f}",
        popup=folium.Popup(popup_html, max_width=300)
        #     f"Vulnerability Score: {score_str}"
        # )
    ).add_to(m)
    

# # Contextual overlays using the centralized load_overlay_data dictionary
if show_washrooms and not overlays['washrooms'].empty:
    for _, row in overlays['washrooms'].iterrows():
        folium.Marker(
            location=[row['lat'], row['lon']],
            tooltip=f"🚻 {row['location']}",
            icon=folium.Icon(color="red", icon="toilet", prefix='fa')
        ).add_to(m)

if show_comm_recs_parks and not overlays['comm_recs_parks'].empty:
    for _, row in overlays['comm_recs_parks'].iterrows():
        folium.Marker(
            location=[row['lat'], row['lon']],
            tooltip=f"🏛️ {row['Name']}",
            icon=folium.Icon(color="blue", icon="building", prefix='fa')
        ).add_to(m)

if show_libraries and not overlays['libraries'].empty:
    for _, row in overlays['libraries'].iterrows():
        folium.Marker(
            location=[row['lat'], row['lon']],
            tooltip=f"📚 {row['BranchName']}",
            icon=folium.Icon(color="green", icon="book", prefix='fa')
        ).add_to(m)

if show_transit and not overlays['transit'].empty:
    for _, row in overlays['transit'].iterrows():
        folium.Marker(
            location=[row['stop_lat'], row['stop_lon']],  # Conformed to TTC structure
            tooltip=f"🚇 {row['stop_name']}",
            icon=folium.Icon(color="orange", icon="subway", prefix='fa')
        ).add_to(m)

st_folium(m, use_container_width=True, height=520, returned_objects=[])
