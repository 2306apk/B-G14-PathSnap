"""Script: generate_notebooks.py

Generates the 4 standard Phase 1 Jupyter Notebooks under notebooks/:
1. 01_eda.ipynb
2. 02_preprocessing.ipynb
3. 03_candidate_generation.ipynb
4. 04_feature_engineering.ipynb
"""

import nbformat as nbf
from pathlib import Path

notebooks_dir = Path(__file__).resolve().parent.parent / "notebooks"
notebooks_dir.mkdir(parents=True, exist_ok=True)

# Helper to create code cell
def code_cell(src: str):
    return nbf.v4.new_code_cell(src.strip())

def md_cell(src: str):
    return nbf.v4.new_markdown_cell(src.strip())

# --- 1. 01_eda.ipynb ---
nb1 = nbf.v4.new_notebook()
nb1.cells = [
    md_cell("# PathSnap: Notebook 01 — Exploratory Data Analysis & Data Quality Audit\n\n"
            "This notebook audits the PathSnap raw dataset: file completeness, GPS sampling intervals, "
            "geographic extents, and road network topology across training and student-test partitions."),
    code_cell("""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path("../src").resolve()))
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathsnap.io import load_trip, list_trips, resolve_data_dir
from pathsnap.geo import get_trip_crs
from pathsnap.graph import RoadNetwork
from pathsnap.visualize import plot_timestamp_gap_distribution

data_dir = resolve_data_dir()
print(f"Dataset root: {data_dir}")
"""),
    md_cell("## 1. Dataset Inventory"),
    code_cell("""
train_trips = list_trips('train', base_dir=data_dir)
test_trips = list_trips('student_test', base_dir=data_dir)
print(f"Total Training Trips: {len(train_trips)}")
print(f"Total Student-Testing Trips: {len(test_trips)}")

# Load inventory summary CSV
inv_df = pd.read_csv('../outputs/csv/dataset_inventory.csv')
inv_df.groupby('partition')[['num_observations', 'num_nodes', 'num_arcs', 'num_route_arcs']].sum()
"""),
    md_cell("## 2. GPS Quality & Sampling Regularity"),
    code_cell("""
gps_df = pd.read_csv('../outputs/csv/gps_quality_audit.csv')
exact_1s = gps_df['exact_1s_gaps'].sum()
total_pts = gps_df['num_observations'].sum()
print(f"Exact 1.0s sampling gaps: {exact_1s:,} ({exact_1s/total_pts*100:.2f}%)")
print(f"Gaps > 10s: {gps_df['gaps_gt_10s'].sum():,}")
print(f"Gaps > 60s: {gps_df['gaps_gt_60s'].sum():,}")
print(f"Max observed time gap: {gps_df['max_time_gap'].max():.1f}s")
"""),
    md_cell("## 3. Road Network Topology & Independence"),
    code_cell("""
net_df = pd.read_csv('../outputs/csv/network_quality_audit.csv')
print(f"Mean Weakly Connected Components: {net_df['num_wcc'].mean():.1f}")
print(f"Mean Largest WCC Fraction: {net_df['largest_wcc_fraction'].mean()*100:.2f}%")
print(f"Mean Consecutive Route Connectivity: {net_df['route_exact_connection_ratio'].mean()*100:.2f}%")
""")
]

with open(notebooks_dir / "01_eda.ipynb", "w") as f:
    nbf.write(nb1, f)

# --- 2. 02_preprocessing.ipynb ---
nb2 = nbf.v4.new_notebook()
nb2.cells = [
    md_cell("# PathSnap: Notebook 02 — Geographical Preprocessing & Metric Projections\n\n"
            "This notebook details coordinate validation, per-trip local UTM selection, straight-line segment "
            "construction, and spatial indexing via Shapely STRtree."),
    code_cell("""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path("../src").resolve()))
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import numpy as np
import pandas as pd
from pathsnap.io import load_trip
from pathsnap.geo import determine_utm_epsg, get_trip_crs, Projector
from pathsnap.graph import RoadNetwork
from pathsnap.visualize import create_folium_map
"""),
    md_cell("## 1. Dynamic Per-Trip UTM Projection Determination"),
    code_cell("""
crs_df = pd.read_csv('../outputs/csv/trip_crs_mapping.csv')
print(f"Identified {crs_df['epsg'].nunique()} distinct UTM EPSG zones across the dataset.")
crs_df.head(10)
"""),
    md_cell("## 2. Road Network and Spatial Index Construction"),
    code_cell("""
trip = load_trip('00000001', partition='train')
crs = get_trip_crs(trip.track)
net = RoadNetwork(trip.trip_id, trip.nodes, trip.arcs, crs)
print(f"Trip {trip.trip_id}: {net.num_nodes} nodes, {net.num_arcs} arcs, CRS: {crs.description}")
"""),
    md_cell("## 3. Interactive Folium Trajectory Map"),
    code_cell("""
# Folium uses [lat, lon] ordering with contrasting blue (GPS) and green (Route) layers
fmap = create_folium_map(net, trip.track, trip.route, output_html_path='../outputs/maps/trip_00000001.html', trip_id=trip.trip_id)
fmap
""")
]

with open(notebooks_dir / "02_preprocessing.ipynb", "w") as f:
    nbf.write(nb2, f)

# --- 3. 03_candidate_generation.ipynb ---
nb3 = nbf.v4.new_notebook()
nb3.cells = [
    md_cell("# PathSnap: Notebook 03 — Candidate Generation & Radius Justification\n\n"
            "Evaluates point-to-reference-arc distances on training data, sweeps candidate radii, "
            "and justifies selecting search radius R = 50.0m."),
    code_cell("""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path("../src").resolve()))
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
"""),
    md_cell("## 1. Candidate Search Radius Sweep Results"),
    code_cell("""
sweep_df = pd.read_csv('../outputs/csv/candidate_radius_sweep.csv')
sweep_df
"""),
    md_cell("## 2. Candidate Recall vs Search Radius Curve"),
    code_cell("""
plt.figure(figsize=(8, 4.5))
plt.plot(sweep_df['radius_meters'], sweep_df['recall'] * 100, marker='o', color='#1f77b4', linewidth=2)
plt.axvline(50.0, color='green', linestyle='--', label='Selected R = 50.0m')
plt.title('Candidate Recall (%) vs Search Radius (m)')
plt.xlabel('Search Radius R (m)')
plt.ylabel('Recall (%)')
plt.ylim(75, 101)
plt.legend()
plt.tight_layout()
plt.show()
""")
]

with open(notebooks_dir / "03_candidate_generation.ipynb", "w") as f:
    nbf.write(nb3, f)

# --- 4. 04_feature_engineering.ipynb ---
nb4 = nbf.v4.new_notebook()
nb4.cells = [
    md_cell("# PathSnap: Notebook 04 — Feature Engineering & Leakage Safeguards\n\n"
            "Documents the 14-feature schema, past-only kinematic extraction, target label association, "
            "and candidate miss rate accounting."),
    code_cell("""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path("../src").resolve()))
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathsnap.features import FEATURE_COLUMNS, FEATURE_SCHEMA
"""),
    md_cell("## 1. Feature Schema & Definitions"),
    code_cell("""
schema_records = []
for k, v in FEATURE_SCHEMA.items():
    schema_records.append({'Feature': k, 'Unit': v['unit'], 'Description': v['description']})
pd.DataFrame(schema_records)
"""),
    md_cell("## 2. Feature Data Inspection"),
    code_cell("""
# Inspect processed training features sample
sample_df = pd.read_csv('../data/processed/train_features_sample_1000.csv')
print(f"Sample shape: {sample_df.shape}")
sample_df[['observation_id', 'candidate_rank', 'dist_to_arc', 'heading_diff', 'speed_est', 'label']].head(10)
"""),
    md_cell("## 3. Feature Distribution by Target Class"),
    code_cell("""
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(data=sample_df, x='dist_to_arc', hue='label', bins=30, stat='density', common_norm=False, ax=ax1, palette={0: '#d62728', 1: '#2ca02c'})
ax1.set_title('Perpendicular Distance: Match (1) vs Non-Match (0)')

sns.histplot(data=sample_df, x='heading_diff', hue='label', bins=30, stat='density', common_norm=False, ax=ax2, palette={0: '#d62728', 1: '#2ca02c'})
ax2.set_title('Heading Difference: Match (1) vs Non-Match (0)')
plt.tight_layout()
plt.show()
""")
]

with open(notebooks_dir / "04_feature_engineering.ipynb", "w") as f:
    nbf.write(nb4, f)

print("Successfully generated all 4 Phase 1 notebooks under notebooks/!")

