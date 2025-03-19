import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Set Streamlit page config
st.set_page_config(page_title="FPL Predictions", layout="wide", initial_sidebar_state="expanded")

# File paths
assists_path = "GW_highest_Predicted_assists.csv"
goals_path = "GW_highest_Predicted_goals.csv"
cards_path = "GW25_Predicted_Cards.csv"
clean_sheets_path = "GW25_Predicted_Clean_Sheets.csv"

# Custom Streamlit styles
st.markdown(
    """
    <style>
        .stApp {
            background-color: #f5f5f5;
        }
        .stTitle {
            color: #38003c;  /* Premier League purple */
            text-align: center;
            font-weight: bold;
        }
        .metric-card {
            background-color: white;
            border-radius: 5px;
            padding: 15px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            text-align: center;
            height: 100%;
        }
        .metric-value {
            font-size: 32px;
            font-weight: bold;
            color: #38003c;
        }
        .metric-label {
            font-size: 14px;
            color: #555;
        }
        .section-header {
            margin-top: 20px;
            margin-bottom: 10px;
            padding: 5px 0;
            border-bottom: 2px solid #38003c;
            color: #38003c;
        }
        .highlight {
            background-color: #e6f7ff;
            padding: 10px;
            border-radius: 5px;
            border-left: 3px solid #38003c;
        }
        .small-text {
            font-size: 12px;
            color: #666;
        }
        .team-filter {
            background-color: white;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        /* FPL-inspired color scheme */
        .goal-color { color: #00ff85; }
        .assist-color { color: #04f5ff; }
        .card-color { color: #ffcb05; }
        .cs-color { color: #38003c; }
        
        /* Position colors */
        .pos-GKP { background-color: #ebff00; color: black; }
        .pos-DEF { background-color: #00ff85; color: black; }
        .pos-MID { background-color: #04f5ff; color: black; }
        .pos-FWD { background-color: #ff3977; color: white; }
        
        /* Make the position labels small circles */
        .position-pill {
            display: inline-block;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            text-align: center;
            line-height: 24px;
            font-weight: bold;
            font-size: 12px;
            margin-right: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Prediction generation functions
def generate_goals_predictions():
    """Generate goals predictions if the CSV doesn't exist"""
    data_path = "data/2024-25/gws/merged_gw_cleaned.csv"
    
    try:
        # Check if data file exists
        if not os.path.exists(data_path):
            st.error(f"Data file not found: {data_path}")
            return pd.DataFrame(columns=["name", "team", "GW", "predicted_goals"])
        
        # Load the data
        data = pd.read_csv(data_path, engine='python', on_bad_lines='skip')
        
        # Use GW 1-24 for training
        train_data = data[(data["GW"] >= 1) & (data["GW"] <= 24)]
        
        if train_data.empty:
            st.error("No training data found for GW 1-24.")
            return pd.DataFrame(columns=["name", "team", "GW", "predicted_goals"])
        
        # Select features for goals prediction
        features = [
            "minutes", "xP", "threat", "bps", "transfers_in"
        ]
        
        # Add expected_goals if available
        if "expected_goals" in train_data.columns:
            features.append("expected_goals")
        
        # Ensure all needed columns exist
        for col in features + ["goals_scored"]:
            if col not in train_data.columns:
                st.warning(f"Column {col} not found in data. Using simplified model.")
                # Fallback to basic features
                features = [col for col in ["minutes", "bps", "threat"] if col in train_data.columns]
                if not features:
                    return pd.DataFrame(columns=["name", "team", "GW", "predicted_goals"])
                break
        
        target = "goals_scored"
        
        # Prepare data
        train_data = train_data.dropna(subset=features + [target])
        for col in features + [target]:
            train_data[col] = pd.to_numeric(train_data[col], errors='coerce')
        
        # Split and train
        X = train_data[features]
        y = train_data[target]
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Prepare GW 25 data
        numeric_columns = train_data.select_dtypes(include=[np.number]).columns.tolist()
        gw25_data = train_data.groupby(["name", "team"])[numeric_columns].mean().reset_index()
        gw25_data["GW"] = 25
        
        # Predict and filter
        X_future = gw25_data[features]
        gw25_data["predicted_goals"] = model.predict(X_future)
        gw25_data = gw25_data[gw25_data["predicted_goals"] > 0]
        gw25_data_sorted = gw25_data.sort_values(by="predicted_goals", ascending=False)
        
        # Save results
        result_df = gw25_data_sorted[["name", "team", "GW", "predicted_goals"]]
        result_df.to_csv(goals_path, index=False)
        return result_df
        
    except Exception as e:
        st.error(f"Error generating goals predictions: {e}")
        return pd.DataFrame(columns=["name", "team", "GW", "predicted_goals"])

def generate_assists_predictions():
    """Generate assists predictions if the CSV doesn't exist"""
    data_path = "data/2024-25/gws/merged_gw_cleaned.csv"
    
    try:
        # Check if data file exists
        if not os.path.exists(data_path):
            st.error(f"Data file not found: {data_path}")
            return pd.DataFrame(columns=["name", "team", "GW", "predicted_assists"])
        
        # Load the data
        data = pd.read_csv(data_path, engine='python', on_bad_lines='skip')
        
        # Use GW 1-24 for training
        train_data = data[(data["GW"] >= 1) & (data["GW"] <= 24)]
        
        if train_data.empty:
            st.error("No training data found for GW 1-24.")
            return pd.DataFrame(columns=["name", "team", "GW", "predicted_assists"])
        
        # Select features for assists prediction
        assists_features = [
            "minutes", "creativity", "bps", "total_points", "influence"
        ]
        
        # Add expected_assists if available
        if "expected_assists" in train_data.columns:
            assists_features.append("expected_assists")
        
        # Add expected_goal_involvements if available
        if "expected_goal_involvements" in train_data.columns:
            assists_features.append("expected_goal_involvements")
        
        # Ensure all needed columns exist
        for col in assists_features + ["assists"]:
            if col not in train_data.columns:
                st.warning(f"Column {col} not found in data. Using simplified model.")
                # Fallback to basic features
                assists_features = [col for col in ["minutes", "bps", "creativity"] if col in train_data.columns]
                if not assists_features:
                    return pd.DataFrame(columns=["name", "team", "GW", "predicted_assists"])
                break
        
        target_assists = "assists"
        
        # Prepare data
        train_data = train_data.dropna(subset=assists_features + [target_assists])
        for col in assists_features + [target_assists]:
            train_data[col] = pd.to_numeric(train_data[col], errors='coerce')
        
        # One-hot encode position if available
        if "position" in train_data.columns:
            train_data = pd.get_dummies(train_data, columns=["position"])
            position_cols = [col for col in train_data.columns if col.startswith("position_")]
            assists_features.extend(position_cols)
        
        # Split and train
        X_assists = train_data[assists_features]
        y_assists = train_data[target_assists]
        model_assists = RandomForestRegressor(n_estimators=100, random_state=42)
        model_assists.fit(X_assists, y_assists)
        
        # Prepare GW 25 data
        gw25_data_assists = train_data.groupby(["name", "team"])[assists_features].mean().reset_index()
        gw25_data_assists["GW"] = 25
        
        # Predict and sort
        X_future_assists = gw25_data_assists[assists_features]
        gw25_data_assists["predicted_assists"] = model_assists.predict(X_future_assists)
        gw25_data_assists["predicted_assists"] = np.clip(gw25_data_assists["predicted_assists"], 0, None)
        gw25_data_assists_sorted = gw25_data_assists.sort_values(by="predicted_assists", ascending=False)
        
        # Save results
        result_df = gw25_data_assists_sorted[["name", "team", "GW", "predicted_assists"]]
        result_df.to_csv(assists_path, index=False)
        return result_df
        
    except Exception as e:
        st.error(f"Error generating assists predictions: {e}")
        return pd.DataFrame(columns=["name", "team", "GW", "predicted_assists"])

# Position data for players (can be loaded from another file if available)
def load_position_data():
    position_path = "data/2024-25/players_raw.csv"
    if os.path.exists(position_path):
        try:
            players_raw = pd.read_csv(position_path)
            
            # Check if element_type is numeric (as expected)
            if pd.api.types.is_numeric_dtype(players_raw["element_type"]):
                # Map position codes to names
                position_mapping = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
                players_raw["position"] = players_raw["element_type"].map(position_mapping)
            else:
                # If element_type is not numeric, ensure it's a string and use directly
                players_raw["position"] = players_raw["element_type"].astype(str)
                st.sidebar.warning("Position data format unexpected - using raw values")
                
            # Create full name
            players_raw["name"] = players_raw["first_name"] + " " + players_raw["second_name"]
            
            # Ensure position is a string
            players_raw["position"] = players_raw["position"].astype(str)
            
            # Log what positions we found
            unique_positions = players_raw["position"].unique()
            print(f"Found positions: {unique_positions}")
            
            return players_raw[["name", "position", "team", "now_cost"]]
        except Exception as e:
            st.sidebar.warning(f"Could not process position data: {e}")
    return None

# Function to create best XI visualization
def create_best_xi(merged_data, positions_data=None):
    # Add position data if available
    if positions_data is not None:
        try:
            # Ensure position column is string type
            positions_data["position"] = positions_data["position"].astype(str)
            merged_data = pd.merge(merged_data, positions_data[["name", "position"]], on="name", how="left")
            has_position = "position" in merged_data.columns
        except Exception as e:
            print(f"Error merging position data: {e}")
            has_position = False
    else:
        has_position = "position" in merged_data.columns
    
    # If position column exists, ensure it's string type
    if has_position:
        merged_data["position"] = merged_data["position"].astype(str)
    else:
        # If no position data available, create a position column with estimates
        print("No position data available. Creating estimated positions.")
        # Roughly estimate positions based on stats
        conditions = [
            (merged_data['predicted_clean_sheets'] >= 0.4),  # Likely GKP or DEF
            (merged_data['predicted_goals'] < 0.3) & (merged_data['predicted_assists'] >= 0.3),  # Likely DEF
            (merged_data['predicted_goals'] >= 0.3) & (merged_data['predicted_goals'] < 0.5),  # Likely MID
            (merged_data['predicted_goals'] >= 0.5)  # Likely FWD
        ]
        position_values = ['GKP', 'DEF', 'MID', 'FWD']
        merged_data['position'] = np.select(conditions, position_values, default='MID')
        has_position = True
    
    # Continue with the rest of the function...
    # Filter out any rows with non-standard positions
    standard_positions = ['GKP', 'DEF', 'MID', 'FWD']
    merged_data = merged_data[merged_data['position'].isin(standard_positions)]
    
    # If no valid position data remains, return None
    if len(merged_data) == 0:
        return None, pd.DataFrame()
    
    # Calculate a combined score
    # Goals and assists are positive, cards are negative, clean sheets matter for defenders
    merged_data['player_score'] = (
        merged_data['predicted_goals'] * 4 + 
        merged_data['predicted_assists'] * 3 - 
        merged_data['predicted_cards'] * 2
    )
    
    # For defenders and keepers, add clean sheet bonus
    if 'position' in merged_data.columns:
        merged_data.loc[merged_data['position'].isin(['GKP', 'DEF']), 'player_score'] += (
            merged_data['predicted_clean_sheets'] * 4
        )
    
    # Get positions array for easier access
    positions = merged_data['position'] if 'position' in merged_data.columns else None
    
    # Select top players for each position for a 4-4-2 formation
    if positions is not None:
        best_gkp = merged_data[positions == 'GKP'].nlargest(1, 'player_score')
        best_def = merged_data[positions == 'DEF'].nlargest(4, 'player_score')
        best_mid = merged_data[positions == 'MID'].nlargest(4, 'player_score')
        best_fwd = merged_data[positions == 'FWD'].nlargest(2, 'player_score')
    else:
        # If no position data, just get top 11 players
        best_xi = merged_data.nlargest(11, 'player_score')
        return None, best_xi
    
    # Combine into best XI
    best_xi = pd.concat([best_gkp, best_def, best_mid, best_fwd])
    
    # Create a formation diagram
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Setup the pitch
    pitch_color = '#B0E0B0'  # Light green
    ax.set_facecolor(pitch_color)
    
    # Draw field markings
    # Center circle
    center_circle = plt.Circle((5, 5), 1, fill=False, color='white', linewidth=2)
    ax.add_artist(center_circle)
    
    # Field boundaries
    ax.plot([0, 10], [0, 0], color='white', linewidth=2)  # Bottom
    ax.plot([0, 10], [10, 10], color='white', linewidth=2)  # Top
    ax.plot([0, 0], [0, 10], color='white', linewidth=2)  # Left
    ax.plot([10, 10], [0, 10], color='white', linewidth=2)  # Right
    
    # Goal areas
    ax.plot([3, 7], [0, 0], color='white', linewidth=4)  # Bottom goal
    ax.plot([3, 7], [10, 10], color='white', linewidth=4)  # Top goal
    
    # Position coordinates for 4-4-2 formation (x, y)
    positions_coords = {
        'GKP': [(5, 1)],
        'DEF': [(2, 2.5), (4, 2.5), (6, 2.5), (8, 2.5)],
        'MID': [(2, 5), (4, 5), (6, 5), (8, 5)],
        'FWD': [(3.5, 8), (6.5, 8)]
    }
    
    # Colors for different positions
    position_colors = {
        'GKP': '#ebff00',  # Yellow
        'DEF': '#00ff85',  # Green
        'MID': '#04f5ff',  # Blue
        'FWD': '#ff3977'   # Red
    }
    
    # Plot players in their positions
    for position, coords_list in positions_coords.items():
        players = best_xi[best_xi['position'] == position]
        
        for i, (_, player_data) in enumerate(players.iterrows()):
            if i >= len(coords_list):
                break
                
            x, y = coords_list[i]
            
            # Plot player circle
            player_circle = plt.Circle(
                (x, y), 0.6, 
                color=position_colors[position], 
                alpha=0.8, 
                edgecolor='white', 
                linewidth=2
            )
            ax.add_artist(player_circle)
            
            # Get last name for better display
            last_name = player_data['name'].split()[-1] if len(player_data['name'].split()) > 0 else player_data['name']
            
            # Add player name
            ax.text(
                x, y, 
                last_name, 
                ha='center', 
                va='center', 
                fontsize=9, 
                fontweight='bold', 
                color='black'
            )
            
            # Add expected contributions based on position
            if position in ['FWD', 'MID']:
                ax.text(
                    x, y-0.3, 
                    f"G: {player_data.get('predicted_goals', 0):.1f} A: {player_data.get('predicted_assists', 0):.1f}", 
                    ha='center', 
                    va='center', 
                    fontsize=7, 
                    color='black'
                )
            else:  # DEF, GKP
                ax.text(
                    x, y-0.3, 
                    f"CS: {player_data.get('predicted_clean_sheets', 0):.2f}", 
                    ha='center', 
                    va='center', 
                    fontsize=7, 
                    color='black'
                )
    
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 10.5)
    ax.set_title("Predicted Best XI - GW25", fontsize=16, fontweight='bold')
    ax.axis('off')
    
    return fig, best_xi

# Function to create correlation heatmap
def create_correlation_heatmap(merged_data):
    # Select numeric columns for correlation
    numeric_cols = merged_data.select_dtypes(include=[np.number]).columns.tolist()
    
    # Drop GW column if present
    if 'GW' in numeric_cols:
        numeric_cols.remove('GW')
    
    # Compute correlation matrix
    corr_matrix = merged_data[numeric_cols].corr()
    
    # Create custom colormap (FPL-inspired colors)
    cmap = LinearSegmentedColormap.from_list(
        'fpl_colors', 
        ['#ff3977', '#FFFFFF', '#00ff85'], 
        N=256
    )
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, 
        mask=mask,
        annot=True, 
        fmt='.2f', 
        cmap=cmap, 
        vmin=-1, 
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8},
        ax=ax
    )
    
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    
    return fig

# Function to visualize predictions by position
def create_position_chart(merged_data, metric='predicted_goals'):
    if 'position' not in merged_data.columns:
        return None
    
    # Ensure positions are strings to avoid comparison errors
    merged_data['position'] = merged_data['position'].astype(str)
    
    # Define position order
    position_order = {"GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}
    
    try:
        # Calculate averages by position
        pos_avg = merged_data.groupby('position')[metric].mean().reset_index()
        
        # Sort positions in the standard order
        pos_avg['order'] = pos_avg['position'].map(lambda x: position_order.get(x, 999))
        pos_avg = pos_avg.sort_values('order')
        
        # Create the chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Colors for positions
        colors = {
            'GKP': '#ebff00',
            'DEF': '#00ff85',
            'MID': '#04f5ff',
            'FWD': '#ff3977'
        }
        
        # Create bar chart
        bars = ax.bar(
            pos_avg['position'], 
            pos_avg[metric], 
            color=[colors.get(pos, '#38003c') for pos in pos_avg['position']]
        )
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2., 
                height + 0.02,
                f'{height:.2f}', 
                ha='center', 
                fontsize=12, 
                fontweight='bold'
            )
        
        # Customize chart
        metric_labels = {
            'predicted_goals': 'Average Predicted Goals',
            'predicted_assists': 'Average Predicted Assists',
            'predicted_cards': 'Average Predicted Cards',
            'predicted_clean_sheets': 'Average Predicted Clean Sheet Probability'
        }
        
        ax.set_title(f'{metric_labels.get(metric, metric)} by Position', fontsize=15, fontweight='bold')
        ax.set_xlabel('Position', fontsize=12)
        ax.set_ylabel(metric_labels.get(metric, metric), fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add Premier League styling
        fig.set_facecolor('#f5f5f5')
        ax.set_facecolor('#f5f5f5')
        
        return fig
    except Exception as e:
        print(f"Error creating position chart: {e}")
        return None

# Load data and check if files exist
files_exist = True
missing_files = []

# Check for essential files
if not os.path.exists(assists_path):
    missing_files.append("Assists predictions file")
    files_exist = False
    
if not os.path.exists(goals_path):
    missing_files.append("Goals predictions file")
    files_exist = False

# Optional files
cards_exist = os.path.exists(cards_path)
clean_sheets_exist = os.path.exists(clean_sheets_path)

if not cards_exist:
    st.sidebar.warning("Cards predictions file not found. Will continue without cards data.")
    
if not clean_sheets_exist:
    st.sidebar.warning("Clean sheets predictions file not found. Will continue without clean sheets data.")

if not files_exist:
    st.error(f"Missing essential files: {', '.join(missing_files)}")
    st.markdown(
        """
        Please run the prediction models to generate the necessary CSV files:
        ```
        python updated-train-goals.py
        python updated-train-assists.py
        ```
        
        Optionally, you can also run:
        ```
        python updated-train-cards.py
        python fixed-clean-sheets-simple.py
        ```
        """
    )
    # Load all available data
# Check if files exist, generate if needed
if not os.path.exists(goals_path):
    st.info("Generating goals predictions...")
    goals_df = generate_goals_predictions()
else:
    goals_df = pd.read_csv(goals_path)

if not os.path.exists(assists_path):
    st.info("Generating assists predictions...")
    assists_df = generate_assists_predictions()
else:
    assists_df = pd.read_csv(assists_path)

if cards_exist:
    cards_df = pd.read_csv(cards_path)
else:
    cards_df = pd.DataFrame(columns=["name", "team", "GW", "predicted_cards"])

if clean_sheets_exist:
    clean_sheets_df = pd.read_csv(clean_sheets_path)
else:
    clean_sheets_df = pd.DataFrame(columns=["name", "team", "GW", "predicted_clean_sheets"])

# Ensure column names are consistent for merging
if 'predicted_assists' not in assists_df.columns and 'assists' in assists_df.columns:
    assists_df.rename(columns={'assists': 'predicted_assists'}, inplace=True)

if 'predicted_goals' not in goals_df.columns and 'goals_scored' in goals_df.columns:
    goals_df.rename(columns={'goals_scored': 'predicted_goals'}, inplace=True)

if cards_exist and 'predicted_cards' not in cards_df.columns and 'total_cards' in cards_df.columns:
    cards_df.rename(columns={'total_cards': 'predicted_cards'}, inplace=True)

if clean_sheets_exist and 'predicted_clean_sheets' not in clean_sheets_df.columns and 'clean_sheets' in clean_sheets_df.columns:
    clean_sheets_df.rename(columns={'clean_sheets': 'predicted_clean_sheets'}, inplace=True)

# Load position data regardless of clean sheets existence
positions_df = load_position_data()

# ---- DASHBOARD HEADER ----
st.title("⚽ Fantasy Premier League AI Predictions")
st.write("Advanced analytics and predictions for Gameweek 25 using machine learning models")

# ---- SIDEBAR FILTERS ----
st.sidebar.header("Filters")

# Team filter
teams = sorted(set(assists_df["team"].unique()) | set(goals_df["team"].unique()))
if cards_exist:
    teams = sorted(set(teams) | set(cards_df["team"].unique()))
if clean_sheets_exist:
    teams = sorted(set(teams) | set(clean_sheets_df["team"].unique()))

selected_team = st.sidebar.selectbox("Team", ["All Teams"] + teams)

# Fix position types before sorting
if positions_df is not None:
    # Convert position values to strings to avoid comparison errors
    positions_df["position"] = positions_df["position"].astype(str)
    
    # Define preferred position order
    position_order = {"GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}
    
    # Get unique positions
    unique_positions = positions_df["position"].unique()
    
    # Try to sort positions safely
    try:
        # Sort by custom position order if possible
        ordered_positions = sorted(
            [pos for pos in unique_positions if pos in position_order],
            key=lambda x: position_order.get(x, 999)
        )
        
        # Add any other positions that don't match the standard ones
        other_positions = [pos for pos in unique_positions if pos not in position_order]
        positions = ordered_positions + sorted(other_positions)
    except:
        # Fallback if sorting fails
        positions = ["GKP", "DEF", "MID", "FWD"]
        st.sidebar.warning("Could not sort positions. Using default order.")
    
    selected_position = st.sidebar.selectbox("Position", ["All Positions"] + positions)
else:
    selected_position = "All Positions"

# Metric threshold sliders
st.sidebar.header("Thresholds")

# Add regenerate button
if st.sidebar.button("Regenerate Predictions"):
    st.sidebar.info("Regenerating predictions...")
    goals_df = generate_goals_predictions()
    assists_df = generate_assists_predictions()
    st.sidebar.success("Predictions regenerated! Refresh the page to see updates.")

    
    min_goals = st.sidebar.slider("Minimum Goals", 0.0, 1.0, 0.0, 0.1)
    min_assists = st.sidebar.slider("Minimum Assists", 0.0, 1.0, 0.0, 0.1)
    
    if cards_exist:
        max_cards = st.sidebar.slider("Maximum Cards", 0.0, 1.0, 1.0, 0.1)
    else:
        max_cards = 1.0
        
    if clean_sheets_exist:
        min_clean_sheets = st.sidebar.slider("Minimum Clean Sheet Probability", 0.0, 0.6, 0.0, 0.05)
    else:
        min_clean_sheets = 0.0
    
    # Merge all data for comprehensive analysis
    # Start with goals and assists which are essential
    merged_df = pd.merge(
        goals_df[["name", "team", "predicted_goals"]], 
        assists_df[["name", "predicted_assists"]], 
        on="name", 
        how="outer"
    )
    
    # Add cards if available
    if cards_exist:
        merged_df = pd.merge(
            merged_df,
            cards_df[["name", "predicted_cards"]],
            on="name",
            how="outer"
        )
    else:
        merged_df["predicted_cards"] = 0
    
    # Add clean sheets if available
    if clean_sheets_exist:
        merged_df = pd.merge(
            merged_df,
            clean_sheets_df[["name", "predicted_clean_sheets"]],
            on="name",
            how="outer"
        )
    else:
        merged_df["predicted_clean_sheets"] = 0
    
    # Add position data if available
    if positions_df is not None:
        merged_df = pd.merge(
            merged_df,
            positions_df[["name", "position"]],
            on="name",
            how="left"
        )
    
    # Fill missing values with zeros
    merged_df.fillna(0, inplace=True)
    
    # Apply filters
    if selected_team != "All Teams":
        filtered_merged = merged_df[merged_df["team"] == selected_team]
    else:
        filtered_merged = merged_df
        
    if selected_position != "All Positions" and "position" in filtered_merged.columns:
        filtered_merged = filtered_merged[filtered_merged["position"] == selected_position]
    
    # Apply threshold filters
    filtered_merged = filtered_merged[
        (filtered_merged["predicted_goals"] >= min_goals) & 
        (filtered_merged["predicted_assists"] >= min_assists) & 
        (filtered_merged["predicted_cards"] <= max_cards)
    ]
    
    if clean_sheets_exist:
        filtered_merged = filtered_merged[filtered_merged["predicted_clean_sheets"] >= min_clean_sheets]
    
    # Create combined score
    filtered_merged["total_contribution"] = (
        filtered_merged["predicted_goals"] * 4 + 
        filtered_merged["predicted_assists"] * 3 - 
        filtered_merged["predicted_cards"] * 2
    )
    
    # Add clean sheet bonus for defenders/goalkeepers
    if "position" in filtered_merged.columns:
        def_gk_mask = filtered_merged["position"].isin(["DEF", "GKP"])
        filtered_merged.loc[def_gk_mask, "total_contribution"] += filtered_merged.loc[def_gk_mask, "predicted_clean_sheets"] * 4
    
    # ---- KEY METRICS ROW ----
    st.markdown('<div class="section-header"><h2>Key Metrics</h2></div>', unsafe_allow_html=True)
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{len(filtered_merged)}</div>
                <div class="metric-label">Players Meeting Criteria</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with metric_col2:
        top_scorer = filtered_merged.nlargest(1, "predicted_goals")
        if not top_scorer.empty:
            top_scorer_name = top_scorer.iloc[0]["name"]
            top_scorer_goals = top_scorer.iloc[0]["predicted_goals"]
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value goal-color">{top_scorer_goals:.1f}</div>
                    <div class="metric-label">Top Scorer: {top_scorer_name}</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">N/A</div>
                    <div class="metric-label">Top Scorer</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
    
    with metric_col3:
        top_assister = filtered_merged.nlargest(1, "predicted_assists")
        if not top_assister.empty:
            top_assister_name = top_assister.iloc[0]["name"]
            top_assister_assists = top_assister.iloc[0]["predicted_assists"]
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value assist-color">{top_assister_assists:.1f}</div>
                    <div class="metric-label">Top Assister: {top_assister_name}</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">N/A</div>
                    <div class="metric-label">Top Assister</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
    
    with metric_col4:
        if clean_sheets_exist:
            top_cs = filtered_merged.nlargest(1, "predicted_clean_sheets")
            if not top_cs.empty:
                top_cs_name = top_cs.iloc[0]["name"]
                top_cs_prob = top_cs.iloc[0]["predicted_clean_sheets"]
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-value cs-color">{top_cs_prob:.2f}</div>
                        <div class="metric-label">Highest CS Prob: {top_cs_name}</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-value">N/A</div>
                        <div class="metric-label">Highest CS Probability</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        else:
            mvp = filtered_merged.nlargest(1, "total_contribution")
            if not mvp.empty:
                mvp_name = mvp.iloc[0]["name"]
                mvp_contribution = mvp.iloc[0]["total_contribution"]
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-value">{mvp_contribution:.1f}</div>
                        <div class="metric-label">Highest Contribution: {mvp_name}</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-value">N/A</div>
                        <div class="metric-label">Highest Contribution</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
    
    # ---- TOP PREDICTIONS TABLES ----
    st.markdown('<div class="section-header"><h2>Top Predictions</h2></div>', unsafe_allow_html=True)
    
    # Create tabs for visualization types
    tabs = st.tabs(["Tables", "Best XI", "Position Analysis"])
    
    with tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Top Predicted Goals")
            goals_display = filtered_merged.sort_values(by="predicted_goals", ascending=False)[["name", "team", "predicted_goals"]].head(10)
            
            if "position" in filtered_merged.columns:
                goals_display = pd.merge(
                    goals_display,
                    filtered_merged[["name", "position"]],
                    on="name",
                    how="left"
                )
                
                # Format with colored position indicators
                def format_row_with_position(row):
                    pos = row["position"]
                    name = row["name"]
                    pos_class = f"pos-{pos}" if pos in ["GKP", "DEF", "MID", "FWD"] else ""
                    return f'<span class="position-pill {pos_class}">{pos[0]}</span> {name}'
                
                goals_display["Display Name"] = goals_display.apply(format_row_with_position, axis=1)
                st.write(goals_display[["Display Name", "team", "predicted_goals"]].to_html(escape=False, index=False), unsafe_allow_html=True)
            else:
                st.dataframe(
                    goals_display.style.format({"predicted_goals": "{:.2f}"}),
                    use_container_width=True
                )
        
        with col2:
            st.subheader("🅰️ Top Predicted Assists")
            assists_display = filtered_merged.sort_values(by="predicted_assists", ascending=False)[["name", "team", "predicted_assists"]].head(10)
            
            if "position" in filtered_merged.columns:
                assists_display = pd.merge(
                    assists_display,
                    filtered_merged[["name", "position"]],
                    on="name",
                    how="left"
                )
                
                assists_display["Display Name"] = assists_display.apply(format_row_with_position, axis=1)
                st.write(assists_display[["Display Name", "team", "predicted_assists"]].to_html(escape=False, index=False), unsafe_allow_html=True)
            else:
                st.dataframe(
                    assists_display.style.format({"predicted_assists": "{:.2f}"}),
                    use_container_width=True
                )
        
        col3, col4 = st.columns(2)
        
        with col3:
            if cards_exist:
                st.subheader("🟨 Top Predicted Cards")
                cards_display = filtered_merged.sort_values(by="predicted_cards", ascending=False)[["name", "team", "predicted_cards"]].head(10)
                
                if "position" in filtered_merged.columns:
                    cards_display = pd.merge(
                        cards_display,
                        filtered_merged[["name", "position"]],
                        on="name",
                        how="left"
                    )
                    
                    cards_display["Display Name"] = cards_display.apply(format_row_with_position, axis=1)
                    st.write(cards_display[["Display Name", "team", "predicted_cards"]].to_html(escape=False, index=False), unsafe_allow_html=True)
                else:
                    st.dataframe(
                        cards_display.style.format({"predicted_cards": "{:.2f}"}),
                        use_container_width=True
                    )
            else:
                st.write("Cards prediction data not available")
        
        with col4:
            if clean_sheets_exist:
                st.subheader("🧤 Top Predicted Clean Sheets")
                cs_display = filtered_merged.sort_values(by="predicted_clean_sheets", ascending=False)[["name", "team", "predicted_clean_sheets"]].head(10)
                
                if "position" in filtered_merged.columns:
                    cs_display = pd.merge(
                        cs_display,
                        filtered_merged[["name", "position"]],
                        on="name",
                        how="left"
                    )
                    
                    cs_display["Display Name"] = cs_display.apply(format_row_with_position, axis=1)
                    st.write(cs_display[["Display Name", "team", "predicted_clean_sheets"]].to_html(escape=False, index=False), unsafe_allow_html=True)
                else:
                    st.dataframe(
                        cs_display.style.format({"predicted_clean_sheets": "{:.2f}"}),
                        use_container_width=True
                    )
            else:
                st.write("Clean sheets prediction data not available")
    
    with tabs[1]:
        # Create best XI visualization
        best_xi_fig, best_xi_data = create_best_xi(filtered_merged, positions_df)
        
        if best_xi_fig is not None:
            st.pyplot(best_xi_fig)
        else:
            st.warning("Position data not available for creating Best XI visualization")
        
        st.subheader("Best XI Players")
        if "position" in best_xi_data.columns:
            display_cols = ["name", "team", "position", "predicted_goals", "predicted_assists", "predicted_cards"]
            if clean_sheets_exist:
                display_cols.append("predicted_clean_sheets")
            
            st.dataframe(
                best_xi_data[display_cols].style.format({
                    "predicted_goals": "{:.2f}",
                    "predicted_assists": "{:.2f}",
                    "predicted_cards": "{:.2f}",
                    "predicted_clean_sheets": "{:.2f}"
                }),
                use_container_width=True
            )
        else:
            st.warning("Position data not available for Best XI table")
    
    with tabs[2]:
        # Position Analysis tab
        if "position" in filtered_merged.columns:
            metric_tabs = st.tabs(["Goals", "Assists", "Cards", "Clean Sheets"])
            
            with metric_tabs[0]:
                goals_by_pos_fig = create_position_chart(filtered_merged, 'predicted_goals')
                st.pyplot(goals_by_pos_fig)
                
                # Show top players by position
                st.subheader("Top Goal Scorers by Position")
                positions_list = filtered_merged["position"].unique()
                
                for pos in positions_list:
                    pos_players = filtered_merged[filtered_merged["position"] == pos]
                    top_pos_players = pos_players.nlargest(5, "predicted_goals")[["name", "team", "predicted_goals"]]
                    
                    if not top_pos_players.empty:
                        st.write(f"**{pos}**")
                        st.dataframe(
                            top_pos_players.style.format({"predicted_goals": "{:.2f}"}),
                            use_container_width=True
                        )
            
            with metric_tabs[1]:
                assists_by_pos_fig = create_position_chart(filtered_merged, 'predicted_assists')
                st.pyplot(assists_by_pos_fig)
                
                # Show top players by position
                st.subheader("Top Assisters by Position")
                for pos in positions_list:
                    pos_players = filtered_merged[filtered_merged["position"] == pos]
                    top_pos_players = pos_players.nlargest(5, "predicted_assists")[["name", "team", "predicted_assists"]]
                    
                    if not top_pos_players.empty:
                        st.write(f"**{pos}**")
                        st.dataframe(
                            top_pos_players.style.format({"predicted_assists": "{:.2f}"}),
                            use_container_width=True
                        )
            
            with metric_tabs[2]:
                if cards_exist:
                    cards_by_pos_fig = create_position_chart(filtered_merged, 'predicted_cards')
                    st.pyplot(cards_by_pos_fig)
                    
                    # Show top players by position
                    st.subheader("Most Likely to Receive Cards by Position")
                    for pos in positions_list:
                        pos_players = filtered_merged[filtered_merged["position"] == pos]
                        top_pos_players = pos_players.nlargest(5, "predicted_cards")[["name", "team", "predicted_cards"]]
                        
                        if not top_pos_players.empty:
                            st.write(f"**{pos}**")
                            st.dataframe(
                                top_pos_players.style.format({"predicted_cards": "{:.2f}"}),
                                use_container_width=True
                            )
                else:
                    st.write("Cards prediction data not available")
            
            with metric_tabs[3]:
                if clean_sheets_exist:
                    cs_by_pos_fig = create_position_chart(filtered_merged, 'predicted_clean_sheets')
                    st.pyplot(cs_by_pos_fig)
                    
                    # Show top players by position
                    st.subheader("Highest Clean Sheet Probability by Position")
                    # Filter for only GKP and DEF
                    def_gkp_positions = [pos for pos in positions_list if pos in ["GKP", "DEF"]]
                    
                    for pos in def_gkp_positions:
                        pos_players = filtered_merged[filtered_merged["position"] == pos]
                        top_pos_players = pos_players.nlargest(5, "predicted_clean_sheets")[["name", "team", "predicted_clean_sheets"]]
                        
                        if not top_pos_players.empty:
                            st.write(f"**{pos}**")
                            st.dataframe(
                                top_pos_players.style.format({"predicted_clean_sheets": "{:.2f}"}),
                                use_container_width=True
                            )
                else:
                    st.write("Clean sheets prediction data not available")
        else:
            st.warning("Position data not available for analysis")
    
    # ---- COMBINED PREDICTION ANALYSIS ----
    st.markdown('<div class="section-header"><h2>Combined Predictions Analysis</h2></div>', unsafe_allow_html=True)
    
    # Create tabs for different analysis views
    analysis_tabs = st.tabs(["Goals vs Assists", "Team Analysis", "Correlation Analysis"])
    
    with analysis_tabs[0]:
        # ---- PLOT: PREDICTED GOALS VS ASSISTS ----
        st.subheader("📊 Predicted Goals vs Assists")
        
        # Set figure size
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create scatter plot
        if "position" in filtered_merged.columns:
            # Color by position
            position_colors = {
                'GKP': '#ebff00',
                'DEF': '#00ff85',
                'MID': '#04f5ff',
                'FWD': '#ff3977'
            }
            
            # Map positions to colors
            scatter_colors = [position_colors.get(pos, '#38003c') for pos in filtered_merged['position']]
            
            scatter = ax.scatter(
                filtered_merged["predicted_goals"],
                filtered_merged["predicted_assists"],
                c=scatter_colors,
                s=filtered_merged["total_contribution"]*20,
                alpha=0.7,
                edgecolor='black'
            )
            
            # Create a legend for positions
            position_handles = [plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=color, markersize=10, label=pos) 
                            for pos, color in position_colors.items() if pos in filtered_merged['position'].unique()]
            ax.legend(handles=position_handles, title="Position", loc="upper left")
        else:
            # Use team as color if position not available
            scatter = ax.scatter(
                filtered_merged["predicted_goals"],
                filtered_merged["predicted_assists"],
                c=pd.factorize(filtered_merged["team"])[0],
                s=filtered_merged["total_contribution"]*20,
                alpha=0.7,
                edgecolor='black'
            )
            
            # Create a legend for teams
            teams_legend = ax.legend(*scatter.legend_elements(),
                                title="Teams", loc="upper left")
        
        # Add player names to top contributors
        top_contributors = filtered_merged.nlargest(7, "total_contribution")
        for _, player in top_contributors.iterrows():
            ax.annotate(
                player["name"].split()[-1],  # Use last name for clarity
                (player["predicted_goals"], player["predicted_assists"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold"
            )
        
        # Labels & Title
        ax.set_xlabel("Predicted Goals", fontsize=12)
        ax.set_ylabel("Predicted Assists", fontsize=12)
        ax.set_title("Predicted Goals vs Assists", fontsize=14, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.6)
        
        # Add Premier League styling
        fig.set_facecolor('#f5f5f5')
        ax.set_facecolor('#f5f5f5')
        
        # Show plot
        st.pyplot(fig)
    
    with analysis_tabs[1]:
        # ---- TEAM ANALYSIS ----
        st.subheader("📊 Team Performance Analysis")
        
        # Calculate team averages
        team_stats = filtered_merged.groupby("team").agg({
            "predicted_goals": "mean",
            "predicted_assists": "mean",
            "predicted_cards": "mean" if cards_exist else "count",
            "predicted_clean_sheets": "mean" if clean_sheets_exist else "count",
            "name": "count"
        }).reset_index()
        
        team_stats.rename(columns={"name": "player_count"}, inplace=True)
        team_stats["team"] = team_stats["team"].astype(str)  
        
        # Create team comparison chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Sort by goals + assists
        team_stats["offensive_output"] = team_stats["predicted_goals"] + team_stats["predicted_assists"]
        team_stats = team_stats.sort_values("offensive_output", ascending=False)
        
        # Plot bars
        bars = ax.bar(
            team_stats["team"],
            team_stats["offensive_output"],
            color="#38003c"
        )
        
        # Add values on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.02,
                f'{height:.2f}',
                ha='center',
                fontsize=9
            )
        
        # Customize chart
        ax.set_title("Average Offensive Output by Team (Goals + Assists)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Team", fontsize=12)
        ax.set_ylabel("Average Goals + Assists", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add Premier League styling
        fig.set_facecolor('#f5f5f5')
        ax.set_facecolor('#f5f5f5')
        fig.tight_layout()
        
        # Show plot
        st.pyplot(fig)
        
        # Show team stats table
        st.subheader("Team Stats Summary")
        st.dataframe(
            team_stats.style.format({
                "predicted_goals": "{:.2f}",
                "predicted_assists": "{:.2f}",
                "predicted_cards": "{:.2f}",
                "predicted_clean_sheets": "{:.2f}",
                "offensive_output": "{:.2f}"
            }),
            use_container_width=True
        )
    
    with analysis_tabs[2]:
        # ---- CORRELATION HEATMAP ----
        corr_fig = create_correlation_heatmap(filtered_merged)
        st.pyplot(corr_fig)
        
        st.markdown("""
        <div class="highlight">
        <p><strong>Understanding the Correlation Heatmap:</strong></p>
        <ul>
          <li>Positive values (green) indicate metrics that tend to increase together</li>
          <li>Negative values (red) indicate inverse relationships</li>
          <li>Values closer to 0 (white) indicate weaker correlations</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # ---- PLAYER RANKINGS ----
    st.markdown('<div class="section-header"><h2>Overall Player Rankings</h2></div>', unsafe_allow_html=True)
    
    # Explain the combined score
    st.markdown("""
    <div class="highlight">
    <p><strong>Combined Player Score Formula:</strong></p>
    <ul>
      <li>Goals: 4 points</li>
      <li>Assists: 3 points</li>
      <li>Cards: -2 points</li>
      <li>Clean Sheets (DEF/GKP only): 4 points</li>
    </ul>
    <p class="small-text">The combined score provides a single metric that balances all predicted stats according to their FPL value.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display top players by combined score
    top_players = filtered_merged.sort_values(by="total_contribution", ascending=False)
    
    # Create a more detailed view with all predictions
    display_cols = ["name", "team", "total_contribution", "predicted_goals", "predicted_assists", "predicted_cards"]
    if clean_sheets_exist:
        display_cols.append("predicted_clean_sheets")
    if "position" in filtered_merged.columns:
        display_cols.insert(2, "position")
    
    st.dataframe(
        top_players[display_cols].head(20).style.format({
            "predicted_goals": "{:.2f}",
            "predicted_assists": "{:.2f}",
            "predicted_cards": "{:.2f}",
            "predicted_clean_sheets": "{:.2f}",
            "total_contribution": "{:.2f}"
        }),
        use_container_width=True
    )
    
    # ---- FULL DATA EXPANDERS ----
    st.markdown('<div class="section-header"><h2>Complete Datasets</h2></div>', unsafe_allow_html=True)
    
    with st.expander("Show All Predictions"):
        st.dataframe(
            filtered_merged[display_cols].style.format({
                "predicted_goals": "{:.2f}",
                "predicted_assists": "{:.2f}",
                "predicted_cards": "{:.2f}",
                "predicted_clean_sheets": "{:.2f}",
                "total_contribution": "{:.2f}"
            }),
            use_container_width=True
        )
    
    # ---- FOOTER ----
    st.markdown("""
    <div class="small-text" style="text-align: center; margin-top: 40px;">
    Fantasy Premier League AI Predictions Dashboard | Created using Streamlit and machine learning models | Data refreshed for GW25
    </div>
    """, unsafe_allow_html=True)

