import pandas as pd
from pathlib import Path

from nwp import get_nwp_data


# ============================================================
# SAFE CSV READER
# ============================================================

def read_csv_safe(path):

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(path)

    if df.empty:

        raise ValueError(
            f"Dataset is empty: {path.name}"
        )

    return df


# ============================================================
# LOAD ALL DATA
# ============================================================

def load_all_data(data_dir):

    data_dir = Path(data_dir)

    return {

        "observational":
            read_csv_safe(
                data_dir /
                "observational_weather_data.csv"
            ),

        "radar":
            read_csv_safe(
                data_dir /
                "radar_data.csv"
            ),

        "satellite":
            read_csv_safe(
                data_dir /
                "satelite.csv"
            ),

        "river":
            read_csv_safe(
                data_dir /
                "river_water_level.csv"
            ),

        "ice":
            read_csv_safe(
                data_dir /
                "ice_glacier_data.csv"
            ),

        "terrain":
            read_csv_safe(
                data_dir /
                "terrain_data.csv"
            )
    }


# ============================================================
# GET LATEST ROW
# ============================================================

def latest_row(df):

    df = df.copy()

    if "timestamp" in df.columns:

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce"
        )

        df = df.sort_values(
            "timestamp"
        )

    return df.iloc[-1]


# ============================================================
# LOCATION SCOPING
# ============================================================

def select_location_data(df, city, location):
    """Return the rows relevant to a selected city.

    City-aware feeds should include a `city` (or `location`) column.  Point
    feeds such as river gauges and terrain grids may instead include latitude
    and longitude; for those, the nearest available point is selected.  A
    legacy feed with neither is left unchanged, which makes the limitation
    visible in the API rather than silently inventing city data.
    """

    scoped = df.copy()
    normalized_city = city.casefold()

    for column in ("city", "location", "location_name"):
        if column in scoped.columns:
            matches = scoped[column].astype(str).str.strip().str.casefold() == normalized_city
            if matches.any():
                return scoped.loc[matches], "city"

    if {"latitude", "longitude"}.issubset(scoped.columns):
        latitude = pd.to_numeric(scoped["latitude"], errors="coerce")
        longitude = pd.to_numeric(scoped["longitude"], errors="coerce")
        distance = (latitude - location["latitude"]) ** 2 + (longitude - location["longitude"]) ** 2
        if distance.notna().any():
            nearest_index = distance.idxmin()
            # Keep the full time series for the selected gauge/location.
            same_point = (latitude == latitude.loc[nearest_index]) & (longitude == longitude.loc[nearest_index])
            return scoped.loc[same_point], "nearest_coordinate"

    return scoped, "unscoped_legacy_feed"


# ============================================================
# NUMERIC VALUE HELPER
# ============================================================

def numeric_value(
    row,
    column,
    default=0.0
):

    if column not in row.index:

        return default

    value = pd.to_numeric(
        row[column],
        errors="coerce"
    )

    if pd.isna(value):

        return default

    return float(value)


# ============================================================
# OBSERVATIONAL WEATHER
# ============================================================

def get_observational_values(df):

    row = latest_row(df)

    return {

        "temperature_c":
            numeric_value(
                row,
                "temperature_c"
            ),

        "humidity_pct":
            numeric_value(
                row,
                "humidity_pct"
            ),

        "pressure_hpa":
            numeric_value(
                row,
                "pressure_hpa"
            ),

        "wind_speed_kmh":
            numeric_value(
                row,
                "wind_speed_kmh"
            ),

        "rainfall_1h_mm":
            numeric_value(
                row,
                "rainfall_1h_mm"
            ),

        "cloud_cover_pct":
            numeric_value(
                row,
                "cloud_cover_pct"
            )
    }


# ============================================================
# RADAR
# ============================================================

def get_radar_values(df):

    row = latest_row(df)

    return {

        "radar_reflectivity_dbz":
            numeric_value(
                row,
                "radar_reflectivity_dbz"
            ),

        "storm_cell_density":
            numeric_value(
                row,
                "storm_cell_density"
            )
    }


# ============================================================
# SATELLITE
# ============================================================

def get_satellite_values(df):

    row = latest_row(df)

    return {

        "satellite_rain_index":
            numeric_value(
                row,
                "satellite_rain_index"
            ),

        "cloud_top_temp_c":
            numeric_value(
                row,
                "cloud_top_temp_c"
            )
    }


# ============================================================
# RIVER
# ============================================================

def get_river_values(df):

    row = latest_row(df)

    return {

        "river_name":
            str(
                row.get(
                    "river_name",
                    "Unknown"
                )
            ),

        "station_id":
            str(
                row.get(
                    "station_id",
                    "Unknown"
                )
            ),

        "water_level_m":
            numeric_value(
                row,
                "water_level_m"
            ),

        "warning_level_m":
            numeric_value(
                row,
                "warning_level_m"
            ),

        "danger_level_m":
            numeric_value(
                row,
                "danger_level_m"
            ),

        "flow_rate_cumecs":
            numeric_value(
                row,
                "flow_rate_cumecs"
            )
    }


# ============================================================
# ICE / GLACIER
# ============================================================

def get_ice_values(df):

    row = latest_row(df)

    return {

        "region":
            str(
                row.get(
                    "region",
                    "Unknown"
                )
            ),

        "ice_melt_index":
            numeric_value(
                row,
                "ice_melt_index"
            ),

        "melting_rate_mm_day":
            numeric_value(
                row,
                "melting_rate_mm_day"
            ),

        "runoff_contribution_mm":
            numeric_value(
                row,
                "runoff_contribution_mm"
            )
    }


# ============================================================
# TERRAIN
# ============================================================

def get_terrain_values(df):

    row = latest_row(df)

    return {

        "elevation_m":
            numeric_value(
                row,
                "elevation_m"
            ),

        "slope_deg":
            numeric_value(
                row,
                "slope_deg"
            ),

        "drainage_density":
            numeric_value(
                row,
                "drainage_density"
            ),

        "inundation_susceptibility":
            numeric_value(
                row,
                "inundation_susceptibility"
            )
    }


# ============================================================
# NWP DATA
# ============================================================

def get_nwp_values(
    latitude=16.3067,
    longitude=80.4365,
    fallback=None
):
    try:
        data = get_nwp_data(
            latitude,
            longitude
        )

        hourly = data["hourly"]

        return {

            "nwp_rainfall_mm":
                float(
                    hourly["precipitation"][0]
                ),

            "nwp_temperature_c":
                float(
                    hourly["temperature_2m"][0]
                ),

            "nwp_humidity_pct":
                float(
                    hourly[
                        "relative_humidity_2m"
                    ][0]
                ),

            "nwp_pressure_hpa":
                float(
                    hourly[
                        "surface_pressure"
                    ][0]
                ),

            "nwp_wind_speed_kmh":
                float(
                    hourly[
                        "wind_speed_10m"
                    ][0]
                ),

            "source": "open_meteo"
        }

    except Exception:
        # Keep the dashboard usable when the forecast provider is unavailable.
        # These are explicitly labelled fallback values, not a city forecast.
        fallback = fallback or {}
        return {
            "nwp_rainfall_mm": fallback.get("rainfall_1h_mm", 0.0),
            "nwp_temperature_c": fallback.get("temperature_c", 0.0),
            "nwp_humidity_pct": fallback.get("humidity_pct", 0.0),
            "nwp_pressure_hpa": fallback.get("pressure_hpa", 0.0),
            "nwp_wind_speed_kmh": fallback.get("wind_speed_kmh", 0.0),
            "source": "observational_fallback"
        }


# ============================================================
# BUILD ML INPUT
# ============================================================

def build_ml_input(
    observational,
    radar,
    satellite
):

    return {

        "temperature_c":
            observational[
                "temperature_c"
            ],

        "humidity_pct":
            observational[
                "humidity_pct"
            ],

        "pressure_hpa":
            observational[
                "pressure_hpa"
            ],

        "wind_speed_kmh":
            observational[
                "wind_speed_kmh"
            ],

        "rainfall_1h_mm":
            observational[
                "rainfall_1h_mm"
            ],

        "cloud_cover_pct":
            observational[
                "cloud_cover_pct"
            ],

        "radar_reflectivity_dbz":
            radar[
                "radar_reflectivity_dbz"
            ],

        "satellite_rain_index":
            satellite[
                "satellite_rain_index"
            ],

        "storm_cell_density":
            radar[
                "storm_cell_density"
            ],

        "cloud_top_temp_c":
            satellite[
                "cloud_top_temp_c"
            ]
    }


# ============================================================
# RAINFALL DATA FUSION SCORE
# ============================================================

def calculate_fusion_score(
    observational,
    radar,
    satellite,
    nwp
):

    rainfall_score = min(
        max(
            observational[
                "rainfall_1h_mm"
            ] / 50,
            0
        ),
        1
    )


    radar_score = min(
        max(
            radar[
                "radar_reflectivity_dbz"
            ] / 60,
            0
        ),
        1
    )


    satellite_score = min(
        max(
            satellite[
                "satellite_rain_index"
            ] / 100,
            0
        ),
        1
    )


    storm_score = min(
        max(
            radar[
                "storm_cell_density"
            ],
            0
        ),
        1
    )


    nwp_rain_score = min(
        max(
            nwp[
                "nwp_rainfall_mm"
            ] / 50,
            0
        ),
        1
    )


    score = (

        rainfall_score * 0.25

        + radar_score * 0.25

        + satellite_score * 0.20

        + storm_score * 0.10

        + nwp_rain_score * 0.20

    )


    return round(
        score * 100,
        2
    )


# ============================================================
# RIVER RISK
# ============================================================

def calculate_river_risk(river):

    water = river[
        "water_level_m"
    ]

    warning = river[
        "warning_level_m"
    ]

    danger = river[
        "danger_level_m"
    ]


    if danger <= 0:

        return 0


    risk = water / danger


    return round(
        min(
            max(
                risk,
                0
            ),
            1
        ) * 100,
        2
    )


# ============================================================
# INUNDATION RISK
# ============================================================

def calculate_inundation_risk(
    rainfall_score,
    river_risk,
    terrain,
    ice
):

    terrain_score = min(
        max(
            terrain[
                "inundation_susceptibility"
            ],
            0
        ),
        1
    ) * 100


    ice_score = min(
        max(
            ice[
                "ice_melt_index"
            ],
            0
        ),
        1
    ) * 100


    inundation_score = (

        rainfall_score * 0.40

        + river_risk * 0.35

        + terrain_score * 0.20

        + ice_score * 0.05

    )


    return round(
        inundation_score,
        2
    )


# ============================================================
# RISK LEVEL
# ============================================================

def risk_level(score):

    if score < 25:

        return "LOW"

    elif score < 50:

        return "MEDIUM"

    elif score < 75:

        return "HIGH"

    return "CRITICAL"


# ============================================================
# WARNING MESSAGE
# ============================================================

def create_warning(level):

    if level == "LOW":

        return (
            "Current conditions show low flood risk. "
            "Continue monitoring weather conditions."
        )


    if level == "MEDIUM":

        return (
            "Moderate flood risk detected. "
            "Monitor rainfall and river conditions closely."
        )


    if level == "HIGH":

        return (
            "High inundation risk detected. "
            "Heavy rainfall and rising river levels "
            "may cause localized flooding."
        )


    return (
        "Critical inundation risk detected. "
        "Follow official emergency guidance and "
        "avoid vulnerable flood-prone areas."
    )


# ============================================================
# BUILD DASHBOARD
# ============================================================

def build_dashboard(
    data,
    model,
    city="Guntur",
    location=None
):

    if location is None:
        location = {"latitude": 16.3067, "longitude": 80.4365}

    scoped_data = {}
    source_scope = {}
    for source, frame in data.items():
        scoped_data[source], source_scope[source] = select_location_data(
            frame, city, location
        )

    # --------------------------------------------------------
    # OBSERVATIONAL
    # --------------------------------------------------------

    observational = get_observational_values(
        scoped_data["observational"]
    )


    # --------------------------------------------------------
    # RADAR
    # --------------------------------------------------------

    radar = get_radar_values(
        scoped_data["radar"]
    )


    # --------------------------------------------------------
    # SATELLITE
    # --------------------------------------------------------

    satellite = get_satellite_values(
        scoped_data["satellite"]
    )


    # --------------------------------------------------------
    # RIVER
    # --------------------------------------------------------

    river = get_river_values(
        scoped_data["river"]
    )


    # --------------------------------------------------------
    # ICE / GLACIER
    # --------------------------------------------------------

    ice = get_ice_values(
        scoped_data["ice"]
    )


    # --------------------------------------------------------
    # TERRAIN
    # --------------------------------------------------------

    terrain = get_terrain_values(
        scoped_data["terrain"]
    )


    # --------------------------------------------------------
    # NWP
    # --------------------------------------------------------

    nwp = get_nwp_values(
        location["latitude"],
        location["longitude"],
        fallback=observational
    )


    # --------------------------------------------------------
    # ML INPUT
    # --------------------------------------------------------

    ml_input = build_ml_input(
        observational,
        radar,
        satellite
    )


    # --------------------------------------------------------
    # ML PREDICTION
    # --------------------------------------------------------

    ml_result = model.predict(
        ml_input
    )


    # --------------------------------------------------------
    # RAINFALL FUSION
    # --------------------------------------------------------

    rainfall_score = calculate_fusion_score(
        observational,
        radar,
        satellite,
        nwp
    )


    # --------------------------------------------------------
    # RIVER RISK
    # --------------------------------------------------------

    river_risk = calculate_river_risk(
        river
    )


    # --------------------------------------------------------
    # INUNDATION RISK
    # --------------------------------------------------------

    inundation_score = calculate_inundation_risk(
        rainfall_score,
        river_risk,
        terrain,
        ice
    )


    # --------------------------------------------------------
    # FINAL RISK LEVEL
    # --------------------------------------------------------

    inundation_level = risk_level(
        inundation_score
    )


    # --------------------------------------------------------
    # FINAL DASHBOARD RESULT
    # --------------------------------------------------------

    return {

        "location": {
            "city": city,
            "latitude": location["latitude"],
            "longitude": location["longitude"]
        },

        "data_scope": source_scope,

        "risk_level":
            inundation_level,


        "risk_score":
            inundation_score,


        "rainfall_score":
            rainfall_score,


        "river_risk":
            river_risk,


        "ml_prediction":
            ml_result[
                "risk_label"
            ],


        "ml_confidence":
            ml_result[
                "confidence"
            ],


        "ml_probabilities":
            ml_result[
                "probabilities"
            ],


        "observational":
            observational,


        "radar":
            radar,


        "satellite":
            satellite,


        "nwp":
            nwp,


        "river":
            river,


        "ice":
            ice,


        "terrain":
            terrain,


        "warning":
            create_warning(
                inundation_level
            )
    }


# ============================================================
# COMPATIBILITY FUNCTION
# ============================================================

def predict_risk(
    data,
    model,
    payload=None
):

    return build_dashboard(
        data,
        model
    )
