import streamlit as st
import pandas as pd
import math
from datetime import timedelta, date
from io import BytesIO
import pulp
import os
import pickle
from supabase import create_client

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AYD Shipment Scheduler",
    page_icon="🚚",
    layout="wide"
)


# ============================================================
# GLOBAL SETTINGS
# ============================================================

DEFAULT_DAILY_CAPACITY = 18
STATE_FILE = "ayd_scheduler_saved_state.pkl"

TRANSIT_TIME = {
    "Road": 1,
    "Rail": 2
}

PRIORITY_WEIGHT = {
    "Urgent": 3,
    "High": 3,
    "Normal": 2,
    "Medium": 2,
    "Low": 1,
    "Acil": 3,
    "Yüksek": 3,
    "Yuksek": 3,
    "Orta": 2,
    "Düşük": 1,
    "Dusuk": 1
}

# Python weekday:
# Monday=0, Tuesday=1, Wednesday=2, Thursday=3, Friday=4, Saturday=5, Sunday=6
DISPATCH_AVAILABLE_WEEKDAYS = [0, 1, 2, 3, 4, 5]
RAIL_AVAILABLE_WEEKDAYS = [0, 2, 5]
DAY_NAMES_TR = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday"
}
MONTH_NAMES_TR = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December"
}

# ============================================================
# SESSION STATE
# ============================================================

if "input_df" not in st.session_state:
    st.session_state.input_df = pd.DataFrame(columns=[
        "Shipment_ID",
        "Destination",
        "Shipment_Type",
        "Free_Entry_Date",
        "Cutoff_Date",
        "Earliest_Dispatch_Date",
        "Latest_Dispatch_Date",
        "Vehicle_Count",
        "Priority",
        "Preferred_Mode",
        "Source"
    ])

if "output_df" not in st.session_state:
    st.session_state.output_df = pd.DataFrame()

if "daily_summary" not in st.session_state:
    st.session_state.daily_summary = pd.DataFrame()

if "objective_summary" not in st.session_state:
    st.session_state.objective_summary = pd.DataFrame()

if "validation_df" not in st.session_state:
    st.session_state.validation_df = pd.DataFrame()

if "options_df" not in st.session_state:
    st.session_state.options_df = pd.DataFrame()
if "last_uploaded_file_id" not in st.session_state:
    st.session_state.last_uploaded_file_id = None
if "calendar_week_start" not in st.session_state:
    today = pd.Timestamp.today().normalize()
    st.session_state.calendar_week_start = today - pd.Timedelta(days=today.weekday())
if "needs_optimization" not in st.session_state:
    st.session_state.needs_optimization = False

# ============================================================
# HELPER FUNCTIONS
# ============================================================
PERSISTED_STATE_KEYS = [
    "input_df",
    "output_df",
    "daily_summary",
    "objective_summary",
    "validation_df",
    "options_df",
    "last_uploaded_file_id",
    "calendar_week_start",
    "needs_optimization"
]


INPUT_DB_TO_APP_COLUMNS = {
    "shipment_id": "Shipment_ID",
    "destination": "Destination",
    "shipment_type": "Shipment_Type",
    "free_entry_date": "Free_Entry_Date",
    "cutoff_date": "Cutoff_Date",
    "earliest_dispatch_date": "Earliest_Dispatch_Date",
    "latest_dispatch_date": "Latest_Dispatch_Date",
    "vehicle_count": "Vehicle_Count",
    "priority": "Priority",
    "preferred_mode": "Preferred_Mode",
    "source": "Source"
}

INPUT_APP_TO_DB_COLUMNS = {v: k for k, v in INPUT_DB_TO_APP_COLUMNS.items()}

SCHEDULE_DB_TO_APP_COLUMNS = {
    "shipment_id": "Shipment_ID",
    "destination": "Destination",
    "shipment_type": "Shipment_Type",
    "recommended_mode": "Recommended_Mode",
    "dispatch_date": "Dispatch_Date",
    "arrival_date": "Arrival_Date",
    "free_entry_date": "Free_Entry_Date",
    "cutoff_date": "Cutoff_Date",
    "earliest_dispatch_date": "Earliest_Dispatch_Date",
    "latest_dispatch_date": "Latest_Dispatch_Date",
    "slack_before_cutoff": "Slack_Before_Cutoff",
    "dispatch_deviation": "Dispatch_Deviation",
    "vehicle_count": "Vehicle_Count",
    "priority": "Priority",
    "priority_weight": "Priority_Weight",
    "risk_score": "Risk_Score",
    "preferred_mode": "Preferred_Mode",
    "status": "Status",
    "manual_override": "Manual_Override",
    "delay_reason": "Delay_Reason",
    "manual_warnings": "Manual_Warnings",
    "source": "Source"
}

SCHEDULE_APP_TO_DB_COLUMNS = {v: k for k, v in SCHEDULE_DB_TO_APP_COLUMNS.items()}

SUMMARY_DB_TO_APP_COLUMNS = {
    "dispatch_date": "Dispatch_Date",
    "daily_total_vehicles": "Daily_Total_Vehicles",
    "number_of_shipments": "Number_of_Shipments",
    "port_shipments": "Port_Shipments",
    "target_shipments": "Target_Shipments",
    "rail_shipments": "Rail_Shipments",
    "road_shipments": "Road_Shipments",
    "over_capacity": "Over_Capacity"
}

SUMMARY_APP_TO_DB_COLUMNS = {v: k for k, v in SUMMARY_DB_TO_APP_COLUMNS.items()}


def get_supabase_client():
    """
    Returns a Supabase client if secrets are configured.
    If secrets are not available, the app falls back to local pickle storage.
    """
    try:
        supabase_url = st.secrets["SUPABASE_URL"]
        supabase_key = st.secrets["SUPABASE_KEY"]
        return create_client(supabase_url, supabase_key)
    except Exception:
        return None


def _convert_value_for_db(value):
    """
    Converts pandas/numpy values into Supabase-compatible Python values.
    """
    if pd.isna(value):
        return None

    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()

    if hasattr(value, "isoformat") and value.__class__.__name__ in ["date", "datetime"]:
        return value.isoformat()

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return value


def _df_to_supabase_records(df, app_to_db_columns):
    """
    Converts an app dataframe to a list of dictionaries matching Supabase column names.
    """
    if df is None or df.empty:
        return []

    work_df = df.copy()
    records = []

    for _, row in work_df.iterrows():
        record = {}

        for app_col, db_col in app_to_db_columns.items():
            if app_col in work_df.columns:
                record[db_col] = _convert_value_for_db(row.get(app_col))

        records.append(record)

    return records


def _supabase_records_to_df(records, db_to_app_columns):
    """
    Converts Supabase records into app dataframe column names.
    """
    if not records:
        return pd.DataFrame(columns=list(db_to_app_columns.values()))

    df = pd.DataFrame(records)

    # Remove Supabase metadata columns if present.
    drop_cols = [col for col in ["id", "created_at", "updated_at"] if col in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    df = df.rename(columns=db_to_app_columns)

    return df


def _delete_all_rows(supabase, table_name):
    """
    Deletes all rows from a Supabase table.
    The id column is positive bigserial, so id >= 0 clears the table.
    """
    try:
        supabase.table(table_name).delete().gte("id", 0).execute()
    except Exception:
        # Empty tables or API differences should not break the app.
        pass


def save_app_state():
    """
    Saves current input, optimization result, and daily summary.
    On Streamlit Cloud, it saves to Supabase.
    If Supabase secrets are missing, it falls back to local pickle storage.
    """
    supabase = get_supabase_client()

    if supabase is None:
        state_to_save = {}

        for key in PERSISTED_STATE_KEYS:
            if key in st.session_state:
                state_to_save[key] = st.session_state[key]

        with open(STATE_FILE, "wb") as f:
            pickle.dump(state_to_save, f)

        return

    try:
        # INPUT SHIPMENTS
        _delete_all_rows(supabase, "input_shipments")
        input_records = _df_to_supabase_records(
            st.session_state.input_df,
            INPUT_APP_TO_DB_COLUMNS
        )
        if input_records:
            supabase.table("input_shipments").upsert(
                input_records,
                on_conflict="shipment_id"
            ).execute()

        # SCHEDULED SHIPMENTS
        _delete_all_rows(supabase, "scheduled_shipments")
        schedule_records = _df_to_supabase_records(
            st.session_state.output_df,
            SCHEDULE_APP_TO_DB_COLUMNS
        )
        if schedule_records:
            supabase.table("scheduled_shipments").upsert(
                schedule_records,
                on_conflict="shipment_id"
            ).execute()

        # DAILY SUMMARY
        _delete_all_rows(supabase, "daily_summary")
        summary_records = _df_to_supabase_records(
            st.session_state.daily_summary,
            SUMMARY_APP_TO_DB_COLUMNS
        )
        if summary_records:
            supabase.table("daily_summary").upsert(
                summary_records,
                on_conflict="dispatch_date"
            ).execute()

    except Exception as e:
        st.warning(f"Supabase save failed: {e}")


def load_app_state_once():
    """
    Loads saved input, schedule, and daily summary only once when the app starts.
    On Streamlit Cloud, it loads from Supabase.
    If Supabase secrets are missing, it falls back to local pickle storage.
    """
    if st.session_state.get("state_loaded", False):
        return

    supabase = get_supabase_client()

    if supabase is None:
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "rb") as f:
                    saved_state = pickle.load(f)

                for key, value in saved_state.items():
                    st.session_state[key] = value

            except Exception as e:
                st.warning(f"Saved state could not be loaded: {e}")

        st.session_state.state_loaded = True
        return

    try:
        input_response = supabase.table("input_shipments").select("*").execute()
        schedule_response = supabase.table("scheduled_shipments").select("*").execute()
        summary_response = supabase.table("daily_summary").select("*").execute()

        loaded_input_df = _supabase_records_to_df(
            input_response.data,
            INPUT_DB_TO_APP_COLUMNS
        )
        loaded_schedule_df = _supabase_records_to_df(
            schedule_response.data,
            SCHEDULE_DB_TO_APP_COLUMNS
        )
        loaded_summary_df = _supabase_records_to_df(
            summary_response.data,
            SUMMARY_DB_TO_APP_COLUMNS
        )

        if not loaded_input_df.empty:
            st.session_state.input_df = normalize_input_for_session(loaded_input_df)

        if not loaded_schedule_df.empty:
            for col in [
                "Dispatch_Date",
                "Arrival_Date",
                "Free_Entry_Date",
                "Cutoff_Date",
                "Earliest_Dispatch_Date",
                "Latest_Dispatch_Date"
            ]:
                if col in loaded_schedule_df.columns:
                    loaded_schedule_df[col] = pd.to_datetime(
                        loaded_schedule_df[col],
                        errors="coerce"
                    )

            st.session_state.output_df = loaded_schedule_df

        if not loaded_summary_df.empty:
            if "Dispatch_Date" in loaded_summary_df.columns:
                loaded_summary_df["Dispatch_Date"] = pd.to_datetime(
                    loaded_summary_df["Dispatch_Date"],
                    errors="coerce"
                )
            st.session_state.daily_summary = loaded_summary_df

        st.session_state.needs_optimization = False

        if not st.session_state.output_df.empty:
            st.session_state.calendar_week_start = get_week_start_from_output(
                st.session_state.output_df
            )

    except Exception as e:
        st.warning(f"Supabase load failed: {e}")

    st.session_state.state_loaded = True


def clear_saved_state():
    """
    Deletes saved app state from Supabase if configured.
    Otherwise deletes the local pickle file.
    """
    supabase = get_supabase_client()

    if supabase is None:
        if os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)
    else:
        try:
            _delete_all_rows(supabase, "daily_summary")
            _delete_all_rows(supabase, "scheduled_shipments")
            _delete_all_rows(supabase, "input_shipments")
        except Exception as e:
            st.warning(f"Supabase clear failed: {e}")

    st.session_state.input_df = pd.DataFrame(columns=[
        "Shipment_ID",
        "Destination",
        "Shipment_Type",
        "Free_Entry_Date",
        "Cutoff_Date",
        "Earliest_Dispatch_Date",
        "Latest_Dispatch_Date",
        "Vehicle_Count",
        "Priority",
        "Preferred_Mode",
        "Source"
    ])
    st.session_state.output_df = pd.DataFrame()
    st.session_state.daily_summary = pd.DataFrame()
    st.session_state.objective_summary = pd.DataFrame()
    st.session_state.validation_df = pd.DataFrame()
    st.session_state.options_df = pd.DataFrame()
    st.session_state.last_uploaded_file_id = None
    st.session_state.needs_optimization = False


def normalize_date(value):
    if pd.isna(value) or value == "":
        return pd.NaT
    return pd.to_datetime(value, errors="coerce")


def clean_mode(value):
    if pd.isna(value) or str(value).strip() == "":
        return "Auto"

    value = str(value).strip()

    mode_map = {
        "auto": "Auto",
        "AUTO": "Auto",
        "Auto": "Auto",
        "road": "Road",
        "ROAD": "Road",
        "Road": "Road",
        "karayolu": "Road",
        "Karayolu": "Road",
        "KARAYOLU": "Road",
        "rail": "Rail",
        "RAIL": "Rail",
        "Rail": "Rail",
        "tren": "Rail",
        "Tren": "Rail",
        "TREN": "Rail"
    }

    return mode_map.get(value, "Auto")


def get_allowed_modes(preferred_mode):
    preferred_mode = clean_mode(preferred_mode)

    if preferred_mode == "Road":
        return ["Road"]
    elif preferred_mode == "Rail":
        return ["Rail"]
    else:
        return ["Road", "Rail"]


def calculate_arrival(dispatch_date, mode):
    return dispatch_date + timedelta(days=TRANSIT_TIME[mode])

DATE_COLUMNS = [
    "Free_Entry_Date",
    "Cutoff_Date",
    "Earliest_Dispatch_Date",
    "Latest_Dispatch_Date"
]


def normalize_input_for_session(df):
    """
    Excel + manual input karıştığında tarih tiplerini standartlaştırır.
    Manuel eklenen datetime.date değerlerini pandas Timestamp formatına çevirir.
    """
    df = df.copy()

    for col in DATE_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NaT
        df[col] = pd.to_datetime(df[col], errors="coerce")

    if "Vehicle_Count" in df.columns:
        df["Vehicle_Count"] = pd.to_numeric(df["Vehicle_Count"], errors="coerce")

    if "Source" not in df.columns:
        df["Source"] = "Unknown"
    df["Source"] = df["Source"].fillna("Unknown").astype(str)

    return df


def make_display_df(df):
    """
    st.dataframe için güvenli görünüm üretir.
    Tarihleri stringe çevirir, böylece pyarrow date type hatası oluşmaz.
    """
    display_df = df.copy()

    for col in DATE_COLUMNS:
        if col in display_df.columns:
            display_df[col] = pd.to_datetime(
                display_df[col],
                errors="coerce"
            ).dt.strftime("%Y-%m-%d")

            display_df[col] = display_df[col].fillna("")

    return display_df

def calculate_daily_summary(schedule_df, daily_capacity):
    if schedule_df.empty:
        return pd.DataFrame(columns=[
            "Dispatch_Date",
            "Daily_Total_Vehicles",
            "Number_of_Shipments",
            "Port_Shipments",
            "Target_Shipments",
            "Rail_Shipments",
            "Road_Shipments",
            "Over_Capacity"
        ])

    active_df = schedule_df[schedule_df["Status"] != "Cancelled"].copy()

    if active_df.empty:
        return pd.DataFrame(columns=[
            "Dispatch_Date",
            "Daily_Total_Vehicles",
            "Number_of_Shipments",
            "Port_Shipments",
            "Target_Shipments",
            "Rail_Shipments",
            "Road_Shipments",
            "Over_Capacity"
        ])

    daily_summary = active_df.groupby("Dispatch_Date", as_index=False).agg(
        Daily_Total_Vehicles=("Vehicle_Count", "sum"),
        Number_of_Shipments=("Shipment_ID", "count"),
        Port_Shipments=("Shipment_Type", lambda x: (x == "PORT").sum()),
        Target_Shipments=("Shipment_Type", lambda x: (x == "TARGET").sum()),
        Rail_Shipments=("Recommended_Mode", lambda x: (x == "Rail").sum()),
        Road_Shipments=("Recommended_Mode", lambda x: (x == "Road").sum())
    )

    daily_summary["Over_Capacity"] = daily_summary["Daily_Total_Vehicles"].apply(
        lambda value: max(0, value - daily_capacity)
    )

    daily_summary = daily_summary.sort_values("Dispatch_Date").reset_index(drop=True)

    return daily_summary


def get_upcoming_over_capacity_days(daily_summary):
    """
    Returns only today and future over-capacity days.
    Past over-capacity days are excluded from the sidebar count and table.
    """
    if daily_summary.empty or "Over_Capacity" not in daily_summary.columns:
        return pd.DataFrame()

    upcoming_df = daily_summary.copy()
    upcoming_df["Dispatch_Date"] = pd.to_datetime(
        upcoming_df["Dispatch_Date"],
        errors="coerce"
    ).dt.normalize()

    today = pd.Timestamp.today().normalize()

    upcoming_df = upcoming_df[
        (upcoming_df["Dispatch_Date"] >= today)
        & (upcoming_df["Over_Capacity"] > 0)
    ].copy()

    upcoming_df = upcoming_df.sort_values("Dispatch_Date").reset_index(drop=True)

    return upcoming_df


def clear_results_after_input_change():
    """
    Input değiştiğinde eski takvimi silme.
    Sadece kullanıcıya optimizasyonun güncel olmadığını işaretle.
    Böylece over-capacity günleri ve mevcut calendar görünmeye devam eder.
    """
    st.session_state.needs_optimization = True


def mark_optimization_complete():
    st.session_state.needs_optimization = False

def validate_input_dataframe(raw_df):
    df = raw_df.copy()

    df.columns = df.columns.str.strip()

    required_columns = [
        "Shipment_ID",
        "Destination",
        "Shipment_Type",
        "Vehicle_Count",
        "Priority"
    ]

    missing_cols = [col for col in required_columns if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    optional_columns = [
        "Free_Entry_Date",
        "Cutoff_Date",
        "Earliest_Dispatch_Date",
        "Latest_Dispatch_Date",
        "Preferred_Mode"
    ]

    for col in optional_columns:
        if col not in df.columns:
            df[col] = pd.NA

    df = df.dropna(how="all")
    df = df[df["Shipment_ID"].notna()].copy()

    df["Shipment_ID"] = df["Shipment_ID"].astype(str).str.strip()
    df["Destination"] = df["Destination"].astype(str).str.strip()
    df["Shipment_Type"] = df["Shipment_Type"].astype(str).str.strip().str.upper()

    valid_types = ["PORT", "TARGET"]
    invalid_type_rows = df[~df["Shipment_Type"].isin(valid_types)]

    if not invalid_type_rows.empty:
        raise ValueError("Shipment_Type must be either PORT or TARGET.")

    df["Vehicle_Count"] = pd.to_numeric(df["Vehicle_Count"], errors="coerce")

    if df["Vehicle_Count"].isna().any():
        bad_rows = df[df["Vehicle_Count"].isna()][["Shipment_ID", "Vehicle_Count"]]
        raise ValueError(f"Invalid Vehicle_Count values:\n{bad_rows}")

    # Araç sayısı operasyonel olarak tam sayı olmalı.
    # Excel'den küsuratlı gelirse yukarı yuvarlıyoruz.
    df["Vehicle_Count"] = df["Vehicle_Count"].apply(lambda x: int(math.ceil(float(x))))

    if (df["Vehicle_Count"] <= 0).any():
        bad_rows = df[df["Vehicle_Count"] <= 0][["Shipment_ID", "Vehicle_Count"]]
        raise ValueError(f"Vehicle_Count must be greater than 0:\n{bad_rows}")

    df["Priority"] = df["Priority"].astype(str).str.strip()
    df["Priority"] = df["Priority"].replace({"nan": "Normal", "": "Normal"})
    df["Priority_Weight"] = df["Priority"].map(PRIORITY_WEIGHT).fillna(2)

    df["Preferred_Mode"] = df["Preferred_Mode"].apply(clean_mode)

    date_columns = [
        "Free_Entry_Date",
        "Cutoff_Date",
        "Earliest_Dispatch_Date",
        "Latest_Dispatch_Date"
    ]

    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    port_rows = df[df["Shipment_Type"] == "PORT"]

    if port_rows[["Free_Entry_Date", "Cutoff_Date"]].isna().any(axis=1).any():
        raise ValueError("PORT shipments must have Free_Entry_Date and Cutoff_Date.")

    invalid_port_window = port_rows[
        port_rows["Cutoff_Date"] < port_rows["Free_Entry_Date"]
    ]

    if not invalid_port_window.empty:
        raise ValueError("For PORT shipments, Cutoff_Date cannot be earlier than Free_Entry_Date.")

    target_rows = df[df["Shipment_Type"] == "TARGET"]

    if target_rows[["Earliest_Dispatch_Date", "Latest_Dispatch_Date"]].isna().any(axis=1).any():
        raise ValueError("TARGET shipments must have Earliest_Dispatch_Date and Latest_Dispatch_Date.")

    invalid_target_window = target_rows[
        target_rows["Latest_Dispatch_Date"] < target_rows["Earliest_Dispatch_Date"]
    ]

    if not invalid_target_window.empty:
        raise ValueError("For TARGET shipments, Latest_Dispatch_Date cannot be earlier than Earliest_Dispatch_Date.")

    return df


def run_optimization(input_df, daily_capacity=18, absolute_daily_capacity=None):
    df = validate_input_dataframe(input_df)

    options = []

    for _, row in df.iterrows():
        shipment_id = row["Shipment_ID"]
        shipment_type = row["Shipment_Type"]
        preferred_mode = row["Preferred_Mode"]
        allowed_modes = get_allowed_modes(preferred_mode)

        if shipment_type == "PORT":
            free_entry = row["Free_Entry_Date"]
            cutoff = row["Cutoff_Date"]

            start_date = free_entry - timedelta(days=max(TRANSIT_TIME.values()) + 2)
            end_date = cutoff

            candidate_dates = pd.date_range(start=start_date, end=end_date, freq="D")

            for dispatch_date in candidate_dates:

                # Hard rule: no Sunday dispatch
                if dispatch_date.weekday() not in DISPATCH_AVAILABLE_WEEKDAYS:
                    continue

                for mode in allowed_modes:

                    # Model-level rail availability rule
                    if mode == "Rail" and dispatch_date.weekday() not in RAIL_AVAILABLE_WEEKDAYS:
                        continue

                    arrival_date = calculate_arrival(dispatch_date, mode)

                    if arrival_date < free_entry:
                        continue

                    if arrival_date > cutoff:
                        continue

                    slack = (cutoff - arrival_date).days

                    if slack >= 2:
                        risk_score = 0
                    elif slack == 1:
                        risk_score = 1
                    else:
                        risk_score = 3

                    options.append({
                        "Shipment_ID": shipment_id,
                        "Shipment_Type": shipment_type,
                        "Dispatch_Date": dispatch_date,
                        "Mode": mode,
                        "Arrival_Date": arrival_date,
                        "Risk_Score": risk_score,
                        "Slack_Before_Cutoff": slack,
                        "Dispatch_Deviation": 0
                    })

        elif shipment_type == "TARGET":
            earliest_dispatch = row["Earliest_Dispatch_Date"]
            latest_dispatch = row["Latest_Dispatch_Date"]

            candidate_dates = pd.date_range(start=earliest_dispatch, end=latest_dispatch, freq="D")

            planned_date = earliest_dispatch + (latest_dispatch - earliest_dispatch) / 2

            for dispatch_date in candidate_dates:

                # Hard rule: no Sunday dispatch
                if dispatch_date.weekday() not in DISPATCH_AVAILABLE_WEEKDAYS:
                    continue

                for mode in allowed_modes:

                    # Model-level rail availability rule
                    if mode == "Rail" and dispatch_date.weekday() not in RAIL_AVAILABLE_WEEKDAYS:
                        continue

                    arrival_date = calculate_arrival(dispatch_date, mode)
                    dispatch_deviation = abs((dispatch_date - planned_date).days)

                    options.append({
                        "Shipment_ID": shipment_id,
                        "Shipment_Type": shipment_type,
                        "Dispatch_Date": dispatch_date,
                        "Mode": mode,
                        "Arrival_Date": arrival_date,
                        "Risk_Score": 0,
                        "Slack_Before_Cutoff": None,
                        "Dispatch_Deviation": dispatch_deviation
                    })

    options_df = pd.DataFrame(options)

    if options_df.empty:
        raise ValueError("No feasible shipment options found. Please check dates, preferred modes, and Sunday rule.")

    input_shipments = set(df["Shipment_ID"])
    option_shipments = set(options_df["Shipment_ID"])
    no_option_shipments = input_shipments - option_shipments

    if no_option_shipments:
        raise ValueError(
            "Some shipments have no feasible assignment: "
            + ", ".join(sorted(no_option_shipments))
            + ". Please check time windows, Preferred_Mode, rail days, and Sunday rule."
        )

    model = pulp.LpProblem("Shipment_Scheduling_Model", pulp.LpMinimize)

    x = {}

    for _, opt in options_df.iterrows():
        key = (
            opt["Shipment_ID"],
            opt["Dispatch_Date"].strftime("%Y-%m-%d"),
            opt["Mode"]
        )

        x[key] = pulp.LpVariable(
            name=f"x_{opt['Shipment_ID']}_{opt['Dispatch_Date'].strftime('%Y%m%d')}_{opt['Mode']}",
            cat="Binary"
        )

    planning_dates = sorted(options_df["Dispatch_Date"].dt.strftime("%Y-%m-%d").unique())

    over = {
        d: pulp.LpVariable(f"over_{d}", lowBound=0, cat="Integer")
        for d in planning_dates
    }

    max_daily_load = pulp.LpVariable("max_daily_load", lowBound=0, cat="Integer")

    for shipment_id in df["Shipment_ID"]:
        shipment_options = [key for key in x.keys() if key[0] == shipment_id]

        model += (
            pulp.lpSum(x[key] for key in shipment_options) == 1,
            f"Assign_once_{shipment_id}"
        )

    daily_load = {}

    for d in planning_dates:
        daily_terms = []

        for key in x.keys():
            shipment_id, dispatch_date, mode = key

            if dispatch_date == d:
                vehicle_count = df.loc[df["Shipment_ID"] == shipment_id, "Vehicle_Count"].iloc[0]
                daily_terms.append(vehicle_count * x[key])

        daily_load[d] = pulp.lpSum(daily_terms)

        model += (
            daily_load[d] <= daily_capacity + over[d],
            f"Daily_capacity_soft_{d}"
        )

        model += (
            daily_load[d] <= max_daily_load,
            f"Peak_load_{d}"
        )

        if absolute_daily_capacity is not None:
            model += (
                daily_load[d] <= absolute_daily_capacity,
                f"Absolute_daily_capacity_{d}"
            )

    BIG_M_OVER = 100000
    BIG_M_PEAK = 1000
    BIG_M_RISK = 100
    BIG_M_DEVIATION = 10
    RAIL_BONUS = 0

    over_capacity_term = pulp.lpSum(over[d] for d in planning_dates)

    risk_term = []
    target_deviation_term = []
    rail_bonus_term = []

    for _, opt in options_df.iterrows():
        shipment_id = opt["Shipment_ID"]
        dispatch_date = opt["Dispatch_Date"]
        mode = opt["Mode"]

        key = (
            shipment_id,
            dispatch_date.strftime("%Y-%m-%d"),
            mode
        )

        priority_weight = df.loc[df["Shipment_ID"] == shipment_id, "Priority_Weight"].iloc[0]

        risk_term.append(priority_weight * opt["Risk_Score"] * x[key])
        target_deviation_term.append(priority_weight * opt["Dispatch_Deviation"] * x[key])

        if mode == "Rail":
            rail_bonus_term.append(x[key])

    model += (
        BIG_M_OVER * over_capacity_term
        + BIG_M_PEAK * max_daily_load
        + BIG_M_RISK * pulp.lpSum(risk_term)
        + BIG_M_DEVIATION * pulp.lpSum(target_deviation_term)
        - RAIL_BONUS * pulp.lpSum(rail_bonus_term)
    )

    solver = pulp.PULP_CBC_CMD(msg=False)
    result_status = model.solve(solver)

    solver_status = pulp.LpStatus[result_status]

    if solver_status not in ["Optimal", "Feasible"]:
        raise ValueError("No feasible solution found. Please check input data and constraints.")

    total_over_capacity = sum(
        pulp.value(over[d]) if pulp.value(over[d]) is not None else 0
        for d in planning_dates
    )

    total_risk = 0
    total_target_deviation = 0
    total_rail_assignments = 0
    total_road_assignments = 0

    solution_rows = []

    for key, var in x.items():
        if pulp.value(var) is not None and pulp.value(var) > 0.5:
            shipment_id, dispatch_date_str, mode = key

            shipment_row = df[df["Shipment_ID"] == shipment_id].iloc[0]

            opt_row = options_df[
                (options_df["Shipment_ID"] == shipment_id)
                & (options_df["Dispatch_Date"].dt.strftime("%Y-%m-%d") == dispatch_date_str)
                & (options_df["Mode"] == mode)
            ].iloc[0]

            priority_weight = shipment_row["Priority_Weight"]

            total_risk += priority_weight * opt_row["Risk_Score"]
            total_target_deviation += priority_weight * opt_row["Dispatch_Deviation"]

            if mode == "Rail":
                total_rail_assignments += 1
            elif mode == "Road":
                total_road_assignments += 1

            solution_rows.append({
                "Shipment_ID": shipment_id,
                "Destination": shipment_row["Destination"],
                "Shipment_Type": shipment_row["Shipment_Type"],
                "Recommended_Mode": mode,
                "Dispatch_Date": opt_row["Dispatch_Date"],
                "Arrival_Date": opt_row["Arrival_Date"],
                "Free_Entry_Date": shipment_row.get("Free_Entry_Date", pd.NaT),
                "Cutoff_Date": shipment_row.get("Cutoff_Date", pd.NaT),
                "Earliest_Dispatch_Date": shipment_row.get("Earliest_Dispatch_Date", pd.NaT),
                "Latest_Dispatch_Date": shipment_row.get("Latest_Dispatch_Date", pd.NaT),
                "Slack_Before_Cutoff": opt_row["Slack_Before_Cutoff"],
                "Dispatch_Deviation": opt_row["Dispatch_Deviation"],
                "Vehicle_Count": shipment_row["Vehicle_Count"],
                "Priority": shipment_row["Priority"],
                "Priority_Weight": shipment_row["Priority_Weight"],
                "Risk_Score": opt_row["Risk_Score"],
                "Preferred_Mode": shipment_row["Preferred_Mode"],
                "Status": "Scheduled",
                "Manual_Override": "No",
                "Delay_Reason": "",
                "Manual_Warnings": ""
            })

    output_df = pd.DataFrame(solution_rows)

    if output_df.empty:
        raise ValueError("The model solved, but no shipment assignment was found.")

    priority_order = {
        "Urgent": 1,
        "High": 2,
        "Acil": 1,
        "Yüksek": 2,
        "Yuksek": 2,
        "Normal": 3,
        "Medium": 3,
        "Orta": 3,
        "Low": 4,
        "Düşük": 4,
        "Dusuk": 4
    }

    output_df["Priority_Order"] = output_df["Priority"].map(priority_order).fillna(3)
    output_df = output_df.sort_values(
        ["Dispatch_Date", "Priority_Order", "Shipment_ID"]
    ).drop(columns=["Priority_Order"])

    daily_summary = calculate_daily_summary(output_df, daily_capacity)

    calculated_objective = (
        BIG_M_OVER * total_over_capacity
        + BIG_M_PEAK * pulp.value(max_daily_load)
        + BIG_M_RISK * total_risk
        + BIG_M_DEVIATION * total_target_deviation
        - RAIL_BONUS * total_rail_assignments
    )

    objective_summary = pd.DataFrame({
        "Metric": [
            "Solver Status",
            "Solver Objective",
            "Calculated Objective",
            "Total Over Capacity",
            "Maximum Daily Load",
            "Weighted Risk Score",
            "Weighted Target Dispatch Deviation",
            "Rail Assignments",
            "Road Assignments",
            "Daily Capacity",
            "Sunday Dispatch Rule"
        ],
        "Value": [
            solver_status,
            pulp.value(model.objective),
            calculated_objective,
            total_over_capacity,
            pulp.value(max_daily_load),
            total_risk,
            total_target_deviation,
            total_rail_assignments,
            total_road_assignments,
            daily_capacity,
            "No dispatch on Sunday"
        ]
    })

    validation_rows = []

    assigned_counts = output_df.groupby("Shipment_ID").size().reset_index(name="Assignment_Count")

    for _, row in assigned_counts.iterrows():
        validation_rows.append({
            "Check": "Each shipment assigned exactly once",
            "Shipment_ID": row["Shipment_ID"],
            "Result": "OK" if row["Assignment_Count"] == 1 else "ERROR",
            "Details": f"Assignment count = {row['Assignment_Count']}"
        })

    for _, row in output_df.iterrows():
        no_sunday_dispatch = row["Dispatch_Date"].weekday() in DISPATCH_AVAILABLE_WEEKDAYS

        validation_rows.append({
            "Check": "Dispatch is not scheduled on Sunday",
            "Shipment_ID": row["Shipment_ID"],
            "Result": "OK" if no_sunday_dispatch else "ERROR",
            "Details": f"Dispatch weekday = {row['Dispatch_Date'].day_name()}"
        })

        if row["Shipment_Type"] == "PORT":
            inside_window = (
                row["Arrival_Date"] >= row["Free_Entry_Date"]
                and row["Arrival_Date"] <= row["Cutoff_Date"]
            )

            validation_rows.append({
                "Check": "PORT arrival within free-entry and cut-off window",
                "Shipment_ID": row["Shipment_ID"],
                "Result": "OK" if inside_window else "ERROR",
                "Details": (
                    f"Arrival={row['Arrival_Date'].date()}, "
                    f"FreeEntry={row['Free_Entry_Date'].date()}, "
                    f"Cutoff={row['Cutoff_Date'].date()}"
                )
            })

        elif row["Shipment_Type"] == "TARGET":
            inside_dispatch_window = (
                row["Dispatch_Date"] >= row["Earliest_Dispatch_Date"]
                and row["Dispatch_Date"] <= row["Latest_Dispatch_Date"]
            )

            validation_rows.append({
                "Check": "TARGET dispatch within earliest-latest dispatch window",
                "Shipment_ID": row["Shipment_ID"],
                "Result": "OK" if inside_dispatch_window else "ERROR",
                "Details": (
                    f"Dispatch={row['Dispatch_Date'].date()}, "
                    f"Earliest={row['Earliest_Dispatch_Date'].date()}, "
                    f"Latest={row['Latest_Dispatch_Date'].date()}"
                )
            })

        if row["Recommended_Mode"] == "Rail":
            rail_day_ok = row["Dispatch_Date"].weekday() in RAIL_AVAILABLE_WEEKDAYS

            validation_rows.append({
                "Check": "Rail used only on Monday, Wednesday, Saturday",
                "Shipment_ID": row["Shipment_ID"],
                "Result": "OK" if rail_day_ok else "ERROR",
                "Details": f"Dispatch weekday = {row['Dispatch_Date'].day_name()}"
            })

    validation_df = pd.DataFrame(validation_rows)

    return output_df, daily_summary, objective_summary, validation_df, options_df


def create_excel_bytes(output_df, daily_summary, objective_summary, validation_df, options_df):
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        output_df.to_excel(writer, sheet_name="Shipment_Schedule", index=False)
        daily_summary.to_excel(writer, sheet_name="Daily_Summary", index=False)
        objective_summary.to_excel(writer, sheet_name="Objective_Summary", index=False)
        validation_df.to_excel(writer, sheet_name="Validation_Checks", index=False)
        options_df.to_excel(writer, sheet_name="Feasible_Options", index=False)

    output.seek(0)
    return output


def create_input_excel_bytes(input_df):
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        make_display_df(input_df).to_excel(writer, sheet_name="Model_Input", index=False)

    output.seek(0)
    return output


def create_template_excel():
    template_df = pd.DataFrame([
        {
            "Shipment_ID": "S001",
            "Destination": "MIP Mersin Port",
            "Shipment_Type": "PORT",
            "Free_Entry_Date": "2026-03-10",
            "Cutoff_Date": "2026-03-12",
            "Earliest_Dispatch_Date": "",
            "Latest_Dispatch_Date": "",
            "Vehicle_Count": 2,
            "Priority": "High",
            "Preferred_Mode": "Auto",
            "Source": "Template"
        },
        {
            "Shipment_ID": "S002",
            "Destination": "Ankara",
            "Shipment_Type": "TARGET",
            "Free_Entry_Date": "",
            "Cutoff_Date": "",
            "Earliest_Dispatch_Date": "2026-03-11",
            "Latest_Dispatch_Date": "2026-03-14",
            "Vehicle_Count": 1,
            "Priority": "Normal",
            "Preferred_Mode": "Road",
            "Source": "Template"
        }
    ])

    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        template_df.to_excel(writer, sheet_name="Model_Input", index=False)

    output.seek(0)
    return output


def apply_manual_change(schedule_df, shipment_id, new_dispatch_date, new_mode, new_status, delay_reason):
    df = schedule_df.copy()

    idx_list = df.index[df["Shipment_ID"] == shipment_id].tolist()

    if len(idx_list) == 0:
        raise ValueError("Shipment not found in schedule.")

    idx = idx_list[0]

    new_dispatch_dt = pd.to_datetime(new_dispatch_date)

    # Hard rule: Sunday cannot be accepted
    if new_dispatch_dt.weekday() not in DISPATCH_AVAILABLE_WEEKDAYS:
        raise ValueError("Sunday dispatch is not allowed. Please select Monday-Saturday.")

    new_mode = clean_mode(new_mode)

    if new_mode == "Auto":
        new_mode = df.loc[idx, "Recommended_Mode"]

    arrival_date = calculate_arrival(new_dispatch_dt, new_mode)

    warnings = []

    if new_mode == "Rail" and new_dispatch_dt.weekday() not in RAIL_AVAILABLE_WEEKDAYS:
        warnings.append("Rail is normally available only on Monday, Wednesday, and Saturday.")

    shipment_type = df.loc[idx, "Shipment_Type"]

    if shipment_type == "PORT":
        free_entry = pd.to_datetime(df.loc[idx, "Free_Entry_Date"])
        cutoff = pd.to_datetime(df.loc[idx, "Cutoff_Date"])

        if arrival_date < free_entry or arrival_date > cutoff:
            warnings.append("Manual change violates free-entry / cut-off window.")

    elif shipment_type == "TARGET":
        earliest = pd.to_datetime(df.loc[idx, "Earliest_Dispatch_Date"])
        latest = pd.to_datetime(df.loc[idx, "Latest_Dispatch_Date"])

        if new_dispatch_dt < earliest or new_dispatch_dt > latest:
            warnings.append("Manual change violates earliest / latest dispatch window.")

    df.loc[idx, "Recommended_Mode"] = new_mode
    df.loc[idx, "Dispatch_Date"] = new_dispatch_dt
    df.loc[idx, "Arrival_Date"] = arrival_date
    df.loc[idx, "Status"] = new_status
    df.loc[idx, "Manual_Override"] = "Yes"
    df.loc[idx, "Delay_Reason"] = delay_reason
    df.loc[idx, "Manual_Warnings"] = " | ".join(warnings)

    return df, warnings

def format_date_safe(value):
    if pd.isna(value) or value == "":
        return "-"
    return pd.to_datetime(value).strftime("%d.%m.%Y")


def get_week_start_from_output(output_df):
    if output_df.empty:
        today = pd.Timestamp.today().normalize()
        return today - pd.Timedelta(days=today.weekday())

    min_date = pd.to_datetime(output_df["Dispatch_Date"]).min().normalize()
    return min_date - pd.Timedelta(days=min_date.weekday())


def status_badge_style(status):
    status = str(status)
    styles = {
        "Scheduled": ("#8E8E93", "rgba(142,142,147,0.15)"),
        "Manually Added": ("#5DADE2", "rgba(93,173,226,0.16)"),
        "Manually Changed": ("#F4D03F", "rgba(244,208,63,0.18)"),
        "Delayed": ("#EB984E", "rgba(235,152,78,0.18)"),
        "Cancelled": ("#EC7063", "rgba(236,112,99,0.18)"),
        "Completed": ("#58D68D", "rgba(88,214,141,0.18)")
    }
    return styles.get(status, ("#8E8E93", "rgba(142,142,147,0.15)"))


def render_shipment_card(row):
    mode = str(row.get("Recommended_Mode", "Road"))
    mode_icon = "🚆" if mode == "Rail" else "🚚"
    mode_color = "#5DADE2" if mode == "Rail" else "#F5B041"
    mode_bg = "rgba(93,173,226,0.12)" if mode == "Rail" else "rgba(245,176,65,0.12)"

    status = str(row.get("Status", "Scheduled"))
    status_color, status_bg = status_badge_style(status)

    if row["Shipment_Type"] == "PORT":
        window_text = (
            f"Free Entry: {format_date_safe(row.get('Free_Entry_Date'))}<br>"
            f"Cut-off: {format_date_safe(row.get('Cutoff_Date'))}"
        )
    else:
        window_text = (
            f"Earliest: {format_date_safe(row.get('Earliest_Dispatch_Date'))}<br>"
            f"Latest: {format_date_safe(row.get('Latest_Dispatch_Date'))}"
        )

    warning = str(row.get("Manual_Warnings", "")).strip()
    warning_html = ""
    if warning not in ["", "nan", "None"]:
        warning_html = f"<br><span style='color:#F5B041;'>⚠️ {warning}</span>"

    card_html = f"""
    <div style="
        border:1px solid rgba(150,150,150,0.35);
        border-left:5px solid {mode_color};
        border-radius:10px;
        padding:8px;
        margin-bottom:6px;
        background:{mode_bg};
        font-size:12px;
        line-height:1.35;
    ">
        <b>{row['Shipment_ID']}</b> | {mode_icon}
        <span style="display:inline-block; padding:1px 7px; border-radius:8px; background:{mode_color}; color:black; font-weight:700;">
            {mode}
        </span><br>
        <b>{row['Destination']}</b><br>
        {window_text}<br>
        Arrival: {format_date_safe(row.get('Arrival_Date'))}<br>
        Vehicles: {row['Vehicle_Count']} | Priority: {row['Priority']}<br>
        <span style="display:inline-block; padding:1px 7px; border-radius:8px; background:{status_bg}; color:{status_color}; font-weight:700;">
            {status}
        </span>
        {warning_html}
    </div>
    """

    st.markdown(card_html, unsafe_allow_html=True)

def render_calendar_actions(row, daily_capacity):
    shipment_id = row["Shipment_ID"]

    with st.popover("⋯", use_container_width=True):
        st.markdown(f"**{shipment_id}**")
        st.caption("View details, move, cancel, or remove this shipment.")

        with st.expander("Details", expanded=True):
            detail_df = pd.DataFrame([row]).copy()
            st.dataframe(make_display_df(detail_df), width="stretch", height=130)

        current_dispatch = pd.to_datetime(row["Dispatch_Date"]).date()

        new_dispatch_date = st.date_input(
            "New Dispatch Date",
            value=current_dispatch,
            key=f"cal_date_{shipment_id}"
        )

        new_mode = st.selectbox(
            "New Mode",
            ["Auto", "Road", "Rail"],
            index=["Auto", "Road", "Rail"].index(row["Recommended_Mode"])
            if row["Recommended_Mode"] in ["Auto", "Road", "Rail"] else 0,
            key=f"cal_mode_{shipment_id}"
        )

        new_status = st.selectbox(
            "Status",
            ["Scheduled", "Manually Changed", "Delayed", "Cancelled", "Completed"],
            index=0,
            key=f"cal_status_{shipment_id}"
        )

        delay_reason = st.selectbox(
            "Reason",
            [
                "",
                "Truck delay",
                "Rail unavailable",
                "Port congestion",
                "Customer request",
                "Production delay",
                "Document issue",
                "Weather / road condition",
                "Other"
            ],
            key=f"cal_reason_{shipment_id}"
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Apply Edit", key=f"apply_{shipment_id}"):
                try:
                    updated_schedule, warnings = apply_manual_change(
                        st.session_state.output_df,
                        shipment_id,
                        new_dispatch_date,
                        new_mode,
                        new_status,
                        delay_reason
                    )

                    st.session_state.output_df = updated_schedule
                    st.session_state.daily_summary = calculate_daily_summary(updated_schedule, daily_capacity)
                    new_dispatch_ts = pd.to_datetime(new_dispatch_date)
                    st.session_state.calendar_week_start = new_dispatch_ts - pd.Timedelta(days=new_dispatch_ts.weekday())
                    
                    save_app_state()

                    if warnings:
                        st.warning("Manual change applied with warnings: " + " | ".join(warnings))
                    else:
                        st.success("Manual change applied.")

                    st.rerun()

                except Exception as e:
                    st.error(str(e))

        with col2:
            if st.button("Cancel", key=f"cancel_{shipment_id}"):
                try:
                    updated_schedule, warnings = apply_manual_change(
                        st.session_state.output_df,
                        shipment_id,
                        current_dispatch,
                        row["Recommended_Mode"],
                        "Cancelled",
                        "Cancelled from calendar"
                    )

                    st.session_state.output_df = updated_schedule
                    st.session_state.daily_summary = calculate_daily_summary(updated_schedule, daily_capacity)
                    save_app_state()
                    st.success("Shipment cancelled.")
                    st.rerun()

                except Exception as e:
                    st.error(str(e))

        st.markdown("---")
        remove_scope = st.radio(
            "Remove option",
            ["Remove from calendar only", "Remove from input and calendar"],
            key=f"remove_scope_{shipment_id}"
        )

        confirm_remove = st.checkbox(
            "I confirm removal",
            key=f"confirm_remove_{shipment_id}"
        )

        if st.button("Remove", key=f"remove_{shipment_id}"):
            if not confirm_remove:
                st.error("Please confirm removal first.")
            else:
                st.session_state.output_df = st.session_state.output_df[
                    st.session_state.output_df["Shipment_ID"].astype(str) != shipment_id
                ].copy()

                if remove_scope == "Remove from input and calendar":
                    st.session_state.input_df = st.session_state.input_df[
                        st.session_state.input_df["Shipment_ID"].astype(str) != shipment_id
                    ].copy()
                    st.session_state.needs_optimization = True

                st.session_state.daily_summary = calculate_daily_summary(st.session_state.output_df, daily_capacity)
                save_app_state()
                st.success("Shipment removed.")
                st.rerun()

def render_week_calendar(schedule_df, week_start, daily_capacity):
    week_start = pd.to_datetime(week_start).normalize()
    week_dates = [week_start + pd.Timedelta(days=i) for i in range(7)]

    if week_dates[0].month == week_dates[-1].month:
        month_title = f"{MONTH_NAMES_TR[week_dates[0].month]} {week_dates[0].year}"
    else:
        month_title = (
            f"{MONTH_NAMES_TR[week_dates[0].month]} {week_dates[0].year} - "
            f"{MONTH_NAMES_TR[week_dates[-1].month]} {week_dates[-1].year}"
        )

    st.markdown(f"### {month_title}")
    st.caption(
        f"Week: {week_dates[0].strftime('%d.%m.%Y')} - "
        f"{week_dates[-1].strftime('%d.%m.%Y')}"
    )

    active_schedule = schedule_df[schedule_df["Status"] != "Cancelled"].copy()
    active_schedule["Dispatch_Date"] = pd.to_datetime(active_schedule["Dispatch_Date"]).dt.normalize()

    cols = st.columns(7)

    for i, day in enumerate(week_dates):
        with cols[i]:
            is_sunday = day.weekday() == 6

            day_df = active_schedule[
                active_schedule["Dispatch_Date"] == day
            ].copy()

            day_df = day_df.sort_values(["Recommended_Mode", "Shipment_ID"])

            total_vehicles = int(day_df["Vehicle_Count"].sum()) if not day_df.empty else 0
            over_capacity = max(0, total_vehicles - daily_capacity)

            with st.container(border=True):
                st.markdown(f"**{DAY_NAMES_TR[day.weekday()]}**")
                st.caption(day.strftime("%d.%m.%Y"))

                if is_sunday:
                    st.error("Closed")
                else:
                    if over_capacity > 0:
                        st.warning(f"Vehicles: {total_vehicles}/{daily_capacity} | Over: {over_capacity}")
                    else:
                        st.success(f"Vehicles: {total_vehicles}/{daily_capacity}")

                if day_df.empty:
                    st.write("No shipments")
                else:
                    for _, row in day_df.iterrows():
                        render_shipment_card(row)
                        render_calendar_actions(row, daily_capacity)

load_app_state_once()
# ============================================================
# UI
# ============================================================


st.title("🚚 AYD Shipment Scheduling Decision Support Prototype")

st.caption(
    "PORT shipments use free-entry / cut-off windows. "
    "TARGET shipments use earliest / latest dispatch windows. "
    "Sunday dispatch is not allowed."
)


# ============================================================
# SIDEBAR SETTINGS
# ============================================================

st.sidebar.header("Model Settings")

daily_capacity = st.sidebar.number_input(
    "Daily vehicle capacity",
    min_value=1,
    value=DEFAULT_DAILY_CAPACITY,
    step=1
)

use_absolute_capacity = st.sidebar.checkbox("Use absolute daily capacity", value=False)

absolute_daily_capacity = None

if use_absolute_capacity:
    absolute_daily_capacity = st.sidebar.number_input(
        "Absolute daily capacity",
        min_value=daily_capacity,
        value=max(daily_capacity, 50),
        step=1
    )

st.sidebar.markdown("---")
st.sidebar.write("Dispatch days: Monday-Saturday")
st.sidebar.write("Sunday: Not allowed")
st.sidebar.write("Rail days: Monday, Wednesday, Saturday")

st.sidebar.markdown("---")
st.sidebar.subheader("Live Summary")
st.sidebar.markdown("---")
with st.sidebar.expander("Saved Data"):
    st.caption("The app automatically keeps the last input and calendar locally.")
    if st.button("Clear Saved Input and Calendar"):
        clear_saved_state()
        st.success("Saved data cleared. Please refresh the app.")
        st.rerun()
st.sidebar.metric("Input Shipments", len(st.session_state.input_df))
st.sidebar.metric("Scheduled Shipments", 0 if st.session_state.output_df.empty else len(st.session_state.output_df[st.session_state.output_df["Status"] != "Cancelled"]))
if not st.session_state.output_df.empty:
    active_sidebar_df = st.session_state.output_df[st.session_state.output_df["Status"] != "Cancelled"].copy()
    st.sidebar.metric("Total Vehicles", int(active_sidebar_df["Vehicle_Count"].sum()))
else:
    st.sidebar.metric("Total Vehicles", 0)

over_capacity_days_sidebar = get_upcoming_over_capacity_days(st.session_state.daily_summary)

st.sidebar.metric("Upcoming Over-capacity Days", len(over_capacity_days_sidebar))

if st.session_state.needs_optimization:
    st.sidebar.warning(
        "Input changed. Summary and over-capacity days are based on the previous optimization."
    )

with st.sidebar.expander("View Upcoming Over-capacity Days"):
    if over_capacity_days_sidebar.empty:
        st.write("No upcoming over-capacity days.")
    else:
        display_over_capacity_df = over_capacity_days_sidebar.copy()
        display_over_capacity_df["Dispatch_Date"] = pd.to_datetime(
            display_over_capacity_df["Dispatch_Date"],
            errors="coerce"
        ).dt.strftime("%Y-%m-%d")

        st.dataframe(display_over_capacity_df, width="stretch", height=220)


# ============================================================
# TABS
# ============================================================

tab_input, tab_calendar, tab_edit, tab_download = st.tabs([
    "1) Input & Optimize",
    "2) Calendar",
    "3) Manual Edit / Logs",
    "4) Download"
])


# ============================================================
# TAB 1: INPUT
# ============================================================

with tab_input:
    st.subheader("Shipment Input")

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown("### Excel Upload")

        uploaded_file = st.file_uploader(
            "Upload input Excel file",
            type=["xlsx"],
            key="excel_uploader"
        )

        if uploaded_file is not None:
            try:
                file_bytes = uploaded_file.getvalue()
                current_file_id = f"{uploaded_file.name}_{len(file_bytes)}"
                if st.session_state.last_uploaded_file_id != current_file_id:
                    
                    uploaded_df = pd.read_excel(BytesIO(file_bytes), sheet_name=0)
                    uploaded_df.columns = uploaded_df.columns.str.strip()

                    uploaded_df["Source"] = "Excel"
                    st.session_state.input_df = normalize_input_for_session(uploaded_df)
                    st.session_state.last_uploaded_file_id = current_file_id
                    clear_results_after_input_change()
                    save_app_state()

                st.success(f"Excel input uploaded successfully. Rows loaded: {len(st.session_state.input_df)}")
                st.markdown("### Uploaded Input Preview")
                st.dataframe(make_display_df(st.session_state.input_df).head(30), width="stretch", height=300)

            except Exception as e:
                st.error(f"Could not read Excel file: {e}")

        template_bytes = create_template_excel()

        st.download_button(
            label="Download input template",
            data=template_bytes,
            file_name="Model_Input_Template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    with col_b:
        st.markdown("### Manual Shipment Entry")

        shipment_type_label = st.radio(
            "Shipment type",
            ["Port / Export", "Domestic / Target"],
            horizontal=True,
            key="manual_shipment_type"
        )

        shipment_type = "PORT" if shipment_type_label == "Port / Export" else "TARGET"

        with st.form("manual_shipment_form", clear_on_submit=True):

            shipment_id = st.text_input("Shipment ID")
            destination = st.text_input("Destination")

            vehicle_count = st.number_input(
                "Vehicle Count",
                min_value=1,
                value=1,
                step=1
            )

            priority = st.selectbox(
                "Priority",
                ["Urgent", "High", "Normal", "Low"]
            )

            preferred_mode = st.selectbox(
                "Preferred Mode",
                ["Auto", "Road", "Rail"]
            )

            manual_add_action = st.selectbox(
                "After Adding Shipment",
                [
                    "Add to Input List Only",
                    "Add Directly to Calendar"
                ]
            )

            st.caption("Input List Only: the shipment appears in the calendar after optimization.")
            st.caption("Direct Calendar: the shipment is manually scheduled without optimization.")

            direct_dispatch_date = None
            direct_mode = None

            if manual_add_action == "Add Directly to Calendar":
                direct_dispatch_date = st.date_input("Manual Dispatch Date")
                direct_mode = st.selectbox("Manual Mode", ["Road", "Rail"])

            free_entry_date = pd.NaT
            cutoff_date = pd.NaT
            earliest_dispatch_date = pd.NaT
            latest_dispatch_date = pd.NaT

            if shipment_type == "PORT":
                free_entry_date = st.date_input("Free Entry Date")
                cutoff_date = st.date_input("Cut-off Date")
            else:
                earliest_dispatch_date = st.date_input("Earliest Dispatch Date")
                latest_dispatch_date = st.date_input("Latest Dispatch Date")

            add_button = st.form_submit_button("Add Shipment")

        if add_button:
            if shipment_id.strip() == "" or destination.strip() == "":
                st.error("Shipment ID and Destination cannot be empty.")
            elif shipment_id.strip() in st.session_state.input_df["Shipment_ID"].astype(str).values:
                st.error("This Shipment ID already exists. Please use a unique Shipment ID.")
            else:
                new_row = pd.DataFrame([{
                    "Shipment_ID": shipment_id.strip(),
                    "Destination": destination.strip(),
                    "Shipment_Type": shipment_type,
                    "Free_Entry_Date": free_entry_date if shipment_type == "PORT" else pd.NaT,
                    "Cutoff_Date": cutoff_date if shipment_type == "PORT" else pd.NaT,
                    "Earliest_Dispatch_Date": earliest_dispatch_date if shipment_type == "TARGET" else pd.NaT,
                    "Latest_Dispatch_Date": latest_dispatch_date if shipment_type == "TARGET" else pd.NaT,
                    "Vehicle_Count": int(vehicle_count),
                    "Priority": priority,
                    "Preferred_Mode": preferred_mode,
                    "Source": "Manual"
                }])

                st.session_state.input_df = normalize_input_for_session(
                    pd.concat([st.session_state.input_df, new_row], ignore_index=True)
                )

                if manual_add_action == "Add to Input List Only":
                    clear_results_after_input_change()
                    save_app_state()
                    st.success("Shipment added to the input list. Run optimization to place it into the calendar.")

                elif manual_add_action == "Add Directly to Calendar":
                    direct_dispatch_ts = pd.to_datetime(direct_dispatch_date)

                    if direct_dispatch_ts.weekday() == 6:
                        st.error("Sunday dispatch is not allowed. Please select Monday-Saturday.")
                    elif direct_mode == "Rail" and direct_dispatch_ts.weekday() not in RAIL_AVAILABLE_WEEKDAYS:
                        st.error("Rail can only be used on Monday, Wednesday, and Saturday.")
                    else:
                        arrival_date = calculate_arrival(direct_dispatch_ts, direct_mode)
                        manual_warnings = []

                        if shipment_type == "PORT":
                            free_entry_ts = pd.to_datetime(free_entry_date)
                            cutoff_ts = pd.to_datetime(cutoff_date)

                            if arrival_date < free_entry_ts or arrival_date > cutoff_ts:
                                manual_warnings.append("Manual decision violates free-entry / cut-off window.")
                        elif shipment_type == "TARGET":
                            earliest_ts = pd.to_datetime(earliest_dispatch_date)
                            latest_ts = pd.to_datetime(latest_dispatch_date)

                            if direct_dispatch_ts < earliest_ts or direct_dispatch_ts > latest_ts:
                                manual_warnings.append("Manual decision violates earliest / latest dispatch window.")

                        manual_schedule_row = pd.DataFrame([{
                            "Shipment_ID": shipment_id.strip(),
                            "Destination": destination.strip(),
                            "Shipment_Type": shipment_type,
                            "Recommended_Mode": direct_mode,
                            "Dispatch_Date": direct_dispatch_ts,
                            "Arrival_Date": arrival_date,
                            "Free_Entry_Date": pd.to_datetime(free_entry_date) if shipment_type == "PORT" else pd.NaT,
                            "Cutoff_Date": pd.to_datetime(cutoff_date) if shipment_type == "PORT" else pd.NaT,
                            "Earliest_Dispatch_Date": pd.to_datetime(earliest_dispatch_date) if shipment_type == "TARGET" else pd.NaT,
                            "Latest_Dispatch_Date": pd.to_datetime(latest_dispatch_date) if shipment_type == "TARGET" else pd.NaT,
                            "Slack_Before_Cutoff": None,
                            "Dispatch_Deviation": None,
                            "Vehicle_Count": int(vehicle_count),
                            "Priority": priority,
                            "Priority_Weight": PRIORITY_WEIGHT.get(priority, 2),
                            "Risk_Score": None,
                            "Preferred_Mode": preferred_mode,
                            "Status": "Manually Added",
                            "Manual_Override": "Yes",
                            "Delay_Reason": "",
                            "Manual_Warnings": " | ".join(manual_warnings),
                            "Source": "Manual"
                        }])

                        st.session_state.output_df = pd.concat(
                            [st.session_state.output_df, manual_schedule_row],
                            ignore_index=True
                        )

                        st.session_state.daily_summary = calculate_daily_summary(st.session_state.output_df, daily_capacity)
                        st.session_state.calendar_week_start = direct_dispatch_ts - pd.Timedelta(days=direct_dispatch_ts.weekday())
                        st.session_state.needs_optimization = True
                        save_app_state()

                        if manual_warnings:
                            st.warning("Shipment added directly to calendar with warning: " + " | ".join(manual_warnings))
                        else:
                            st.success("Shipment added directly to the calendar as a manual decision.")
    if st.session_state.needs_optimization:
        st.warning(
            "Input data has changed. Existing calendar and over-capacity values are based on the previous optimization. "
            "Run optimization again to update the schedule."
        )
    st.markdown("### Current Input Table")
    
    st.caption(f"Total input rows: {len(st.session_state.input_df)}")
    display_input_df = make_display_df(st.session_state.input_df)
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns([1, 1, 1, 1])
    with filter_col1:
        search_shipment_id = st.text_input("Search Shipment ID")
    with filter_col2:
        search_destination = st.text_input("Search Destination")
    with filter_col3:
        selected_type_filter = st.selectbox(
            "Shipment Type Filter",
            ["All", "PORT", "TARGET"]
        )
    with filter_col4:
        selected_source_filter = st.selectbox(
            "Source Filter",
            ["All", "Excel", "Manual", "Template", "Unknown"]
        )
    filtered_input_df = display_input_df.copy()
    if search_shipment_id.strip() != "":
        filtered_input_df = filtered_input_df[
            filtered_input_df["Shipment_ID"].astype(str).str.contains(
                search_shipment_id.strip(),
                case=False,
                na=False
            )
        ]
    if search_destination.strip() != "":
        filtered_input_df = filtered_input_df[
            filtered_input_df["Destination"].astype(str).str.contains(
                search_destination.strip(),
                case=False,
                na=False
            )
        ]
    if selected_type_filter != "All":
        filtered_input_df = filtered_input_df[
            filtered_input_df["Shipment_Type"].astype(str).str.upper() == selected_type_filter
        ]
    if selected_source_filter != "All" and "Source" in filtered_input_df.columns:
        filtered_input_df = filtered_input_df[
            filtered_input_df["Source"].astype(str) == selected_source_filter
        ]
    st.caption(f"Filtered rows: {len(filtered_input_df)}")
    st.dataframe(filtered_input_df,width="stretch",height=400)
    
    st.markdown("### Optimization")
    if st.session_state.needs_optimization:
        st.warning("Input data has changed. Run optimization to update the model-generated calendar.")

    if st.button("Run Optimization with Current Inputs", type="primary"):
        try:
            output_df, daily_summary, objective_summary, validation_df, options_df = run_optimization(
                st.session_state.input_df,
                daily_capacity=daily_capacity,
                absolute_daily_capacity=absolute_daily_capacity
            )
            st.session_state.output_df = output_df
            st.session_state.daily_summary = daily_summary
            st.session_state.objective_summary = objective_summary
            st.session_state.validation_df = validation_df
            st.session_state.options_df = options_df
            st.session_state.calendar_week_start = get_week_start_from_output(output_df)
            st.session_state.needs_optimization = False
            mark_optimization_complete()
            save_app_state()
            
            st.success("Optimization completed successfully. Calendar has been updated.")
            
        except Exception as e:
           st.error(str(e))
                

    

    if not st.session_state.input_df.empty:
        remove_id = st.selectbox(
            "Remove shipment by Shipment_ID",
            [""] + list(st.session_state.input_df["Shipment_ID"].astype(str).unique())
        )

        if st.button("Remove selected shipment"):
            if remove_id != "":
                st.session_state.input_df = st.session_state.input_df[
                    st.session_state.input_df["Shipment_ID"].astype(str) != remove_id
                ].copy()
                clear_results_after_input_change()
                save_app_state()
                st.success(f"Shipment {remove_id} removed.")


# ============================================================
# TAB 2: CALENDAR
# ============================================================

with tab_calendar:
    st.subheader("Weekly Calendar View")

    if st.session_state.output_df.empty:
        st.warning("Run the optimization first.")
    else:
        schedule_df = st.session_state.output_df.copy()
        schedule_df["Dispatch_Date"] = pd.to_datetime(schedule_df["Dispatch_Date"])

        nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 1, 1, 3])

        with nav_col1:
            if st.button("◀ Previous week"):
                st.session_state.calendar_week_start = (
                    pd.to_datetime(st.session_state.calendar_week_start)
                    - pd.Timedelta(days=7)
                )

        with nav_col2:
            if st.button("Next week ▶"):
                st.session_state.calendar_week_start = (
                    pd.to_datetime(st.session_state.calendar_week_start)
                    + pd.Timedelta(days=7)
                )

        with nav_col3:
            if st.button("First scheduled week"):
                st.session_state.calendar_week_start = get_week_start_from_output(schedule_df)

        with nav_col4:
            st.info("Sunday dispatch is closed. You can edit, move, cancel, or remove shipments directly from the calendar.")

        render_week_calendar(
            schedule_df,
            st.session_state.calendar_week_start,
            daily_capacity
        )


# ============================================================
# TAB 4: MANUAL EDIT
# ============================================================

with tab_edit:
    st.subheader("Manual Edit / Move / Cancel")

    if st.session_state.output_df.empty:
        st.warning("Run the optimization first.")
    else:
        schedule_df = st.session_state.output_df.copy()

        shipment_ids = list(schedule_df["Shipment_ID"].astype(str).unique())

        selected_shipment = st.selectbox(
            "Select Shipment_ID",
            shipment_ids
        )

        selected_row = schedule_df[schedule_df["Shipment_ID"] == selected_shipment].iloc[0]

        st.markdown("### Selected shipment")
        st.dataframe(pd.DataFrame([selected_row]), use_container_width=True)

        current_dispatch = pd.to_datetime(selected_row["Dispatch_Date"]).date()

        new_dispatch_date = st.date_input(
            "New Dispatch_Date",
            value=current_dispatch
        )

        new_mode = st.selectbox(
            "New Mode",
            ["Auto", "Road", "Rail"],
            index=["Auto", "Road", "Rail"].index(selected_row["Recommended_Mode"])
            if selected_row["Recommended_Mode"] in ["Auto", "Road", "Rail"] else 0
        )

        new_status = st.selectbox(
            "Status",
            ["Scheduled", "Manually Changed", "Delayed", "Cancelled", "Completed"]
        )

        delay_reason = st.selectbox(
            "Delay / change reason",
            [
                "",
                "Truck delay",
                "Rail unavailable",
                "Port congestion",
                "Customer request",
                "Production delay",
                "Document issue",
                "Weather / road condition",
                "Other"
            ]
        )

        st.info(
            "Manual changes outside model rules will be accepted with warning, "
            "except Sunday dispatch, which is blocked."
        )

        if st.button("Apply manual change", type="primary"):
            try:
                updated_schedule, warnings = apply_manual_change(
                    st.session_state.output_df,
                    selected_shipment,
                    new_dispatch_date,
                    new_mode,
                    new_status,
                    delay_reason
                )

                st.session_state.output_df = updated_schedule
                st.session_state.daily_summary = calculate_daily_summary(
                    updated_schedule,
                    daily_capacity
                )
                new_dispatch_ts = pd.to_datetime(new_dispatch_date)
                st.session_state.calendar_week_start = (
                    new_dispatch_ts - pd.Timedelta(days=new_dispatch_ts.weekday())
                )
                save_app_state()
                

                if len(warnings) > 0:
                    st.warning("Manual change applied with warnings: " + " | ".join(warnings))
                else:
                    st.success("Manual change applied.")

            except Exception as e:
                st.error(str(e))

        if not st.session_state.daily_summary.empty:
            st.markdown("### Updated Daily Summary")
            st.dataframe(st.session_state.daily_summary, use_container_width=True)


# ============================================================
# TAB 5: DOWNLOAD
# ============================================================

with tab_download:
    st.subheader("Download")

    if not st.session_state.input_df.empty:
        input_excel_bytes = create_input_excel_bytes(st.session_state.input_df)
        st.download_button(
            label="Download Current Input",
            data=input_excel_bytes,
            file_name="Current_Model_Input.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    st.markdown("---")
    st.markdown("### Optimized Schedule")

    if st.session_state.output_df.empty:
        st.warning("Run the optimization first.")
    else:
        excel_bytes = create_excel_bytes(
            st.session_state.output_df,
            st.session_state.daily_summary,
            st.session_state.objective_summary,
            st.session_state.validation_df,
            st.session_state.options_df
        )

        st.download_button(
            label="Download output Excel",
            data=excel_bytes,
            file_name="Model_Output_Streamlit.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.markdown("### Final Shipment Schedule")
        st.dataframe(st.session_state.output_df, use_container_width=True)
