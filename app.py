from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "machine_failure_model.pkl"
DATA_PATH = PROJECT_DIR / "ai4i2020 (1).csv"
TARGET_COLUMN = "Machine failure"
REPORTED_METRICS = {
    "accuracy": 0.98,
    "failure_precision": 0.64,
    "failure_recall": 0.85,
    "failure_f1": 0.73,
    "test_rows": 2000,
}

st.set_page_config(
    page_title="Machine Failure Prediction",
    page_icon="⚙️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f5f8fc 0%, #edf3f8 100%);
    }
    [data-testid="stMain"] {
        color: #10243a;
    }
    [data-testid="stMain"] h1,
    [data-testid="stMain"] h2,
    [data-testid="stMain"] h3,
    [data-testid="stMain"] p,
    [data-testid="stMain"] label,
    [data-testid="stMain"] [data-testid="stWidgetLabel"] {
        color: #10243a;
    }
    [data-testid="stMain"] input,
    [data-testid="stMain"] [data-baseweb="select"] > div {
        background-color: #ffffff;
        color: #10243a;
        border-color: #cbd7e3;
    }
    [data-testid="stSidebar"] {
        background: #10243a;
    }
    [data-testid="stSidebar"] * {
        color: #edf5ff;
    }
    [data-testid="stSidebar"] [data-testid="stMetric"] {
        background: #1b344d;
        border: 1px solid #35516d;
        padding: 0.75rem;
        border-radius: 0.75rem;
    }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #71d5c2;
    }
    .stButton > button {
        min-height: 2.8rem;
        border: 0;
        border-radius: 0.65rem;
        background: #10243a;
        color: #ffffff !important;
        font-weight: 650;
    }
    .stButton > button p {
        color: #ffffff !important;
    }
    .stButton > button:hover {
        background: #1b344d;
        border: 0;
        color: #ffffff !important;
    }
    .stButton > button:hover p {
        color: #ffffff !important;
    }
    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #dce5ee;
        padding: 0.85rem;
        border-radius: 0.75rem;
        box-shadow: 0 2px 8px rgba(16, 36, 58, 0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_artifact():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


def make_model_features(raw_data, label_encoder, feature_columns):
    features = raw_data.copy()
    features["Type"] = label_encoder.transform(features["Type"])
    features["Temperature difference [K]"] = (
        features["Process temperature [K]"] - features["Air temperature [K]"]
    )
    features["Mechanical power [W]"] = (
        features["Torque [Nm]"] * features["Rotational speed [rpm]"] * 2 * np.pi / 60
    )
    features["Torque Tool wear"] = features["Torque [Nm]"] * features["Tool wear [min]"]
    return features[feature_columns]


model_artifact = load_model_artifact()
dataset = load_dataset()
model = model_artifact["model"]
label_encoder = model_artifact["label_encoder"]
scaler = model_artifact["scaler"]
feature_columns = model_artifact["feature_columns"]
model_name = model_artifact["model_name"]

failure_count = int(dataset[TARGET_COLUMN].sum())
failure_rate = failure_count / len(dataset)
normal_count = len(dataset) - failure_count

with st.sidebar:
    st.title("About the dataset")
    st.write(
        "This app uses the AI4I 2020 predictive-maintenance dataset. "
        "Each row describes a machine's operating conditions and whether "
        "a failure occurred."
    )
    st.metric("Machine records", f"{len(dataset):,}")
    st.metric("Input features", len(feature_columns))
    st.metric("Recorded failures", f"{failure_count:,} ({failure_rate:.1%})")
    st.caption(
        f"{normal_count:,} records are labeled as no failure. "
        "The failure classes are imbalanced, so recall and F1 are useful "
        "alongside accuracy."
    )

    st.divider()
    st.subheader("Model performance")
    st.write(f"**Prediction model:** {model_name}")
    st.caption(
        "Reported notebook results · 2,000 held-out records · "
        "precision, recall, and F1 are for failure class (1)."
    )
    st.metric("Overall accuracy", f"{REPORTED_METRICS['accuracy']:.0%}")
    st.metric("Failure precision", f"{REPORTED_METRICS['failure_precision']:.0%}")
    st.metric("Failure recall", f"{REPORTED_METRICS['failure_recall']:.0%}")
    st.metric("Failure F1 score", f"{REPORTED_METRICS['failure_f1']:.0%}")

st.title("Machine failure prediction")
st.write(
    "Estimate the likelihood of a machine failure from its operating "
    "conditions. The saved model is **{0}**.".format(model_name)
)

with st.container(border=True):
    st.subheader("Machine operating conditions")
    first_row = st.columns(2)
    with first_row[0]:
        machine_type = st.selectbox("Machine type", ["L", "M", "H"])
        air_temperature = st.number_input(
            "Air temperature [K]",
            min_value=295.3,
            max_value=304.5,
            value=300.0,
        )
        process_temperature = st.number_input(
            "Process temperature [K]",
            min_value=305.7,
            max_value=313.8,
            value=310.0,
        )
    with first_row[1]:
        rotational_speed = st.number_input(
            "Rotational speed [rpm]",
            min_value=1168,
            max_value=2886,
            value=1500,
        )
        torque = st.number_input(
            "Torque [Nm]",
            min_value=3.8,
            max_value=76.6,
            value=40.0,
        )
        tool_wear = st.number_input(
            "Tool wear [min]",
            min_value=0,
            max_value=253,
            value=100,
        )

    input_data = pd.DataFrame(
        {
            "Type": [machine_type],
            "Air temperature [K]": [air_temperature],
            "Process temperature [K]": [process_temperature],
            "Rotational speed [rpm]": [rotational_speed],
            "Torque [Nm]": [torque],
            "Tool wear [min]": [tool_wear],
        }
    )
    input_features = make_model_features(input_data, label_encoder, feature_columns)

    st.divider()
    st.subheader("Calculated operating indicators")
    st.caption("Calculated from the current operating conditions and used by the model.")
    feature_columns_ui = st.columns(3)
    feature_columns_ui[0].metric(
        "Temperature difference",
        f"{input_features['Temperature difference [K]'].iloc[0]:.2f} K",
    )
    feature_columns_ui[1].metric(
        "Mechanical power",
        f"{input_features['Mechanical power [W]'].iloc[0]:,.2f} W",
    )
    feature_columns_ui[2].metric(
        "Torque × wear",
        f"{input_features['Torque Tool wear'].iloc[0]:,.2f} Nm·min",
    )

    predict_clicked = st.button(
        "Predict machine failure",
        type="primary",
        icon=":material/precision_manufacturing:",
    )

if predict_clicked:
    input_for_prediction = scaler.transform(input_features) if scaler is not None else input_features
    prediction = model.predict(input_for_prediction)[0]

    if prediction == 1:
        st.error("Machine failure predicted. Review the operating conditions.")
    else:
        st.success("No machine failure predicted for these operating conditions.")
