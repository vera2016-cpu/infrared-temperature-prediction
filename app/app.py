import streamlit as st
import pandas as pd
import joblib

 # Configure the page
st.set_page_config(
    page_title="Infrared Temperature Predictor",
    page_icon="🌡️",
    layout="wide"
)

# Add the application title
st.title("🌡️ Infrared Oral Temperature Predictor")

st.write(
    """
    This application uses a machine learning model trained on
    infrared thermography measurements to estimate oral temperature.
    """
)

# Load the saved model
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "infrared_temperature_pipeline.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

# Add a divider
st.divider()

# Create the first input section
st.subheader("👤 Subject Information")

col1, col2, col3 = st.columns(3)
with col1:
    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

with col2:
    age = st.selectbox(
        "Age",
        [
            "18-20",
            "21-25",
            "21-30",
            "26-30",
            "31-40",
            "41-50",
            "51-60",
            ">60"
        ]
    )

with col3:
    ethnicity = st.selectbox(
        "Ethnicity",
        [
            "American Indian or Alaskan Native",
            "Asian",
            "Black or African-American",
            "Hispanic/Latino",
            "Multiracial",
            "White"
        ]
    )

# Environmental measurements
st.subheader("🌡️ Environmental Conditions")

col1, col2, col3, col4 = st.columns(4)

with col1:
    T_atm = st.number_input(
        "Ambient Temperature (°C)",
        min_value=0.0,
        max_value=50.0,
        value=25.0,
        step=0.1
    )

with col2:
    Humidity = st.number_input(
        "Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=0.1
    )

with col3:
    Distance = st.number_input(
        "Distance",
        min_value=0.0,
        max_value=10.0,
        value=0.8,
        step=0.1
    )

with col4:
    T_offset1 = st.number_input(
        "Temperature Offset",
        value=0.0,
        step=0.01
    )

st.subheader("🔬 Infrared Thermography Measurements")

col1, col2, col3 = st.columns(3)

with col1:
    Max1R13_1 = st.number_input("Max1R13_1", value=35.0)
    Max1L13_1 = st.number_input("Max1L13_1", value=35.0)
    aveAllR13_1 = st.number_input("aveAllR13_1", value=35.0)
    aveAllL13_1 = st.number_input("aveAllL13_1", value=35.0)
    T_RC1 = st.number_input("T_RC1", value=35.0)
    T_RC_Dry1 = st.number_input("T_RC_Dry1", value=35.0)
    T_RC_Wet1 = st.number_input("T_RC_Wet1", value=35.0)
    T_RC_Max1 = st.number_input("T_RC_Max1", value=35.0)
    T_LC1 = st.number_input("T_LC1", value=35.0)
    T_LC_Dry1 = st.number_input("T_LC_Dry1", value=35.0)

with col2:
    T_LC_Wet1 = st.number_input("T_LC_Wet1", value=35.0)
    T_LC_Max1 = st.number_input("T_LC_Max1", value=35.0)
    RCC1 = st.number_input("RCC1", value=35.0)
    LCC1 = st.number_input("LCC1", value=35.0)
    canthiMax1 = st.number_input("canthiMax1", value=35.0)
    canthi4Max1 = st.number_input("canthi4Max1", value=35.0)
    T_FHCC1 = st.number_input("T_FHCC1", value=35.0)
    T_FHRC1 = st.number_input("T_FHRC1", value=35.0)
    T_FHLC1 = st.number_input("T_FHLC1", value=35.0)
    T_FHBC1 = st.number_input("T_FHBC1", value=35.0)

with col3:
    T_FHTC1 = st.number_input("T_FHTC1", value=35.0)
    T_FH_Max1 = st.number_input("T_FH_Max1", value=35.0)
    T_FHC_Max1 = st.number_input("T_FHC_Max1", value=35.0)
    T_Max1 = st.number_input("T_Max1", value=35.0)
    T_OR1 = st.number_input("T_OR1", value=35.0)
    T_OR_Max1 = st.number_input("T_OR_Max1", value=35.0)

# Create the prediction DataFrame
st.divider()

st.subheader("🔮 Temperature Prediction")

input_data = pd.DataFrame({
    "Gender": [gender],
    "Age": [age],
    "Ethnicity": [ethnicity],
    "T_atm": [T_atm],
    "Humidity": [Humidity],
    "Distance": [Distance],
    "T_offset1": [T_offset1],
    "Max1R13_1": [Max1R13_1],
    "Max1L13_1": [Max1L13_1],
    "aveAllR13_1": [aveAllR13_1],
    "aveAllL13_1": [aveAllL13_1],
    "T_RC1": [T_RC1],
    "T_RC_Dry1": [T_RC_Dry1],
    "T_RC_Wet1": [T_RC_Wet1],
    "T_RC_Max1": [T_RC_Max1],
    "T_LC1": [T_LC1],
    "T_LC_Dry1": [T_LC_Dry1],
    "T_LC_Wet1": [T_LC_Wet1],
    "T_LC_Max1": [T_LC_Max1],
    "RCC1": [RCC1],
    "LCC1": [LCC1],
    "canthiMax1": [canthiMax1],
    "canthi4Max1": [canthi4Max1],
    "T_FHCC1": [T_FHCC1],
    "T_FHRC1": [T_FHRC1],
    "T_FHLC1": [T_FHLC1],
    "T_FHBC1": [T_FHBC1],
    "T_FHTC1": [T_FHTC1],
    "T_FH_Max1": [T_FH_Max1],
    "T_FHC_Max1": [T_FHC_Max1],
    "T_Max1": [T_Max1],
    "T_OR1": [T_OR1],
    "T_OR_Max1": [T_OR_Max1]
})

# Add the Predict button
st.divider()

st.subheader("🔮 Temperature Prediction")

if st.button("🌡️ Predict Oral Temperature", type="primary"):

    prediction = model.predict(input_data)

    predicted_temperature = prediction[0]

    st.success(
        f"🌡️ Predicted Oral Temperature: {predicted_temperature:.2f} °C"
    )

    st.info(
        "This prediction is generated by a machine learning model "
        "trained on infrared temperature measurements."
    )

    # Temperature interpretation
    if predicted_temperature < 36.1:
        st.warning(
            "The predicted temperature is relatively low."
        )

    elif predicted_temperature <= 37.5:
        st.success(
    "The predicted temperature is within the reference range used for this application."
)

    else:
        st.warning(
            "The predicted temperature is relatively high."
        )

st.caption(
    "Disclaimer: This application is for educational and research purposes "
    "only and is not intended for medical diagnosis or clinical decision-making."
)