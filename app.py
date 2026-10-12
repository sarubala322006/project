import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Crop Recommendation System",
    page_icon="🌱",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODEL AND DATASET
# --------------------------------------------------

model = joblib.load("crop_model.pkl")

dataset = pd.read_csv(
    "data/Crop_recommendation.csv"
)

# --------------------------------------------------
# CROP INFORMATION - ALL 22 CROPS
# --------------------------------------------------

crop_info = {

    "apple": {
        "soil": "Well-drained loamy soil",
        "water": "Moderate",
        "season": "Winter",
        "climate": "Cool"
    },

    "banana": {
        "soil": "Rich, well-drained soil",
        "water": "High",
        "season": "Year-round",
        "climate": "Warm and humid"
    },

    "blackgram": {
        "soil": "Clayey and loamy soil",
        "water": "Moderate",
        "season": "Kharif",
        "climate": "Warm and humid"
    },

    "chickpea": {
        "soil": "Sandy and loamy soil",
        "water": "Low to moderate",
        "season": "Rabi",
        "climate": "Cool and dry"
    },

    "coconut": {
        "soil": "Sandy and loamy soil",
        "water": "High",
        "season": "Year-round",
        "climate": "Warm and humid"
    },

    "coffee": {
        "soil": "Rich, well-drained soil",
        "water": "Moderate",
        "season": "Year-round",
        "climate": "Warm and humid"
    },

    "cotton": {
        "soil": "Black soil",
        "water": "Moderate",
        "season": "Kharif",
        "climate": "Warm"
    },

    "grapes": {
        "soil": "Well-drained loamy soil",
        "water": "Moderate",
        "season": "Year-round",
        "climate": "Warm and dry"
    },

    "jute": {
        "soil": "Alluvial soil",
        "water": "High",
        "season": "Kharif",
        "climate": "Warm and humid"
    },

    "kidneybeans": {
        "soil": "Well-drained loamy soil",
        "water": "Moderate",
        "season": "Kharif",
        "climate": "Warm"
    },

    "lentil": {
        "soil": "Loamy soil",
        "water": "Low to moderate",
        "season": "Rabi",
        "climate": "Cool and dry"
    },

    "maize": {
        "soil": "Well-drained loamy soil",
        "water": "Moderate",
        "season": "Kharif / Rabi",
        "climate": "Warm"
    },

    "mango": {
        "soil": "Deep, well-drained soil",
        "water": "Moderate",
        "season": "Year-round",
        "climate": "Tropical"
    },

    "mothbeans": {
        "soil": "Sandy soil",
        "water": "Low",
        "season": "Kharif",
        "climate": "Hot and dry"
    },

    "mungbean": {
        "soil": "Well-drained loamy soil",
        "water": "Low to moderate",
        "season": "Kharif / Summer",
        "climate": "Warm"
    },

    "muskmelon": {
        "soil": "Sandy loamy soil",
        "water": "Moderate",
        "season": "Summer",
        "climate": "Warm and dry"
    },

    "orange": {
        "soil": "Well-drained sandy loam",
        "water": "Moderate",
        "season": "Year-round",
        "climate": "Warm and subtropical"
    },

    "papaya": {
        "soil": "Well-drained sandy loam",
        "water": "Moderate",
        "season": "Year-round",
        "climate": "Warm and tropical"
    },

    "pigeonpeas": {
        "soil": "Well-drained loamy soil",
        "water": "Low to moderate",
        "season": "Kharif",
        "climate": "Warm and semi-arid"
    },

    "pomegranate": {
        "soil": "Well-drained loamy soil",
        "water": "Low to moderate",
        "season": "Year-round",
        "climate": "Hot and dry"
    },

    "rice": {
        "soil": "Clayey and loamy soil",
        "water": "High",
        "season": "Kharif",
        "climate": "Warm and humid"
    },

    "watermelon": {
        "soil": "Sandy loamy soil",
        "water": "Moderate",
        "season": "Summer",
        "climate": "Warm"
    }
}
# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("🌱 About the Project")

    st.write(
        "This AI-based system recommends suitable crops "
        "using soil nutrients and environmental conditions."
    )

    st.divider()

    st.subheader("🤖 Technology Used")

    st.write("• Python")
    st.write("• Machine Learning")
    st.write("• Random Forest")
    st.write("• Streamlit")
    st.write("• Pandas")

    st.divider()

    st.subheader("📊 Model Information")

    st.write("🤖 Model: Random Forest Classifier")
    st.write("🌾 Crop Classes: 22")
    st.write("📊 Test Accuracy: 99.32%")
    st.write("🔢 Input Parameters: 7")
# --------------------------------------------------
# TITLE
# --------------------------------------------------

# --------------------------------------------------
# PROJECT HEADER
# --------------------------------------------------

st.title("🌱 AI-Based Crop Recommendation System")

st.subheader("Smart Farming Through Artificial Intelligence 🤖🌾")

st.write(
    "Enter soil nutrients and environmental conditions to receive "
    "an AI-powered crop recommendation."
)

st.info(
    "💡 The system analyzes Nitrogen, Phosphorus, Potassium, "
    "Temperature, Humidity, pH and Rainfall."
)

st.divider()
st.divider()

# --------------------------------------------------
# CROP EXAMPLE SELECTION
# --------------------------------------------------

st.subheader("🌾 Choose an Example")

available_crops = sorted(
    dataset["label"].unique()
)

example_options = ["Custom"] + available_crops

selected_example = st.selectbox(
    "Select a crop example:",
    example_options
)
if st.button("🔄 Reset to Custom"):
    st.rerun()
# --------------------------------------------------
# LOAD REAL DATA FROM DATASET
# --------------------------------------------------

if selected_example != "Custom":

    crop_data = dataset[
        dataset["label"] == selected_example
    ]

    example_row = crop_data.sample(
        n=1,
        random_state=42
    ).iloc[0]

    N = float(example_row["N"])
    P = float(example_row["P"])
    K = float(example_row["K"])
    temperature = float(
        example_row["temperature"]
    )
    humidity = float(
        example_row["humidity"]
    )
    ph = float(example_row["ph"])
    rainfall = float(
        example_row["rainfall"]
    )

else:

    N = 90.0
    P = 42.0
    K = 43.0
    temperature = 20.0
    humidity = 80.0
    ph = 6.5
    rainfall = 200.0

# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------

st.subheader("🌾 Soil & Environmental Details")

col1, col2, col3 = st.columns(3)

with col1:

    N = st.number_input(
        "Nitrogen (N)",
        min_value=0.0,
        max_value=200.0,
        value=N
    )

    P = st.number_input(
        "Phosphorus (P)",
        min_value=0.0,
        max_value=200.0,
        value=P
    )

    K = st.number_input(
        "Potassium (K)",
        min_value=0.0,
        max_value=300.0,
        value=K
    )

with col2:

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=0.0,
        max_value=50.0,
        value=temperature
    )

    humidity = st.number_input(
        "Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=humidity
    )

with col3:

    ph = st.number_input(
        "pH Value",
        min_value=0.0,
        max_value=14.0,
        value=ph
    )

    rainfall = st.number_input(
        "Rainfall (mm)",
        min_value=0.0,
        max_value=500.0,
        value=rainfall
    )

st.divider()
# Prediction History
if "history" not in st.session_state:
    st.session_state.history = []
# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button(
    "🌾 Recommend Crop",
    use_container_width=True
):
     # --------------------------------------------------
# INPUT VALIDATION
# --------------------------------------------------

    if N < 0 or P < 0 or K < 0:
        st.error("❌ N, P and K values cannot be negative.")
        st.stop()

    if humidity < 0 or humidity > 100:
        st.error("❌ Humidity must be between 0 and 100%.")
        st.stop()

    if ph < 0 or ph > 14:
        st.error("❌ pH must be between 0 and 14.")
        st.stop()

    if temperature < 0 or temperature > 50:
        st.error("❌ Temperature should be between 0°C and 50°C.")
        st.stop()

    if rainfall < 0 or rainfall > 500:
        st.error("❌ Rainfall should be between 0 and 500 mm.")
        st.stop()
    input_data = pd.DataFrame([{

        "N": N,
        "P": P,
        "K": K,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall

    }])

    # Predict probabilities
    probabilities = model.predict_proba(
        input_data
    )[0]

    crop_names = model.classes_

    # Sort predictions
    results = sorted(
        zip(crop_names, probabilities),
        key=lambda x: x[1],
        reverse=True
    )

    # --------------------------------------------------
    # BEST CROP
    # --------------------------------------------------

    best_crop = results[0][0]

    best_probability = (
        results[0][1] * 100
    )
     # Save this prediction
    st.session_state.history.append({
        "Recommended Crop": best_crop.upper(),
        "Confidence": f"{best_probability:.2f}%"
    })

    st.success(
        f"🌱 Recommended Crop: "
        f"**{best_crop.upper()}**"
    )

    st.metric(
        "Prediction Confidence",
        f"{best_probability:.2f}%"
    )
    # --------------------------------------------------
# SUITABILITY LEVEL
# --------------------------------------------------

    if best_probability >= 80:
        suitability = "🟢 Highly Suitable"
        message = "The entered conditions strongly match this crop."

    elif best_probability >= 50:
        suitability = "🟡 Moderately Suitable"
        message = "The crop may be suitable, but compare the other recommendations."

    else:
        suitability = "🔴 Low Suitability"
        message = "Consider checking the top recommended crops."

    st.subheader("🌱 Crop Suitability")

    st.write(f"### {suitability}")

    st.write(message)
    # --------------------------------------------------
# DOWNLOAD CURRENT PREDICTION
# --------------------------------------------------

    st.subheader("📥 Download Prediction Report")

    report_data = pd.DataFrame([{
        "Nitrogen (N)": N,
        "Phosphorus (P)": P,
        "Potassium (K)": K,
        "Temperature (°C)": temperature,
        "Humidity (%)": humidity,
        "pH": ph,
        "Rainfall (mm)": rainfall,
        "Recommended Crop": best_crop.upper(),
        "Confidence (%)": round(best_probability, 2),
        "Suitability": suitability
    }])

    csv_report = report_data.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="📥 Download Prediction Report",
        data=csv_report,
        file_name="crop_prediction_report.csv",
        mime="text/csv"
    )
    # Farmer-friendly recommendation
    st.subheader("💡 Farmer Recommendation")

    if best_probability >= 80:
        st.success(
            f"{best_crop.upper()} is highly suitable "
            "for the entered soil and environmental conditions."
        )

    elif best_probability >= 50:
        st.warning(
            f"{best_crop.upper()} is reasonably suitable. "
            "Consider checking local soil and weather conditions "
            "before cultivation."
        )

    else:
        st.info(
            f"{best_crop.upper()} has lower prediction confidence. "
            "Consider comparing the top recommended crops."
        )
    # --------------------------------------------------
    # TOP 3 CROPS
    # --------------------------------------------------

    st.subheader(
        "🏆 Top 3 Recommended Crops"
    )

    for i, (crop, probability) in enumerate(
        results[:3],
        start=1
    ):

        percentage = probability * 100

        st.write(
            f"**{i}. {crop.upper()} — "
            f"{percentage:.2f}%**"
        )

        st.progress(
            float(probability)
        )
        # --------------------------------------------------
# TOP 3 VISUAL CHART
# --------------------------------------------------

    chart_data = pd.DataFrame({
        "Crop": [crop.upper() for crop, probability in results[:3]],
        "Confidence": [
            probability * 100
            for crop, probability in results[:3]
        ]
    })

    st.subheader("📊 Crop Prediction Comparison")

    st.bar_chart(
        chart_data.set_index("Crop")
    )
    # --------------------------------------------------
    # CROP INFORMATION
    # --------------------------------------------------

    st.divider()

    st.subheader(
        f"📋 Information About "
        f"{best_crop.upper()}"
    )

    info = crop_info.get(
        best_crop.lower()
    )

    if info:

        info1, info2, info3, info4 = st.columns(4)

        with info1:
            st.write("🌍 **Suitable Soil**")
            st.write(info["soil"])

        with info2:
            st.write("💧 **Water Requirement**")
            st.write(info["water"])

        with info3:
            st.write("📅 **Growing Season**")
            st.write(info["season"])

        with info4:
            st.write("🌦️ **Climate**")
            st.write(info["climate"])
# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

if hasattr(model, "feature_importances_"):

    st.subheader("📊 Important Factors Used by AI")

    feature_names = [
        "Nitrogen (N)",
        "Phosphorus (P)",
        "Potassium (K)",
        "Temperature",
        "Humidity",
        "pH",
        "Rainfall"
    ]

    importance_df = pd.DataFrame({
        "Parameter": feature_names,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False
    )

    st.bar_chart(
        importance_df.set_index("Parameter")
    )
    # --------------------------------------------------
# WHY THIS CROP?
# --------------------------------------------------

    st.subheader("🔍 Why This Crop Was Recommended")

    st.write(
        f"The AI model selected **{best_crop.upper()}** because "
        "the combination of the entered soil and environmental "
        "conditions closely matches the conditions associated "
        "with this crop in the training dataset."
    )

    st.write(
        f"🌱 **Prediction confidence:** "
        f"{best_probability:.2f}%"
    )

    st.write(
        "📊 The recommendation is based on seven parameters: "
        "Nitrogen, Phosphorus, Potassium, Temperature, "
        "Humidity, pH and Rainfall."
    )
# --------------------------------------------------
# FOOTER
# --------------------------------------------------
# --------------------------------------------------
# PREDICTION HISTORY
# --------------------------------------------------

if st.session_state.history:

    st.divider()

    st.subheader("📊 Prediction History")

    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        history_df,
        use_container_width=True
    )

    if st.button("🗑️ Clear Prediction History"):

        st.session_state.history = []

        st.rerun()
st.divider()
# --------------------------------------------------
# PROJECT SUMMARY
# --------------------------------------------------

st.divider()

st.subheader("🌱 Project Summary")

summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

with summary_col1:
    st.metric("🌾 Crop Classes", "22")

with summary_col2:
    st.metric("🤖 Model", "Random Forest")

with summary_col3:
    st.metric("📊 Test Accuracy", "99.32%")

with summary_col4:
    st.metric("🔢 Input Parameters", "7")

st.write(
    "This system uses Machine Learning to recommend a suitable "
    "crop based on soil nutrients and environmental conditions."
)

st.caption(
    "AI-Based Crop Recommendation System | "
    "Machine Learning Project"
)
