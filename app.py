import streamlit as st
import pandas as pd
import joblib
from src.inference.predict import create_feature_dataframe

@st.cache_resource
def load_model():
    """Loads the trained model pipeline from the file."""
    try:
        return joblib.load('models/model.joblib')
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'models/model.joblib' is present.")
    except Exception as e:
        st.error(f"Error loading the model: {e}")
    return None

# --- Load the Model ---
model_pipeline = load_model()

# --- App Title ---
st.title("AI-Based Interior Renovation Cost Estimator")
st.markdown("Enter your project details to get an AI-powered cost estimate.")
st.divider()

if model_pipeline:
    # --- Create UI for User Input ---
    st.sidebar.header("Project Inputs")

    # Grouping inputs in the sidebar for a cleaner main page
    city = st.sidebar.selectbox("City", ["Mumbai", "Delhi", "Bangalore", "Chennai", "Pune", "Hyderabad", "Kanpur", "Indore", "Jaipur", "Kolkata", "Vadodara"])
    room_type = st.sidebar.selectbox("Room Type", ["Bedroom", "Living Room", "Kitchen", "Bathroom", "Dining", "Kids Room"])
    area_sqft = st.sidebar.number_input("Area (in Sq. Ft.)", min_value=50, max_value=2000, value=150, step=10)
    renovation_level = st.sidebar.selectbox("Renovation Level", ["Basic", "Mid", "Luxury"])
    has_electrical = st.sidebar.toggle("Include Electrical Work?", value=True)
    
    st.sidebar.divider()
    
    paint_quality = st.sidebar.selectbox("Paint Quality", ["Economy", "Standard", "Premium"])
    floor_type = st.sidebar.selectbox("Floor Type", ["Vitrified_Tile", "Ceramic_Tile", "Engineered_Wood"])
    floor_quality = st.sidebar.selectbox("Floor Quality", ["Economy", "Standard", "Premium"])
    ceiling_type = st.sidebar.selectbox("False Ceiling Type", ["None", "POP", "Gypsum"])
    ceiling_quality = st.sidebar.selectbox("Ceiling Quality", ["Economy", "Standard", "Premium"])
    
    st.sidebar.divider()
    
    furniture_level = st.sidebar.selectbox("Furniture Level", ["None", "Basic", "Standard"])
    kitchen_package = st.sidebar.selectbox("Kitchen Package", ["None", "Basic_S", "Basic_L", "Premium"])
    bathroom_package = st.sidebar.selectbox("Bathroom Package", ["None", "Basic", "Premium"])


    # --- Prediction Logic ---
    st.header("Cost Estimate")
    if st.button("Calculate Estimated Cost", type="primary"):
        
        # 1. Collect raw inputs into a dictionary
        raw_input_data = {
            'City': city,
            'Room_Type': room_type,
            'Area_Sqft': area_sqft,
            'Renovation_Level': renovation_level,
            'Paint_Quality': paint_quality,
            'Floor_Type': floor_type,
            'Floor_Quality': floor_quality,
            'Ceiling_Type': ceiling_type,
            'Ceiling_Quality': ceiling_quality,
            'Has_Electrical': has_electrical,
            'Furniture_Level': furniture_level,
            'Kitchen_Package': kitchen_package,
            'Bathroom_Package': bathroom_package,
            'City_Tier': 1 # Using a default, a real app might have a city->tier map
        }
        
        try:
            # 2. Run the full feature engineering pipeline
            st.write("⚙️ Generating features and running prediction...")
            feature_df = create_feature_dataframe(raw_input_data)
            
            # 3. Make a prediction using the complete feature dataframe
            prediction = model_pipeline.predict(feature_df)
            predicted_cost = prediction[0]
            
            # 4. Display the result
            st.metric(label="**Estimated Project Cost**", value=f"₹ {predicted_cost:,.0f}")
            st.success("Prediction successful!")
            st.info("Note: This is an AI-generated estimate based on the provided inputs. Actual costs may vary.")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.exception(e) # Provides a more detailed traceback for debugging

else:
    st.warning("Model could not be loaded. Please check the logs.")