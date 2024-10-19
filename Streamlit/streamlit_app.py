import numpy as np 
import pandas as pd 
import streamlit as st
import joblib
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score

df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# Set the title of the app
st.title("Stroke Prediction")

option = st.sidebar.selectbox("Choose an option", ["Streamlit_App", "Stroke_Prediction","Meet Our Team"])

if option == "Streamlit_App":
    tabs = st.tabs(["Dataset Overview", "Data Preparation"])

    with tabs[0]:
        st.markdown('''
            ### **About the Dataset:**
            This dataset contains comprehensive information on various medical and demographic attributes that help in predicting the likelihood of stroke occurrences in patients. Sourced from WHO data, it encompasses a wide range of health metrics and patient details, providing valuable insights for predictive analysis.

            #### **Key Features:**
            - **Medical Attributes:** Includes information such as blood pressure levels, glucose levels, and body mass index (BMI).
            - **Demographic Attributes:** Age, gender, and smoking status of the patients are recorded.
            - **Health Metrics:** History of heart disease, hypertension, and other critical health indicators.

            The dataset offers a balanced mix of attributes that can be used to train machine learning models to accurately predict stroke risks.

            #### **DataFrame Overview:**
        ''')

        st.dataframe(df, use_container_width=True) 

        Customers_Sheets = pd.DataFrame({
        'Column':['ID', 'Gender', 'Age', 'Hypertension', 'Heart Disease', 'Ever Married', 'Work type', 'Residence Type', 'Avg Glucose Level','BMI', 'Smoking Status', 'Stroke'],
        'Description':['A unique identifier for each patient',
                    'The patien gender, which could be "Male" or "Female" ',
                    'The age of the patient in years, a critical factor in assessing stroke risk',
                    'Indicates whether the patient has hyperension(0=No,1=Yes)',
                    'Shows whether the patient has a history of heart disease(0=No,1=Yes)',
                    'Denotes the patient marital status as "Yes" or "No"',
                    'The type of employment the patient engages in, categorized as "Private", "Self-employed", "Govt_job" or "Children" ',
                    'The patients living environment, classified as "Urban" or "Rural"',
                    'The patients average blood glucose level',
                    'Body mass index, representing the patients weight relative to height',
                    'Classifies the patient as "formerly smoked", "never smoked", "smokes" or "Unknown"',
                    'The target variable, where 1 indicates the patient has experienced a  stroke and 0 indicates no stroke',
                    
                    ]})
        
        with st.expander("Meta Data"):
            st.dataframe(Customers_Sheets)


        st.header("Dashboard")

        images = [
         "Dashboard/DEPI_FP-1.jpg",
         "Dashboard/DEPI_FP-2.jpg",
         "Dashboard/DEPI_FP-3.jpg",
         "Dashboard/DEPI_FP-4.jpg"
    
        ]

        # Custom CSS for modern button styles
        st.markdown("""
            <style>
            .btn-style {
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                transition: background-color 0.3s ease-in-out;
            }
            .btn-style:hover {
                background-color: #45a049;
            }
            .slider-indicators {
                text-align: center;
                margin-top: 10px;
            }
            .slider-indicators span {
                height: 15px;
                width: 15px;
                margin: 0 5px;
                display: inline-block;
                background-color: #bbb;
                border-radius: 50%;
            }
            .slider-indicators .active {
                background-color: #717171;
            }
            </style>
        """, unsafe_allow_html=True)

        # Initialize session state for image index and auto-slide timer
        if "carousel_index" not in st.session_state:
            st.session_state.carousel_index = 0


        # Display the current image
        st.image(images[st.session_state.carousel_index], width=700)

        # Navigation buttons (Previous/Next)
        prev, _, next = st.columns([1, 10, 1])

        # Handle the previous button click
        if prev.button("◀", key="prev", help="Previous image", type="primary"):
            st.session_state.carousel_index = (st.session_state.carousel_index - 1) % len(images)

        # Handle the next button click
        if next.button("▶", key="next", help="Next image", type="primary"):
            st.session_state.carousel_index = (st.session_state.carousel_index + 1) % len(images)

        # Extra description or metadata
        st.write(f"Image {st.session_state.carousel_index + 1} of {len(images)}")
        
        st.write("---") 
        # Expander for Dashboard Information
        with st.expander("Dashboard"):
           st.subheader("Dashboard Overview")
           st.markdown("[View Dashboard](https://app.powerbi.com/links/Vg4be6DfcC?ctid=4b6778d7-95aa-4bc6-9b34-f2e341791192&pbi_source=linkShare)")
        
        # Expander for Presentation
        with st.expander("Presentation"):
            st.subheader("Presentation Overview")
            st.markdown("[View Presentation](https://drive.google.com/file/d/1spQG2dCYfxpPuxM8Si3nLdZVNqJlObEM/view?usp=sharing)")


        # Content for Dashboard tab
    with tabs[1]:

        st.markdown('''
            ### **Data Preparation:**

            #### **1. Problem: Invalid Data**
            - **Gender:** The 'Gender' column contained inconsistent values such as 'F', 'Femal', 'M', and 'O'.
                - **Solution:** Standardized all values to 'Female', 'Male', and 'Other' to ensure consistency across the dataset.

            - **Unrealistic Age Values:** Some age values, such as 174, were clearly invalid and impacted data quality.
                - **Solution:** Deleted the rows containing these outlier values to maintain the accuracy of the dataset.

            #### **2. Problem: Check for Duplicates**
            - **Duplicate Records:** The dataset was scanned for duplicate patient entries based on unique identifiers.
                - **Solution:** No duplicate records were found, ensuring data accuracy.

            #### **3. Problem: Missing Values**
            - **BMI and Smoking Status:** Missing values were identified in the "bmi" and "smoking_status" columns, which are crucial for predicting stroke risk.
                - **Solution:** Missing values in the "bmi" column were imputed using the median value of BMI to avoid skewing the data, while "Unknown" values in the "smoking_status" column were retained but carefully considered in the analysis.

            ---
        ''')


    
    
elif option == "Stroke_Prediction":

    st.divider()
    st.markdown('''
    ### **Objective:**
    The primary objective of this project is to develop and evaluate machine learning models that accurately predict stroke occurrences in patients.

    **Key Focus:**
    - **Minimizing False Negatives (FN):** The goal is to reduce cases where a patient has a stroke, but the model incorrectly predicts no stroke, as such errors could lead to missed treatment opportunities and serious health consequences.
    - **Maximizing True Positives (TP):** By optimizing the model for high recall, it ensures that we capture as many actual stroke cases as possible, leading to timely interventions.

    ---

    ### **How Our Model Could Be Used:**

    #### **1. False Negative Avoidance:**
    - The model's focus on reducing false negatives makes it ideal for critical healthcare settings such as emergency rooms, hospitals, or clinics.
    - It helps in flagging potential stroke patients, even with mild symptoms, ensuring timely attention and reducing the risk of missed diagnoses.

    #### **2. Targeted Health Interventions:**
    - By accurately identifying stroke-prone individuals, healthcare providers can take preventive measures.
    - Interventions could include targeted health campaigns focused on smoking cessation, improving diet, or managing chronic diseases.

    #### **3. Proactive Care Programs:**
    - Healthcare systems can use the model to allocate resources efficiently, focusing on high-risk individuals for proactive care.
    - This approach reduces long-term healthcare costs by preventing strokes rather than treating them after they occur.

    #### **4. Health Apps:**
    - A health app powered by the model could allow patients to input their health data and receive personalized advice on their stroke risk.
    - This empowers individuals to take preventive actions based on their own health profiles.

    ---
    ''')


    # تحميل النموذج
    model = joblib.load('ML_Model_For_Stroke_predection.pkl')

    # تعيين عنوان التطبيق
    st.title('Machine Learning Model Predictor')

    # إنشاء حقول إدخال للميزات
    gender = st.selectbox('Gender', ['Male', 'Female'])
    hypertension = st.selectbox('Hypertension', [0, 1])
    heart_disease = st.selectbox('Heart Disease', [0, 1])
    ever_married = st.selectbox('Ever Married', ['Yes', 'No'])
    work_type = st.selectbox('Work Type', ['Private', 'Self-employed', 'Govt_job'])
    residence_type = st.selectbox('Residence Type', ['Urban', 'Rural'])
    avg_glucose_level = st.number_input('Average Glucose Level', value=0.0, format="%.1f")
    bmi = st.number_input('BMI', value=0.0, format="%.1f")
    smoking_status = st.selectbox('Smoking Status', ['formerly smoked', 'never smoked', 'smokes'])

    # ترميز الفئة العمرية
    age = st.selectbox('Age', ['Child: 0-18', 'Adult: 19-45', 'Senior: 46-60', 'Elderly: 61-100'])

    # جمع البيانات المدخلة في DataFrame
    input_data = pd.DataFrame([[str(gender), int(hypertension), int(heart_disease), str(ever_married),
                             str(work_type), str(residence_type), float(avg_glucose_level),
                             float(bmi), str(smoking_status), str(age)]],
                          columns=['gender', 'hypertension', 'heart_disease', 'ever_married',
                                   'work_type', 'Residence_type', 'avg_glucose_level',
                                   'bmi', 'smoking_status', 'age'])

    # طباعة بيانات الإدخال لأغراض التصحيح
    st.write("Input Data:", input_data)

    if st.button('Predict'):
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)  # Get probabilities
        stroke_prob = prediction_proba[0][1]  # Probability of having a stroke

        # Display the results
        st.write("Prediction:", "Stroke" if prediction[0] == 1 else "No Stroke")
        accuracy=stroke_prob * 100 
        st.title("The percentage of your chance of having a stroke")

        # إنشاء شريط الـ Gauge باستخدام Plotly
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=accuracy,
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "green"},
                'bgcolor': "white",
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 100], 'color': "lightgreen"}
                ],
            'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': accuracy}
             }
        ))

    # عرض الرسم في Streamlit
        st.plotly_chart(fig)



elif option == "Meet Our Team":   
    st.title('🌟 Meet Our Team')

    st.divider()

    # LinkedIn icon URL
    linkedin_icon = "https://img.icons8.com/?size=100&id=8808&format=png&color=FFFFFF"

    st.markdown("<h3>Our Team Members</h3>", unsafe_allow_html=True)
    team_members = [
        {"name": "Mahmoud Ahmed", "linkedin": "https://www.linkedin.com/in/mahmoud-ahmed-22505527a"},
        {"name": "Mohamed Fouda", "linkedin": "https://www.linkedin.com/in/mohamed-foda-1932602b1?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app"},
        {"name": "Nada Mohamed", "linkedin": "http://linkedin.com/in/nada-alfeky-3397b328a"},
        {"name": "Kamal Khaled", "linkedin": "https://www.linkedin.com/in/kamal-khaled-2a9096282?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app"},
        {"name": "Weam Taie", "linkedin": "https://www.linkedin.com/in/weam-taie-548535218/"},
        {"name": "Rehab Yehia", "linkedin": "https://www.linkedin.com/in/rehabyehia/"}
    ]

    for member in team_members:
        st.markdown(f"""
            <style>
                .hover-div {{
                    padding: 10px;
                    border-radius: 10px;
                    background-color: #2c413c;
                    margin-bottom: 10px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    transition: background-color 0.3s ease, box-shadow 0.3s ease;
                 }}
                .hover-div:hover {{
                    background-color: #1e7460; /* Slightly lighter background color on hover */
                    box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.2); /* Adds a shadow on hover */
                }}
                .linkedin-icon {{
                    width: 35px; /* Bigger icon size */
                    vertical-align: middle;
                }}
            </style>
            <div class="hover-div">
                <h4 style="margin-left: 15px; color: white;">{member['name']}</h4>
                <a href="{member['linkedin']}" target="_blank" style="margin-right: 25px;">
                    <img src="{linkedin_icon}" class="linkedin-icon"/>
                </a>
             </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.markdown("""
        <h3 style='color: #386641;'>We Value Your Feedback!</h3>
        <p style='font-size: 18px;'>Thank you for visiting our project page! We hope you enjoyed exploring our work. Your feedback is important to us, and we'd love to hear your thoughts, suggestions, or any questions you may have.</p>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("""
        <p style='font-size: 18px;'>A special thanks to <strong>Eng.Islam Adel</strong> for your valuable mentorship and supervision, which has been instrumental in our growth and success.</p>
    """, unsafe_allow_html=True)
  
