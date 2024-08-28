import streamlit as st
from keras.preprocessing import image
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from PIL import Image
# Load the trained model
model = load_model('C:/Users/HP/Desktop/proj/1breast_cancer_classification_model.hdf5')  # Load your trained model here

# Define a function to classify the image
def classify_image(image_file, threshold):
    img = image.load_img(image_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.  # Rescale to [0, 1]
    prediction = model.predict(img_array)
    return prediction[0]

# Streamlit web application
def main():
    html_temp="""
   <div style="background-color:#025246;padding:10px">
   <h2 style="color:white;text-align:center;">Breast Cancer Classification</h2
</div>

"""

 
    st.markdown(html_temp,unsafe_allow_html=True)
    
    # Login section
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if username == "abc" and password == "123":
        st.sidebar.success("Logged In as {}".format(username))
        
        # Patient's information input
        patient_name = st.text_input("Patient's Name", placeholder="Enter patient's name")
        patient_age = st.number_input("Patient's Age", min_value=0, max_value=150, step=1)
        patient_gender = st.selectbox("Patient's Gender", ['Male', 'Female'])

        # Upload image
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg","png"])

        # Add a slider widget to adjust the threshold
        threshold = st.slider("Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01) 
        malig_html="""
          <div style="background-color:#F08080;padding:10px">
          <h2 style="color:black;text-align:center;">The image classified as malignant </h2
          
          </div>
          """ 
        benign_html="""
            <div style="background-color:#F4D03F;padding:10px">
            <h2 style="color:black;text-align:center;">The image classified as Benign </h2
            
            </div>
            """
          
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            st.write("")
            st.write("Classifying...")
            prediction = classify_image(uploaded_image, threshold)
            if prediction > threshold:
                st.write("Patient:", patient_name)
                st.markdown(malig_html ,unsafe_allow_html=True)
            
                
              
                # Store patient's information and result in history
                history_df = pd.DataFrame({'Patient Name': [patient_name],
                                           'Patient Age': [patient_age],
                                           'Patient Gender': [patient_gender],
                                           'Result': ['Malignant']})
            else:
                st.write("Patient:", patient_name)
                st.markdown(benign_html ,unsafe_allow_html=True)
              
                # Store patient's information and result in history
                history_df = pd.DataFrame({'Patient Name': [patient_name],
                                           'Patient Age': [patient_age],
                                           'Patient Gender': [patient_gender],
                                           'Result': ['Benign']})

        # Display history button
    if 'history_df' in locals():
            if st.button('View Classification History'):
                st.write(history_df)
    elif username != "" or password != "":
        st.sidebar.error("Incorrect username or password. Please try again.")

if __name__ == "__main__":
    main()
