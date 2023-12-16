import streamlit as st
import cv2
import numpy as np
from skimage import measure
from PIL import Image

# Function to process the image and count colonies
def count_colonies(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    # Thresholding the image to a binary image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Perform a series of erosions and dilations to remove any small blobs of noise from the image
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)

    # Perform connected component analysis on the thresholded image
    labels = measure.label(thresh, connectivity=2, background=0)

    # Find the contours of the masked regions
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Draw red circles around the colonies
    for c in cnts:
        ((x, y), r) = cv2.minEnclosingCircle(c)
        cv2.circle(np.array(image), (int(x), int(y)), int(r), (255, 0, 0), 2)
    
    return image, len(cnts)

# Streamlit application
st.title('Bacterial Colony Counter')

# File uploader allows user to add their own image
uploaded_file = st.file_uploader("Upload an image of bacterial colonies", type=["jpg", "jpeg", "png"])

# If the user uploaded an image, display it and process it
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Process the image and count the colonies
    processed_image, colony_count = count_colonies(image)
    
    # Display the processed image and the count of colonies
    st.image(processed_image, caption='Processed Image with Counted Colonies', use_column_width=True)
    st.write(f'Number of colonies detected: {colony_count}')

# Run this with `streamlit run your_script.py` in your terminal
