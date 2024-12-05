# from PIL import Image
# import streamlit as st
# import cv2
# import numpy as np
# from ultralytics import YOLO
# import requests
# st.write('heelo world')


# # Function to download the model
# def download_model():
#     url = 'https://github.com/Ismail47727/Fabric_Detection/raw/main/best.pt'
#     response = requests.get(url)
#     with open('best.pt', 'wb') as f:
#         f.write(response.content)



# MODEL_DIR = 'best.pt'


# def main():
#     download_model()
#     # Load the YOLO model
#     model = YOLO(MODEL_DIR)

#     st.sidebar.header("Fabric Defect Detection using YOLOv8\nMembers:\n\n21K-3118\n\n 21K-3010\n\n21K-3079\n\n21K-3115\n\n21K-3052\n\n DevOps Project")
#     st.title("Real-time Fabric Defect Detection")
#     st.write("""
#     This app allows you to upload a fabric image or use your webcam for real-time fabric defect detection using the YOLOv8 model.
#     """)

# # # Sidebar Options
# # option = st.sidebar.radio("Choose Input Method", ("Browse Image", "Real-time Video"))

# # if option == "Browse Image":
# #     uploaded_file = st.file_uploader("Upload a Fabric Image", type=['jpg', 'jpeg', 'png'])
# #     if uploaded_file:
# #         if uploaded_file.type.startswith('image'):
# #             inference_images(uploaded_file, model)

# # # elif option == "Real-time Video":
# # #     st.warning("Ensure your webcam is enabled!")
# # #     real_time_video(model)

# # def inference_images(uploaded_file, model):
# #     image = Image.open(uploaded_file)
# #     # Perform inference on the uploaded image
# #     predict = model.predict(image)
# #     boxes = predict[0].boxes
# #     plotted = predict[0].plot()[:, :, ::-1]

# #     if len(boxes) == 0:
# #         st.markdown("**No Detection**")

# #     st.image(plotted, caption="Detected Image", width=600)

# # # def real_time_video(model):
# # #     # Start video capture
# # #     cap = cv2.VideoCapture(0)  # 0 for the default camera, adjust for external cameras

# # #     stframe = st.empty()  # Create a Streamlit container to hold video frames
# # #     st.button("Stop", key="stop_button")  # Placeholder for stopping (not implemented fully)

# # #     while cap.isOpened():
# # #         ret, frame = cap.read()
# # #         if not ret:
# # #             st.error("Failed to capture video frame.")
# # #             break

# # #         # Perform inference on the current frame
# # #         results = model.predict(frame)
# # #         plotted_frame = results[0].plot()

# # #         # Display the result in real-time
# # #         stframe.image(plotted_frame, channels="BGR", use_column_width=True)

# # #     cap.release()

# if __name__ == '__main__':
#     main()








# import requests
# import torch
# from ultralytics import YOLO
# import streamlit as st
# from PIL import Image
# import io
# import os

# # Download the model weights if not already downloaded
# def download_model():
#     url = 'https://github.com/Ismail47727/Fabric_Detection/raw/main/best.pt'  # Model file URL
#     model_path = 'best.pt'
#     if not os.path.exists(model_path):  # Check if the model already exists
#         st.write("Downloading the model... This might take a while.")
#         response = requests.get(url)
#         with open(model_path, 'wb') as f:
#             f.write(response.content)
#         st.write("Model downloaded successfully!")
#     return model_path

# # Load YOLO model
# def load_model(model_path):
#     model = YOLO(model_path)  # Load the model
#     return model

# # Run predictions on the uploaded image
# def run_inference(model, image):
#     # Convert image to bytes for processing
#     img = Image.open(image)
#     results = model(img)  # Make predictions with the model
#     return results

# def main():
#     st.sidebar.header("Fabric Defect Detection with YOLOv8")
#     st.title("Fabric Defect Detection App")

#     # Download the model at runtime if not already downloaded
#     model_path = download_model()

#     # Load the YOLO model
#     model = load_model(model_path)

#     st.write("Upload an image to detect fabric defects...")

#     uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

#     if uploaded_image is not None:
#         # Show uploaded image
#         st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

#         # Run inference on the image
#         st.write("Classifying...")

#         # Run the model on the uploaded image and show results
#         results = run_inference(model, uploaded_image)
        
#         # Show results (you can modify this based on how results are returned)
#         st.write(f"Predicted Labels: {results.names}")
#         st.write(f"Prediction Results: {results.xywh[0]}")  # This shows the prediction results

#         # You can visualize the output like bounding boxes here:
#         img_with_boxes = results.render()  # Add bounding boxes to image
#         st.image(img_with_boxes[0], caption="Detection Result", use_column_width=True)

# # Run the app
# if __name__ == "__main__":
#     main()












#-------------
import os
import requests
import torch
from ultralytics import YOLO
import streamlit as st
from PIL import Image
import io

# Download the model weights if not already downloaded
def download_model():
    url = 'https://github.com/Ismail47727/Fabric_Detection/raw/main/best.pt'  # Model file URL
    model_path = 'best.pt'
    if not os.path.exists(model_path):  # Check if the model already exists
        st.write("Downloading the model... This might take a while.")
        response = requests.get(url)
        with open(model_path, 'wb') as f:
            f.write(response.content)
        st.write("Model downloaded successfully!")
    return model_path

# Load YOLO model
def load_model(model_path):
    model = YOLO(model_path)  # Load the model
    return model

# Run predictions on the uploaded image
def run_inference(model, image):
    # Convert image to bytes for processing
    img = Image.open(image)
    results = model(img)  # Make predictions with the model
    return results

def main():
    st.sidebar.header("Fabric Defect Detection with YOLOv8")
    st.title("Fabric Defect Detection App")

    # Download the model at runtime if not already downloaded
    model_path = download_model()

    # Load the YOLO model
    model = load_model(model_path)

    st.write("Upload an image to detect fabric defects...")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Show uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Run inference on the image
        st.write("Classifying...")

        # Run the model on the uploaded image and show results
        results = run_inference(model, uploaded_image)
        
        # Accessing the predicted class names and bounding boxes
        predicted_labels = results.names  # The class names
        predicted_coordinates = results.xywh[0]  # Bounding box coordinates

        # Display predicted labels
        st.write(f"Predicted Labels: {predicted_labels}")
        st.write(f"Prediction Coordinates: {predicted_coordinates}")  # Bounding boxes

        # You can visualize the output like bounding boxes here:
        img_with_boxes = results.render()  # Add bounding boxes to image
        st.image(img_with_boxes[0], caption="Detection Result", use_column_width=True)

# Run the app
if __name__ == "__main__":
    main()

