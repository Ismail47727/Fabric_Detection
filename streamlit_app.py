

# #-------------
# import streamlit as st
# import cv2
# import numpy as np
# from PIL import Image
# from ultralytics import YOLO
# import tempfile
# import os

# # Make sure the model is loaded only once
# @st.cache_resource
# def load_model(model_path):
#     return YOLO(model_path)

# def main():
#     st.sidebar.header("Fabric Defect Detection\nMembers:\n\n21K-3118     Ismail Qayyum\n\n 21K-3010 \tHuzaifa Rajput \n\n21K-3079 \tRayyan Ahmed \n\n21K-3115 \tGhazanfar Adnan \n\n21K-3052 \tAbbass Altaf\n\n DevOps Project")
#     st.title("Real-time Fabric Defect Detection")
#     st.write("""This app allows you to upload a fabric image or use your webcam for real-time fabric defect detection using the YOLOv8 model.""")

#     # Load the model (You can either use a local model or a URL)
#     model_path = "best.pt"  # Adjust this if using a different location or model URL
#     model = load_model(model_path)

#     # Sidebar Options
#     option = st.sidebar.radio("Choose Input Method", ("Browse Image", "Real-time Video"))

#     if option == "Browse Image":
#         uploaded_file = st.file_uploader("Upload a Fabric Image", type=['jpg', 'jpeg', 'png'])
#         if uploaded_file:
#             if uploaded_file.type.startswith('image'):
#                 inference_images(uploaded_file, model)

#     elif option == "Real-time Video":
#         st.warning("Ensure your webcam is enabled!")
#         real_time_video(model)

# def inference_images(uploaded_file, model):
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", width=600)
    
#     # Perform inference on the uploaded image
#     predict = model.predict(image)
#     boxes = predict[0].boxes
#     plotted = predict[0].plot()[:, :, ::-1]

#     if len(boxes) == 0:
#         st.markdown("**No Detection**")

#     st.image(plotted, caption="Detected Image", width=600)

# def real_time_video(model):
#     # Start video capture
#     cap = cv2.VideoCapture(0)  # 0 for the default camera, adjust for external cameras

#     stframe = st.empty()  # Create a Streamlit container to hold video frames
#     stop_button = st.button("Stop Video")  # Button to stop the video stream

#     while cap.isOpened():
#         if stop_button:
#             cap.release()
#             stframe.empty()
#             break

#         ret, frame = cap.read()
#         if not ret:
#             st.error("Failed to capture video frame.")
#             break

#         # Perform inference on the current frame
#         results = model.predict(frame)
#         plotted_frame = results[0].plot()

#         # Display the result in real-time
#         stframe.image(plotted_frame, channels="BGR", use_column_width=True)

#     cap.release()

# if __name__ == '__main__':
#     main()










#---------
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os

# Make sure the model is loaded only once
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

def main():
    st.sidebar.header("Fabric Defect Detection\nMembers:\n\n21K-3118     Ismail Qayyum\n\n 21K-3010 \tHuzaifa Rajput \n\n21K-3079 \tRayyan Ahmed \n\n21K-3115 \tGhazanfar Adnan \n\n21K-3052 \tAbbass Altaf\n\n DevOps Project")
    st.title("Real-time Fabric Defect Detection")
    st.write("""This app allows you to upload a fabric image, video, or use your webcam for real-time fabric defect detection using the YOLOv8 model.""")

    # Load the model (You can either use a local model or a URL)
    model_path = "best2.pt"  # Adjust this if using a different location or model URL
    model = load_model(model_path)

    # Sidebar Options
    option = st.sidebar.radio("Choose Input Method", ("Browse Image", "Browse Video", "Real-time Video"))

    if option == "Browse Image":
        uploaded_file = st.file_uploader("Upload a Fabric Image", type=['jpg', 'jpeg', 'png'])
        if uploaded_file:
            if uploaded_file.type.startswith('image'):
                inference_images(uploaded_file, model)

    elif option == "Browse Video":
        uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi"])
        if uploaded_video:
            process_video(uploaded_video, model)

    elif option == "Real-time Video":
        st.warning("Ensure your webcam is enabled!")
        real_time_video(model)

def inference_images(uploaded_file, model):
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=600)
    
    # Perform inference on the uploaded image
    predict = model.predict(image)
    boxes = predict[0].boxes
    plotted = predict[0].plot()[:, :, ::-1]

    if len(boxes) == 0:
        st.markdown("**No Detection**")

    st.image(plotted, caption="Detected Image", width=600)

def process_video(uploaded_video, model):
    # Create a temporary file for the uploaded video
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_video.read())
        video_path = temp_file.name

    # Open the video file using OpenCV
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()  # Create a Streamlit container to hold video frames

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform inference on the current frame
        results = model.predict(frame)
        plotted_frame = results[0].plot()

        # Display the result in real-time
        stframe.image(plotted_frame, channels="BGR", use_column_width=True)

    cap.release()

def real_time_video(model):
    # Start video capture
    cap = cv2.VideoCapture(0)  # 0 for the default camera, adjust for external cameras

    stframe = st.empty()  # Create a Streamlit container to hold video frames
    stop_button = st.button("Stop Video")  # Button to stop the video stream

    while cap.isOpened():
        if stop_button:
            cap.release()
            stframe.empty()
            break

        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video frame.")
            break

        # Perform inference on the current frame
        results = model.predict(frame)
        plotted_frame = results[0].plot()

        # Display the result in real-time
        stframe.image(plotted_frame, channels="BGR", use_column_width=True)

    cap.release()

if __name__ == '__main__':
    main()


