import openvino as ov
import cv2
import numpy as np
import streamlit as st
import logic

# Load models
core = ov.Core()
model_face = core.read_model(model='models/face-detection-adas-0001.xml')
compiled_model_face = core.compile_model(model=model_face, device_name="CPU")

input_layer_face = compiled_model_face.input(0)
output_layer_face = compiled_model_face.output(0)

model_emo = core.read_model(model='models/emotions-recognition-retail-0003.xml')
compiled_model_emo = core.compile_model(model=model_emo, device_name="CPU")

input_layer_emo = compiled_model_emo.input(0)
output_layer_emo = compiled_model_emo.output(0)

model_ag = core.read_model(model='models/age-gender-recognition-retail-0013.xml')
compiled_model_ag = core.compile_model(model=model_ag, device_name="CPU")

input_layer_ag = compiled_model_ag.input(0)
output_layer_ag = compiled_model_ag.output

# Standard confidence threshold
CONFIDENCE_THRESHOLD = 0.2

def find_faceboxes(image, results, confidence_threshold):
    results = results.squeeze()
    scores = results[:, 2]
    boxes = results[:, -4:]
    face_boxes = boxes[scores >= confidence_threshold]
    scores = scores[scores >= confidence_threshold]

    image_h, image_w, image_channels = image.shape
    face_boxes = face_boxes * np.array([image_w, image_h, image_w, image_h])
    face_boxes = face_boxes.astype(np.int64)

    return face_boxes, scores

def draw_age_gender_emotion(face_boxes, image):
    EMOTION_NAMES = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Anger']
    show_image = image.copy()
    recommendations = []  # Store recommendations to display later

    for i in range(len(face_boxes)):
        xmin, ymin, xmax, ymax = face_boxes[i]
        face = image[ymin:ymax, xmin:xmax]

        # --- Emotion ---
        input_image = logic.preprocess(face, input_layer_emo)
        results_emo = compiled_model_emo([input_image])[output_layer_emo]
        results_emo = results_emo.squeeze()
        index = np.argmax(results_emo)
        emotion_label = EMOTION_NAMES[index]

        # --- Age and Gender ---
        input_image_ag = logic.preprocess(face, input_layer_ag)
        results_ag = compiled_model_ag([input_image_ag])
        age = int(np.squeeze(results_ag[1]) * 100)
        gender = np.squeeze(results_ag[0])
        
        if (gender[0] > 0.65):
            gender = "Female"
        elif (gender[1] >= 0.55):
            gender = "Male"
        else:
            gender = "Unknown"

        # Get recommendations
        recs = logic.recommend_menu(age, gender, emotion_label)
        recommendations.append((gender, age, emotion_label, recs))

        # Draw rectangle around the face
        cv2.rectangle(show_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    return show_image, recommendations

def predict_image(image, conf_threshold):
    input_image = logic.preprocess(image, input_layer_face)
    results = compiled_model_face([input_image])[output_layer_face]
    face_boxes, scores = find_faceboxes(image, results, conf_threshold)
    visualize_image, recommendations = draw_age_gender_emotion(face_boxes, image)
    return visualize_image, recommendations

# Streamlit app setup
st.set_page_config(
    page_title="Swensen's Menu Recommendation Kiosk 👨‍🍳",
    page_icon="👨‍🍳",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("Swensen's Menu Recommendation Kiosk 👨‍🍳")

# Use st.camera_input to capture photo
image_file = st.camera_input("Capture Photo")

if image_file is not None:
    # Convert the image to an OpenCV format
    image = np.array(cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR))

    # Run prediction
    visualized_image, recommendations = predict_image(image, CONFIDENCE_THRESHOLD)
    st.image(visualized_image, channels="BGR")

    # Display recommendations below the image
    if recommendations:
        for gender, age, emotion, recs in recommendations:
            st.write(f"Gender: {gender}")
            st.write(f"Age: {age}")
            st.write(f"Emotion: {emotion}")
            st.write(f"Recommendations: {', '.join(recs)}")
    else:
        st.write("No faces detected.")