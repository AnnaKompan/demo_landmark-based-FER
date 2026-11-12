# https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker?tab=Python
# class_names = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
# import json
# with open("class_names.json", "w") as f:
#     json.dump(class_names, f)

# 1. Import libraries, load model, scaler and landmarker
import json
import time
import joblib
import tensorflow as tf
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = 'mlp_model.h5'
LANDMARKER_PATH = "face_landmarker.task"
SCALER_PATH = 'scaler.pkl'

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)

print("model input shape: ", model.input_shape)
print("model output shape: ", model.output_shape)

# 2. load class names in order trained on, check if model output shape == num of classes

CLASS_NAMES_PATH = 'class_names.json'

with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = json.load(f)

print("Number of classes: ", len(class_names))

if (model.output_shape[-1] != len(class_names)):
    print("Model output is: ", model.output_shape[-1], " and not equal to ", len(class_names), " number of classes")

# 3. Initializr MP landmarker

BASE_OPTIONS = python.BaseOptions(model_asset_path=LANDMARKER_PATH)
IMG_OPTIONS = vision.FaceLandmarkerOptions(base_options=BASE_OPTIONS,
                                       output_face_blendshapes=True,
                                       running_mode=vision.RunningMode.IMAGE,
                                       num_faces=1)
VIDEO_OPTIONS = vision.FaceLandmarkerOptions(base_options=BASE_OPTIONS,
                                             running_mode=vision.RunningMode.VIDEO,
                                             num_faces=1)


# 4. Extract landmarks

def extract_features(features):
    features_arr = []
    for feat in features:
        features_arr.extend([
            feat.x,
            feat.y,
            feat.z
        ])
    features_arr = np.array(features_arr, dtype=np.float32)
    # add batch dims (1D to 2D)  => 1434 to (1, 1434) for MLP model
    features_arr = features_arr.reshape(1, -1)
    return features_arr

# 5. Preprocessing: StandardScaler
def preprocessing(features):
    presprocessed_feat = scaler.transform(features)
    return presprocessed_feat.astype(np.float32)

# 6. Predict emotions

def prediction(features):
    probabilities = model.predict(features, verbose=0)[0]
    top2_predicted_idx = np.argsort(probabilities)[-2:][::-1]
    top2_predictions = []
    for idx in top2_predicted_idx:
        emotion = class_names[idx]
        confidence = probabilities[idx] * 100
        top2_predictions.append((emotion, confidence))
    return top2_predictions, probabilities

# 7. Test Live
TEST_IMG = 'image.png'
OUTPUT_IMG_EMOTION = 'result_image.png'

def crop_face(frame, landmarks, scale=1.35):
    height, width, _ = frame.shape

    # 7.1 get face bbox
    xs = np.array([p.x for p in landmarks]) * width
    ys = np.array([p.y for p in landmarks]) * height
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()

    # square crop with padding
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    side = max(int(max(max_x - min_x, max_y-min_y)*scale),2)
    x1 = int(round(center_x - side/2))
    y1 = int(round(center_y - side/2))
    face_crop = np.zeros((side,side,3),dtype=frame.dtype)

    # coordinates inside original framw
    sx1= max(0,x1)
    sy1= max(0,y1)
    sx2 = min(width, x1+side)
    sy2 = min(height, y1+side)

    # coordinates inside square crop
    dx1 = sx1-x1
    dy1 = sy1-y1
    dx2 = dx1 + (sx2-sx1)
    dy2 = dy1 + (sy2-sy1)

    face_crop[dy1:dy2, dx1:dx2] = frame[sy1:sy2, sx1:sx2]
    # visible bbox
    bbox = (sx1, sy1, sx2, sy2)
    return face_crop, bbox

def run_webcam():
    cap = cv2.VideoCapture(0)
    print('Webcam started, press "q" to quit.')
    # monotonic time for FPS calculation
    start_time = time.monotonic()
    last_timestamp_ms = 0

    try:
        with vision.FaceLandmarker.create_from_options(
            IMG_OPTIONS
        ) as locator, vision.FaceLandmarker.create_from_options(
            VIDEO_OPTIONS
        ) as feature_landmarker:
            while True:
                # 1. Capture webcam frame
                ret, frame = cap.read()
                height, width = frame.shape[:2]
                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )
                mp_frame = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(rgb_frame)
                )
                # Detect face
                full_result = locator.detect(mp_frame)
                if not full_result.face_landmarks:
                    cv2.putText(
                        frame,
                        "No face detected",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )
                    cv2.imshow("FER with trained MLP", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    continue
                # first detected face
                full_landmarks = full_result.face_landmarks[0]
                # crop face detected, resize !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                face_crop, bbox = crop_face(
                    frame,
                    full_landmarks
                )
                face_crop = cv2.resize(
                    face_crop,
                    (96, 96),
                    interpolation=cv2.INTER_AREA
                )

                # face_crop = cv2.resize(frame, (96, 96))
                # w/o cropping, just resize the whole frame to 96x96, for testing
                rgb_crop = cv2.cvtColor(
                    face_crop,
                    cv2.COLOR_BGR2RGB
                )
                mp_crop = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(rgb_crop)
                )
                # generate increasing timestamp for video frames, monotonic time in ms
                timestamp_ms = max(
                    last_timestamp_ms + 1,
                    int((time.monotonic() - start_time) * 1000)
                )
                last_timestamp_ms = timestamp_ms
                # extract landmarks, timestamp_ms shows the frame order, otherwise it will return the same result for all frames
                crop_result = feature_landmarker.detect_for_video(
                    mp_crop,
                    timestamp_ms
                )
                if len(crop_result.face_landmarks) == 0:
                    continue
                crop_landmarks = crop_result.face_landmarks[0]
                features = extract_features(crop_landmarks)
                features = preprocessing(features)
                top_predictions, probabilities = prediction(features)
                # see all preds
                for emotion, probability in zip(class_names, probabilities):
                    print(f"{emotion}: {probability * 100:.2f}%")
                # draw landmarks
                for landmark in full_landmarks:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    cv2.circle(frame,(x, y),1,(0, 255, 0),-1)
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame,(x1, y1),(x2, y2),(255, 200, 0),2)
                # show 2 predicted emotions
                for rank, (emotion, confidence) in enumerate(top_predictions,start=1):
                    label = f"{rank}. {emotion}: {confidence:.2f}%"
                    cv2.putText(frame,label,(15, 35 + rank * 40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 255, 0),2)
                cv2.putText(frame,"Press Q to quit",(15, height - 20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255, 255, 255),1)
                cv2.imshow("Live Facial Emotion Recognition",frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam()







