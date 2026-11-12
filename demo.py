# https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker?tab=Python
# class_names = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
# import json
# with open("class_names.json", "w") as f:
#     json.dump(class_names, f)

# 1. Import libraries, load model and landmarker
import json
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
OPTIONS = vision.FaceLandmarkerOptions(base_options=BASE_OPTIONS,
                                       output_face_blendshapes=True,
                                       running_mode=vision.RunningMode.IMAGE,
                                       num_faces=1)
DETECTOR = vision.FaceLandmarker.create_from_options(OPTIONS)


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

# 7. Test new img
TEST_IMG = 'image_happy.png'
OUTPUT_IMG_EMOTION = 'result_image.png'

def test_img(img_path):
    # 7.1 get prediction
    img = cv2.imread(img_path)
    output_img = img.copy()
    # resize to 96*96 (trained on) for MLP model, inter area is best for shrinking
    img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_AREA)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
    result = DETECTOR.detect(mp_image)

    if not result.face_landmarks:
        print("No face detected")
        return
    landmarks = result.face_landmarks[0]
    features = extract_features(landmarks)
    print("Features shape:", features.shape)
    features = preprocessing(features)
    # predict
    top_predictions,probabilities = prediction(features)
    for rank, (emotion, confidence) in enumerate(top_predictions, start=1):
        print(f"{rank}. {emotion}: {confidence:.2f}%")
    for name,score in zip(class_names, probabilities):
        print(f"{name}: {score*100}")

    # 7.2. visualize
    height, width, _ = output_img.shape
    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        cv2.circle(output_img, (x, y), 1, (0, 255, 0), -1)
    # display emotion text
    for rank, (emotion, confidence) in enumerate(top_predictions):
        label = f"{rank+1}.{emotion}: {confidence:.2f}%"
        cv2.putText(output_img, label, (10, 30 + rank * 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
    cv2.imwrite(OUTPUT_IMG_EMOTION, output_img)
    # show result
    cv2.imshow("Result", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_img(TEST_IMG)







