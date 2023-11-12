import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import json
import numpy as np
from sklearn.preprocessing import OneHotEncoder
def load_images_from_folder(folder, target_size):
    images = []
    for filename in sorted(os.listdir(folder)):
        if filename.endswith('.png'):  # 또는 해당 이미지 형식
            img_path = os.path.join(folder, filename)
            img = load_img(img_path, target_size=target_size)
            images.append(img_to_array(img))
    return np.array(images)


def load_json_from_folder(folder):
    json_data = []
    for filename in sorted(os.listdir(folder)):
        if filename.endswith('.json'):
            file_path = os.path.join(folder, filename)
            with open(file_path, 'r', encoding='utf-8-sig') as file:  # UTF-8 인코딩 지정
                data = json.load(file)
                json_data.append(data)
    return json_data

def process_json_data(json_data):
    processed_data = []

    for item in json_data:
        # 예시: 트랙 데이터에서 각 센싱 특징을 추출
        features = []
        for track_item in item['track']:
            sensing_features = track_item['sensing_feature']
            feature = [
                sensing_features['time_interval'],
                sensing_features['action'],  # 'DOWN', 'UP', 'MOVE'와 같은 문자열
                sensing_features['x'],
                sensing_features['y'],
                sensing_features['pressure'],
                sensing_features['velocity_magnitude'],
                sensing_features['acceleration_magnitude'],
                sensing_features['ratio_minimum_maximum_speed_5-samples'],
                sensing_features['angle_consecutive_samples'],
                sensing_features['sine_consecutive_samples'],
                sensing_features['cosine_consecutive_samples'],
                sensing_features['log_curvature_radius'],
                # 차분 값들
                sensing_features['difference']['x'],
                sensing_features['difference']['y'],
                sensing_features['difference']['pressure'],
                sensing_features['difference']['velocity_magnitude'],
                sensing_features['difference']['acceleration_magnitude'],
                sensing_features['difference']['angle_consecutive_samples'],
                # 2차 차분 값들
                sensing_features['second_order_difference']['x'],
                sensing_features['second_order_difference']['y'],
                # 필기 샘플 길이 비율
                sensing_features['stroke_length_ratio']['3-samples'],
                sensing_features['stroke_length_ratio']['5-samples'],
                sensing_features['stroke_length_ratio']['7-samples'],
                sensing_features['stroke_length_ratio']['12-samples']
            ]
            features.append(feature)
        processed_data.append(features)

    return processed_data



