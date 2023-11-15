import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import json
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

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




def load_json_to_dataframe(json_file_path):
    # JSON 파일 로드
    with open(json_file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)

    # 데이터프레임으로 변환
    df = pd.json_normalize(json_data, record_path=['image'], 
                            meta=[['track', 'timestamp'], 
                                ['environments', 'location'], 
                                ['environments', 'weather'], 
                                ['person', 'id'], 
                                ['person', 'sex'], 
                                ['person', 'age_group']])
    return df

def map_images_to_json(image_folder, json_df):
    image_files = os.listdir(image_folder)
    mapped_data = []

    for image_file in image_files:
        if image_file in json_df['image_info.file_name'].values:
            row = json_df[json_df['image_info.file_name'] == image_file].iloc[0]
            mapped_data.append((image_file, row))

    return pd.DataFrame(mapped_data, columns=['image_file', 'json_data'])


def add_video_id_column(df, base_folder):
    df['video_id'] = df['image_file'].apply(lambda x: os.path.relpath(os.path.dirname(x), base_folder))
    return df

# 데이터셋 준비
def prepare_dataset(base_folder, json_file_path):
    json_df = load_json_to_dataframe(json_file_path)

    for root, dirs, files in os.walk(base_folder):
        for dir in dirs:
            image_folder = os.path.join(root, dir)
            mapped_df = map_images_to_json(image_folder, json_df)
            final_df = add_video_id_column(mapped_df, base_folder)
            # 여기서 final_df를 사용하여 데이터셋을 준비하거나 저장
    return final_df
