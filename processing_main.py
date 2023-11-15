import os
import processing as pc  # 필요한 처리 함수가 있는 모듈
from PIL import Image
import numpy as np
import data_setting as ds
import pandas as pd

def preprocess():

    all_images = []
    all_track_features = []
    all_labels = []

    for sentence in ds.sentences:
        for type in ds.types:
            for state in ds.states:
                image_folder = os.path.join(ds.base_image_folder, sentence, type, state)
                json_folder = os.path.join(ds.base_json_folder, sentence, type, state)

                # 이미지 로드 및 전처리
                images = pc.load_images_from_folder(image_folder, target_size=(ds.IMAGE_HEIGHT,ds.IMAGE_WIDTH))
                all_images.extend(images)

                # JSON 파일 로드 및 전처리
                json_data = pc.load_json_from_folder(json_folder)
                track_features = pc.process_json_data(json_data)
                all_track_features.extend(track_features)

                # 라벨 생성
                labels = [1 if type == '모사' else 0] * len(images)
                all_labels.extend(labels)
    for item in json_data[:2]:
        print(item)
        
        
def main():
    for sentence in ds.sentences:
        for type_ in ds.types:
            for state in ds.states:
                for i in range(1, 6):  # 각 상태별로 1부터 5까지 폴더가 있다고 가정
                    folder_name = f"{state}"  # 폴더 이름 형식 맞추기
                    image_folder = os.path.join(ds.base_image_folder, sentence, type_, folder_name)
                    json_folder = os.path.join(ds.base_json_folder, sentence, type_, folder_name)

                        # 데이터셋 준비
                    dataset = pc.prepare_dataset(image_folder, json_folder)
    print(dataset.shape)
if __name__ == "__main__":
    main()