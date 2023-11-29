import json
import os
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np

image_dir = r"..\fontpro\jeju_img"
json_dir = r"C:\Users\kwonh\Desktop\fontpro\jeju_label"

#이미지와 라벨링 데이터를 로드하는 함수
def load_data(image_dir, json_dir, target_size=(256, 256)):
    data = []
    for json_file in os.listdir(json_dir):
        json_path = os.path.join(json_dir, json_file)
        with open(json_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
            image_file = json_data['meta']['file_name']
            image_path = os.path.join(image_dir, image_file)

            # 이미지의 원본 크기를 가져옵니다
            with Image.open(image_path) as img:
                original_size = img.size

            annotations = json_data['annotations']
            for annotation in annotations:
                text = annotation['ocr']['text']
                x = annotation['ocr']['x']
                y = annotation['ocr']['y']
                width = annotation['ocr']['width']
                height = annotation['ocr']['height']
                data.append((image_path, text, (x, y, x + width, y + height), original_size))
    return data

def resize_box(original_size, new_size, box):
    x_ratio = new_size[0] / original_size[0]
    y_ratio = new_size[1] / original_size[1]
    resized_box = (
        int(box[0] * x_ratio),
        int(box[1] * y_ratio),
        int(box[2] * x_ratio),
        int(box[3] * y_ratio)
    )
    return resized_box


# 이미지와 라벨링 데이터 로드
t_data = load_data(image_dir, json_dir)

#이미지를 넘파이 배열로 바꾸기
def images_to_numpy(image_dir, target_size=(256, 256)):    
    image_list = []
    for img_filename in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_filename)
        img = Image.open(img_path)
        img_resized = img.resize(target_size)
        img_array = np.array(img_resized)
        image_list.append(img_array)
    return image_list

#이미지 넘파이 배열 리스트
i_data = images_to_numpy(image_dir, target_size=(256, 256))

#하나의 데이터로 만듬
#training_data = list(zip(i_data, t_data))

save_img_path = 'timg'

for i, img in enumerate(i_data):
    image = Image.fromarray(img)
    filename = f'image_{i}.tif'
    save_path = os.path.join(save_img_path, filename)
    image.save(save_path)


def save_box_and_text_files(t_data, box_dir, text_dir, target_size=(256, 256)):
    # 이미지 경로별로 데이터를 그룹화
    grouped_data = {}
    for image_path, text, box, original_size in t_data:
        if image_path not in grouped_data:
            grouped_data[image_path] = []
        resized_box = resize_box(original_size, target_size, box)
        grouped_data[image_path].append((text, resized_box))

    # 각 이미지 파일에 대해 하나의 .box와 .txt 파일 생성
    for i, (image_path, annotations) in enumerate(grouped_data.items()):
        filename = f'image_{i}'
        box_file_path = os.path.join(box_dir, filename + '.box')
        text_file_path = os.path.join(text_dir, filename + '.txt')

        with open(box_file_path, 'w', encoding='utf-8') as box_file, open(text_file_path, 'w', encoding='utf-8') as text_file:
            for text, box in annotations:
                box_line = f"{text} {box[0]} {box[1]} {box[2]} {box[3]} 0\n"
                box_file.write(box_line)
                text_file.write(text + '\n')

# 이미지와 라벨링 데이터 로드
# 예시: data = load_data('path/to/image_dir', 'path/to/json_dir')

# .box 파일과 .txt 파일을 저장할 폴더
box_dir = 'timg'
text_dir = 'ttxt'

save_box_and_text_files(t_data,box_dir,text_dir)




#script로 진행
#for img in *.tif; do
#    tesseract "$img" "${img%.*}" --psm 6 lstm.train
#done

# 3개 파일 오류로 지움

#lstmf 파일 경로를 txt 파일로 생성

#combine_tessdata -e C:\Users\kwonh\Desktop\fontpro\kor_1.traineddata kor.lstm을 사용하여 kor.lstm 생성

#lstmtraining "--model_output" "C:\Users\kwonh\Desktop\fontpro\output\model" 
# "--continue_from" "C:\Users\kwonh\Desktop\fontpro\kor.lstm"  "--traineddata" 
# "C:/Users/kwonh/Desktop/fontpro/kor.traineddata" "--train_listfile" 
# "C:\Users\kwonh\Desktop\fontpro\timg\training_files.txt" 
# "--max_iterations" "100"            ->을 사용하여 진행 -> Deserialize header failed (lstmf 생성 시 문제 혹은 모델 호환성 문제로 훈련실패)





