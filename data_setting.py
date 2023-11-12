import os
sentences = ['문장01', '문장02', '문장03', '문장04', '문장05', '문장06', '문장07', '문장08', '문장09']

types = ['모사', '자필']

states = ['벽에기대서_01', '벽에기대서_02', '벽에기대서_03', '벽에기대서_04', '벽에기대서_05', 
                '앉아서_01', '앉아서_02', '앉아서_03', '앉아서_04', '앉아서_05', 
                '일어서서_01', '일어서서_02', '일어서서_03', '일어서서_04', '일어서서_05']

IMAGE_HEIGHT, IMAGE_WIDTH = 768, 88

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(script_dir, '..')

# 이미지 및 JSON 기본 폴더 경로
base_image_folder = os.path.join(base_dir, '1.원천데이터')
base_json_folder = os.path.join(base_dir, '2.라벨링데이터')

