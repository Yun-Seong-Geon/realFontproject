#json 파일을 살펴보는 코드
import json
with open('JJ_BF01_M0001_1952745_1.json', encoding='utf-8') as data_file:
    ist = json.load(data_file)

print(type(ist))
print(ist.keys())
print(ist['meta'].keys())
print(ist['annotations'])