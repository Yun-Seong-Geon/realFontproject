import pytesseract
import cv2
import matplotlib.pyplot as plt
import os

#tesseract 실험
path = r'C:\Users\kwonh\Desktop\fontpro\realFontproject\imgs\test_image_1.png'
os.path.isfile(path)

image = cv2.imread(path)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(rgb_image) 

text = pytesseract.image_to_string(rgb_image, lang='kor')
print(text)