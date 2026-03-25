import kagglehub 
import os
import matplotlib.pyplot as plt
import random 
import cv2

path = kagglehub.dataset_download("asdasdasasdas/garbage-classification")
base_dir = path +'/Garbage classification/Garbage classification'

labels = sorted(os.listdir(base_dir))
print(labels)
all_image_info = []

for label in labels:
    path = os.path.join(base_dir, label)
    for f in os.listdir(path):
        img_path = os.path.join(path, f)
        img = cv2.imread(img_path)
        if img is not None:
            h, w, c = img.shape
            all_image_info.append({
                'path': img_path,
                'label': label,
                'width': w,
                'height': h,
                'format': f.split('.')[-1].lower()
            })
            
# samples = random.sample(all_image_info, 15)
 
# plt.figure(figsize=(15, 10))
# for i, sample in enumerate(samples):
#     img = cv2.imread(sample['path'])
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     plt.subplot(3, 5, i+1)
#     plt.imshow(img)
#     plt.title(f"{sample['label']}\n{sample['width']}x{sample['height']}")
#     plt.axis("off")
# plt.suptitle("Task 5: Visual Inspection Grid", fontsize=16)
# plt.tight_layout()
# plt.show()

