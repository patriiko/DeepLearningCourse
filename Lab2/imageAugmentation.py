import os
import random
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

training_images_path = "dataset/seg_train/seg_train/"
augmented_images_path = "dataset/seg_train/augmented_training"
os.makedirs(augmented_images_path, exist_ok=True)

# Dohvaćanje slika iz svih klasa
all_images = []
for class_folder in os.listdir(training_images_path):
    class_path = os.path.join(training_images_path, class_folder)
    if os.path.isdir(class_path):
        for img_name in os.listdir(class_path):
            all_images.append((os.path.join(class_path, img_name), class_folder))

# 20% slučajnih slika
sample_images = random.sample(all_images, int(0.2 * len(all_images)))

# Kreiraj folder za sample slike
for _, class_folder in sample_images:
    os.makedirs(os.path.join(augmented_images_path, class_folder), exist_ok=True)


for img_path, class_folder in sample_images:
    img = Image.open(img_path).convert("RGB")

    img = img.resize((224, 224), Image.Resampling.LANCZOS)

    img.save(os.path.join(augmented_images_path, class_folder, "orig_" + os.path.basename(img_path)))

    img_color = img.point(lambda p: p * random.uniform(0.8, 1.2))
    img_color.save(os.path.join(augmented_images_path, class_folder, "color_" + os.path.basename(img_path)))

    angle = random.randint(-25, 25)
    img_rot = img.rotate(angle)
    img_rot.save(os.path.join(augmented_images_path, class_folder, "rot_" + os.path.basename(img_path)))

    crop_size = 200
    left = random.randint(0, 224 - crop_size)
    top = random.randint(0, 224 - crop_size)
    img_crop = img.crop((left, top, left + crop_size, top + crop_size)).resize((224, 224))
    img_crop.save(os.path.join(augmented_images_path, class_folder, "crop_" + os.path.basename(img_path)))

    img_blur = img.filter(ImageFilter.GaussianBlur(radius=2))
    img_blur.save(os.path.join(augmented_images_path, class_folder, "blur_" + os.path.basename(img_path)))


plt.imshow(img_rot)
plt.title("Primjer augmentirane slike (rotated)")
plt.axis("off")
plt.show()