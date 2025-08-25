import os
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np

def balance_dataset(input_dir, output_dir):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Go through train and val folders
    for split in ['train', 'val']:
        split_input = os.path.join(input_dir, split)
        split_output = os.path.join(output_dir, split)
        os.makedirs(split_output, exist_ok=True)

        # Copy all classes and files
        for cls in os.listdir(split_input):
            cls_input = os.path.join(split_input, cls)
            cls_output = os.path.join(split_output, cls)
            os.makedirs(cls_output, exist_ok=True)

            files = os.listdir(cls_input)
            for f in files:
                shutil.copy(os.path.join(cls_input, f), cls_output)

        # Balance classes
        classes = os.listdir(split_input)
        counts = {cls: len(os.listdir(os.path.join(split_input, cls))) for cls in classes}
        max_count = max(counts.values())

        print(f"\n[{split}] Balancing dataset...")
        for cls in classes:
            cls_input = os.path.join(split_input, cls)
            cls_output = os.path.join(split_output, cls)

            current_count = counts[cls]
            print(f"  Class {cls}: {current_count} → {max_count}")

            if current_count < max_count:
                files = os.listdir(cls_input)
                i = 0
                while current_count < max_count:
                    img_path = os.path.join(cls_input, files[i % len(files)])
                    img = load_img(img_path)  
                    x = img_to_array(img)
                    x = np.expand_dims(x, axis=0)

                    # Generate augmented image
                    for batch in datagen.flow(x, batch_size=1,
                                              save_to_dir=cls_output,
                                              save_prefix="aug",
                                              save_format="jpg"):
                        break
                    current_count += 1
                    i += 1

    print("\n✅ Balanced dataset created at:", output_dir)


# USAGE
input_dir = "im1"                # your original dataset folder
output_dir = "im1_balanced"      # new folder with balanced data
balance_dataset(input_dir, output_dir)
