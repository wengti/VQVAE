import csv
from pathlib import Path
import numpy as np
import cv2

def extract_mnist(csv_fname, save_folder):

    save_folder = Path(save_folder)
    if not save_folder.is_dir():
        save_folder.mkdir(parents = True,
                          exist_ok = True)

    with open(csv_fname, "r") as file:
        reader = csv.reader(file)

        for idx, row in enumerate(reader):

            if idx == 0:
                continue

            img = np.zeros((28*28))
            img[:] = list(map(int, row[1:]))
            img = img.reshape((28,28))

            save_class_folder = save_folder / row[0]
            if not save_class_folder.is_dir():
                save_class_folder.mkdir(parents = True,
                                        exist_ok = True)

            save_file = save_class_folder / f"{idx}.png"
            cv2.imwrite(save_file, img)

            if (idx+1)%1000 == 0:
                print(f"[INFO] {idx+1} images have been saved into save_folder.")


extract_mnist(csv_fname = "./mnist_train.csv",
              save_folder = "./data/train")

extract_mnist(csv_fname = "./mnist_test.csv",
              save_folder = "./data/test")