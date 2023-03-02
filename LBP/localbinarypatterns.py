# import paket-paket yang diperlukan
from skimage import feature
from skimage.transform import pyramid_reduce
import cv2
import numpy as np
import re
import imutils


class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # simpan nilai number of points dan radius
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):
        # hitung representasi Pola Biner Lokal gambar,
        # lalu gunakan representasi LBP untuk membangun pola histogram
        lbp = feature.local_binary_pattern(image, self.numPoints, self.radius, method='uniform')
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.numPoints + 3), range=(0, self.numPoints + 2))

        # menormalkan histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        # mengembalikan fitur histogram Local Binary Patterns
        return hist

    # -------------------- Utility function ------------------------
    def normalize_label(self, str_):
        str_ = str_.replace(" ", "")
        str_ = str_.translate(str_.maketrans("", "", "()"))
        str_ = str_.split("_")
        return ''.join(str_[:2])

    def normalize_desc(self, folder, sub_folder):
        text = folder + " - " + sub_folder
        text = re.sub(r'\d+', '', text)
        text = text.replace(".", "")
        text = text.strip()
        return text

    def print_progress(self, val, val_len, folder, sub_folder, filename, bar_size=10):
        progr = "#" * round((val) * bar_size / val_len) + " " * round((val_len - (val)) * bar_size / val_len)
        if val == 0:
            print("", end="\n")
        else:
            print("[%s] folder : %s/%s/ ----> file : %s" % (progr, folder, sub_folder, filename), end="\r")

    # jangan gunakan method ini
    def crop_resize(self, image):
        h, w = image.shape
        ymin, ymax, xmin, xmax = h // 3, h * 2 // 3, w // 3, w * 2 // 3
        crop = image[ymin:ymax, xmin:xmax]
        image_resize = cv2.resize(crop, (0, 0), fx=0.5, fy=0.5)
        return image_resize

    # method ini kurang baik
    def crop_resize_interpolation(self, image, square_size):
        height, width = image.shape
        if(height > width):
            differ = height
        else:
            differ = width
        differ += 4

        # square filler
        mask = np.zeros((differ, differ), dtype="uint8")
        x_pos = int((differ - width) / 2)
        y_pos = int((differ - height) / 2)

        # center image inside the square
        mask[y_pos: y_pos + height, x_pos: x_pos + width] = image[0: height, 0: width]

        # downscale if needed
        if differ / square_size > 1:
            mask = pyramid_reduce(mask, differ / square_size)
        else:
            mask = cv2.resize(mask, (square_size, square_size), interpolation=cv2.INTER_AREA)
        return mask

    # gunakan ini, the best
    def crop_resize_imutils(self, image, ratio):
        image_resize = imutils.resize(image, width=ratio)
        return image_resize
    # -------------------- End Utility function ------------------------
