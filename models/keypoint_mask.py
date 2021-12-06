import os
import sys
import cv2
import dlib
import numpy as np

# change directory to this module path
try:
    this_file = __file__
except NameError:
    this_file = sys.argv[0]
this_file = os.path.abspath(this_file)

# print this_file

if os.path.dirname(this_file):
    os.chdir(os.path.dirname(this_file))

script_dir = os.getcwd()
# print script_dir


def get_rect_mask(shape, index_area=0, h=None, w=None, w_ratio=None, h_ratio=None):
    """
    shape : image of landmarks/parts
    index_area:
        0:refer to left eye
        1:refer to right eye
        2:refer to nose;
        3:refer to mouth or face; 
    """
    if index_area == 0:  # 36-41 six points

        if w_ratio is None or h_ratio is None:
            w_ratio = np.random.choice([2.0, 2.05, 2.10, 2.15], 1, p=[0.2, 0.4, 0.2, 0.2])[0]
            h_ratio = np.random.choice([2.3, 2.35, 2.25, 2.38], 1, p=[0.2, 0.4, 0.25, 0.15])[0]

        x_center = int(shape.part(36).x + shape.part(37).x + shape.part(39).x) / 3
        y_center = int(shape.part(37).y + shape.part(41).y) / 2
        width = int(w_ratio * (shape.part(39).x - shape.part(36).x) / 2)
        height = int(h_ratio * (shape.part(41).y - shape.part(37).y) / 2)

        x0 = max(x_center - width, 0)
        y0 = max(y_center - height, 0)

        x1 = min(x_center + width, w)
        y1 = max(y_center - height, 0)

        x2 = min(x_center + width, w)
        y2 = min(y_center + height, h)

        x3 = max(x_center - width, 0)
        y3 = min(y_center + height, h)

        pts = np.asarray([[x0, y0], [x1, y1], [x2, y2], [x3, y3]])

        return pts

    elif index_area == 1:  # 42-47 six points

        if w_ratio is None or h_ratio is None:
            w_ratio = np.random.choice([2.0, 2.05, 2.10, 2.15], 1, p=[0.2, 0.4, 0.2, 0.2])[0]
            h_ratio = np.random.choice([2.3, 2.35, 2.25, 2.38], 1, p=[0.2, 0.4, 0.25, 0.15])[0]

        x_center = int(shape.part(42).x + shape.part(45).x + shape.part(44).x) / 3
        y_center = int(shape.part(43).y + shape.part(47).y) / 2

        width = int(w_ratio * (shape.part(45).x - shape.part(42).x) / 2)
        height = int(h_ratio * (shape.part(47).y - shape.part(43).y) / 2)

        x0 = max(x_center - width, 0)
        y0 = max(y_center - height, 0)

        x1 = min(x_center + width, w)
        y1 = max(y_center - height, 0)

        x2 = min(x_center + width, w)
        y2 = min(y_center + height, h)

        x3 = max(x_center - width, 0)
        y3 = min(y_center + height, h)

        pts = np.asarray([[x0, y0], [x1, y1], [x2, y2], [x3, y3]])

        return pts

    elif index_area == 2:  # 27-35 nine points
        if w_ratio is None or h_ratio is None:
            w_ratio = np.random.choice([1.3, 1.4, 1.45], 1, p=[0.3, 0.4, 0.3])[0]
            h_ratio = np.random.choice([0.8, 0.9, 1.0], 1, p=[0.5, 0.4, 0.1])[0]

        x_center = int(shape.part(31).x + shape.part(35).x + shape.part(27).x) / 3
        width = int(w_ratio * (shape.part(35).x - shape.part(31).x) / 2)

        y_center = int(shape.part(27).y + shape.part(29).y + shape.part(33).y) / 3

        height = int(h_ratio * (shape.part(33).y - shape.part(27).y + 10) / 2)

        x0 = x_center
        y0 = max(y_center - height, 0)

        x1 = max(x_center - width, 0)
        y1 = min(y_center + height, h)

        x2 = min(x_center + width, w)
        y2 = min(y_center + height, h)

        pts = np.asarray([[x0, y0], [x1, y1], [x2, y2]])

        return pts

    elif index_area == 3:  # 1.2,1.5

        if w_ratio is None or h_ratio is None:
            w_ratio = np.random.choice([0.95, 1.15, 1.2, 1.25], 1, p=[0.25, 0.2, 0.25, 0.3])[0]
            h_ratio = np.random.choice([0.85, 0.95, 1.05, 1.1], 1, p=[0.3, 0.25, 0.35, 0.1])[0]

        x_center = int(shape.part(48).x + shape.part(54).x) / 2
        width = int(w_ratio * (shape.part(54).x - shape.part(48).x) / 2)

        y_center = int(shape.part(58).y + shape.part(50).y) / 2

        height = int(h_ratio * (shape.part(58).y - shape.part(50).y) / 2)

        x0 = max(x_center - width, 0)
        y0 = max(y_center - height, 0)

        x1 = min(x_center + width, w)
        y1 = max(y_center - height, 0)

        x2 = min(x_center + width, w)
        y2 = min(y_center + height, h)

        x3 = max(x_center - width, 0)
        y3 = min(y_center + height, h)

        pts = np.asarray([[x0, y0], [x1, y1], [x2, y2], [x3, y3]])

        return pts

    elif index_area == 4:

        if w_ratio is None or h_ratio is None:
            w_ratio = np.random.choice([1.2, 1.35, 1.25, 1.3, 1.32], 1, p=[0.2, 0.2, 0.2, 0.3, 0.1])[0]
            h_ratio = np.random.choice([0.6, 0.66, 0.56, 0.48, 0.5], 1, p=[0.2, 0.2, 0.3, 0.2, 0.1])[0]

        Flags = np.random.choice([0, 1], 1)
        #已关键点确定mask的中心点坐标 (x_center, y_center_)
        if Flags == 0:
            x_point = np.random.choice([3, 4, 5, 6], 2, p=[0.3, 0.3, 0.2, 0.2])
        else:
            x_point = np.random.choice([10, 11, 12, 13], 2, p=[0.25, 0.3, 0.25, 0.2])

        x_center = int((shape.part(x_point[0]).x + shape.part(x_point[1]).x) / 2)
        y_point = np.random.choice([28, 29, 30, 33], 2, p=[0.2, 0.3, 0.3, 0.2])

        y_center = int((shape.part(y_point[0]).y + shape.part(y_point[1]).y) / 2)
        # 中心点的左右宽度
        width = int(w_ratio * (shape.part(29).x - shape.part(2).x) / 2)

        # 中心点的上下高度
        height = int(h_ratio * (shape.part(8).y - shape.part(20).y))


        x0 = max(x_center - width, int(shape.part(2).x))
        y0 = max(y_center - height, 5)

        x1 = min(x_center + width, int(shape.part(16).x))
        y1 = max(y_center - height, 5)

        x2 = min(x_center + width, int(shape.part(16).x))
        y2 = min(y_center + height, h - 5)

        x3 = max(x_center - width, int(shape.part(2).x))
        y3 = min(y_center + height, h - 5)

        pts = np.asarray([[x0, y0], [x1, y1], [x2, y2], [x3, y3]])

        return pts

    else:
        raise ValueError("parameter index_area value is invalid")


def add_mask(dir_path):
    img_list = os.listdir(dir_path)

    save_dir = "./occ_mask"

    for item in img_list:
        full_path = os.path.join(dir_path, item)

        img = cv2.imread(full_path)

        height, width = img.shape[:2]
        print(height, width)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 使用默认人脸识别的模型
        detector = dlib.get_frontal_face_detector()
        # 获取人脸关键点预训练模型
        # 模型获取地址：http://dlib.net/files/
        predictor = dlib.shape_predictor("./config/shape_predictor_68_face_landmarks.dat")
        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        dets = detector(gray, 1)
        #print("Number of faces detected: {}".format(len(dets)))

        for k, d in enumerate(dets):
            print("Left: {} Top: {} Right: {} Bottom: {}".format(
                d.left(), d.top(), d.right(), d.bottom()))
            # Get the landmarks/parts for the face in box d.
            shape = predictor(img, d)
            # print shape.part(0).x, shape.part(0).y

            pts_list = []
            # 随机选择黑框的个数
            nums_mask = np.random.choice([1, 2, 3], 1, p=[0.35, 0.5, 0.15])[0]
            # 随机选择关键点的大致位置
            index_list = np.random.randint(0, 4, nums_mask)
            # print set(index_list)
            #index_list = [4]
            for index in set(index_list):
                # print index
                pts_list.append(get_rect_mask(shape, index, height, width))
            #image = cv2.fillConvexPoly(img,pts,(0,0,0))
            image = cv2.fillPoly(img, pts_list, (0, 0, 0))

            save_name = os.path.join(save_dir, item)
            cv2.imwrite(save_name, image)


if __name__ == "__main__":

    dir_path = sys.argv[1]
    add_mask(dir_path)
