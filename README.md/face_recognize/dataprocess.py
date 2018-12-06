import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

from const import IMAGE_SIZE


def dealwithimage(img, h=64, w=64):
    ''' dealwithimage '''
    #img = cv2.imread(imgpath)
    top, bottom, left, right = getpaddingSize(img.shape[0:2])
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img = cv2.resize(img, (h, w))
    return img

def getpaddingSize(shape):
    ''' get size to make image to be a square rect '''
    h, w = shape
    longest = max(h, w)
    result = (np.array([longest]*4, int) - np.array([h, h, w, w], int)) // 2
    return result.tolist()

def relight(imgsrc, alpha=1, bias=0):
    '''relight'''
    imgsrc = imgsrc.astype(float)
    imgsrc = imgsrc * alpha + bias
    imgsrc[imgsrc < 0] = 0
    imgsrc[imgsrc > 255] = 255
    imgsrc = imgsrc.astype(np.uint8)
    return imgsrc

def _process_image(img_path, saved_dir):
    filename = os.path.splitext(os.path.basename(img_path))[0]
    img = cv2.imread(img_path)
    haar = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(gray_img, 1.3, 5)
    if len(faces) == 0: # no face in the image
        return 
    n = 0
    for f_x, f_y, f_w, f_h in faces:
        n+=1
        face = img[f_y:f_y+f_h, f_x:f_x+f_w]
        face = dealwithimage(face, IMAGE_SIZE, IMAGE_SIZE)
        for inx, (alpha, bias) in enumerate([[1, 1], [1, 50], [0.5, 0]]):
            face_handled = relight(face, alpha, bias)
            cv2.imwrite(os.path.join(saved_dir, "%s_%d_%d.jpg"%(filename, n, inx)), face_handled)
        return face

def process_image(img_dir, saved_dir):
    filelist = os.listdir(img_dir)
    for filename in filelist:
        _process_image(os.path.join(img_dir, filename), saved_dir)



if __name__ == "__main__":
    root_dir = "/home/wanxin/dataset/faces/"
    raw_path = os.path.join(root_dir, "raw")
    processed_path = os.path.join(root_dir, "processed")
    raw_path_file_list = os.listdir(raw_path)

    for folder in raw_path_file_list:
        print("folder name: %s"%folder)
        process_trainfile_path = os.path.join(raw_path, folder, "Train_DataSet")
        process_testfile_path = os.path.join(raw_path, folder, "Test_DataSet")
        assert os.path.exists(process_trainfile_path)
        assert os.path.exists(process_testfile_path)
        saved_trainfile_path = os.path.join(processed_path, folder, "Train_DataSet")
        saved_testfile_path = os.path.join(processed_path, folder, "Test_DataSet")
        if not os.path.exists(saved_trainfile_path):
            os.system("mkdir -p %s"%(saved_trainfile_path))
        if not os.path.exists(saved_testfile_path):
            os.system("mkdir -p %s"%(saved_testfile_path))
        process_image(process_trainfile_path, saved_trainfile_path)
        process_image(process_testfile_path, saved_testfile_path)
    print("Processed Done")
