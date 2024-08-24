import cv2 as cv
import numpy as np
import glob as gb
from keras.models import load_model

files = gb.glob("data\*")
# ['data\\1_1.jpg', 'data\\1_2.jpg',
# 'data\\1_3.jpg', 'data\\2_1.jpg',
# 'data\\2_2.jpg', 'data\\3_1.jpg',
# 'data\\4_1.jpg', 'data\\4_2.png',
# 'data\\5_1.jpg', 'data\\6_1.jpg',
# 'data\\6_2.jpg', 'data\\7_1.jpg',
# 'data\\7_2.jpg', 'data\\7_3.jpg',
# 'data\\8_1.jpg', 'data\\9_1.jpg']


def file_image(files):
    test_feature_list = []
    test_label_list = []
    for i in files:
        img = cv.imread(i) #RGB TYPE if沒有下面兩行 shape = (16, 28, 28, 3)
        #灰階
        img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        # #_,站位符，筆記在,黑轉白 白轉黑
        _,img = cv.threshold(img,127,255,cv.THRESH_BINARY_INV)
        test_feature_list.append(img)
        test_label_list.append(int(i[5:6]))

    return [test_feature_list,test_label_list]



data =  file_image(files)
test_feature = np.array(data[0]) #(16,28,28)
test_label = np.array(data[1]) #(16,)


#transfor to vector as type float32
test_feature_vector = test_feature.reshape(len(test_feature),28*28)
test_feature_vector = test_feature_vector.astype('float32')

#vertor normalizable
test_feature_vector_normal = test_feature_vector/255

try:
    model = load_model('Mnist_mlp_model.h5')
except :
    print("load model")







