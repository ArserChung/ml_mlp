
from keras.datasets import mnist
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt


#------------------------main -------------------

# call data of mnist
(train_feature,train_label),(test_feature,test_label) = mnist.load_data()

test_feature_vector = test_feature.reshape(10000,28*28)

test_feature_vector = test_feature_vector.astype('float32')

test_feature_vector_normal = test_feature_vector/255

#從 Minst_mlp_model.h5載入模組
model_name = 'Mnist_mlp_model.h5'
try:
    model = load_model(model_name)
except FileNotFoundError :
    print("模組仔入發生錯誤")
else :
    print("模組成功載入"+str(model_name))
#------------------try基本用法--------------------
# try:
#     # 嘗試執行的程式碼
# except ExceptionType as e:   or  except FileNotFoundError :
#                                         print("模組仔入發生錯誤")
#     # 如果發生特定的異常，會執行這裡的程式碼
#     print(e)
# else:
#     # 如果沒有發生異常，會執行這裡的程式碼（可選）
# finally:
#     # 無論是否發生異常，最後都會執行這裡的程式碼（可選)



#show test picture on index
def show_test_image(index,picture,picture_label):
    #pictrue = gringin
    print("該照片的label="+str(picture_label[index]))
    plt.imshow(picture[index])
    plt.show()


#prediction part
prediction = model.predict(test_feature_vector_normal)
print(prediction.shape)

i = 0
while(True):
    prediction_index = np.argmax(prediction[i])
    print("模組預測值為="+str(prediction_index))
    show_test_image(i,test_feature,test_label)
    i+=1
    if i==len(prediction):
        print("結束測試集")
        break





