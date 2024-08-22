from keras.datasets import mnist
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Input
import matplotlib.pyplot as plt

print("hellogithub")
#------------------------
(train_feature,train_label),(test_feature,test_label) = mnist.load_data()

#show test picture on index
def show_test_image(index):
    print("該照片的label="+str(test_label[index]))
    plt.imshow(test_feature[index])
    plt.show()


print("總共有"+str(train_feature.shape[0])+"張訓練照片")
print("每張照片有"+str(train_feature.shape[1])+"*"+str(train_feature.shape[2])+"像素")
print("-----------"*7)
print("總共有"+str(test_feature.shape[0])+"張測試照片")
print("每張照片有"+str(test_feature.shape[1])+"*"+str(test_feature.shape[2])+"像素")
print("-----------"*7)

#資料處理 1.將每張照片變成一維向量(28*28)共60000張，二微陣列
#numpyArrary.reshape(向量元素數) 一維陣列(向量)
#numpyArrary.reshape(列數,行數) 二維陣列
#numpyArrary.reshape(z軸數,列數,行數) 三維陣列
train_feature_vector = train_feature.reshape(60000,28*28)
test_feature_vector = test_feature.reshape(10000,28*28)
train_feature_vector = train_feature_vector.astype('float32')
test_feature_vector = test_feature_vector.astype('float32')

print("變成"+str(train_feature_vector.shape))

#將照片標準化
train_feature_vector_normal = train_feature_vector/255
test_feature_vector_normal = test_feature_vector/255

# 將label 轉換成 one hot enconding
# one hot encoding train and test label
train_label_onehot = np.eye(10)[train_label]
test_label_onehot = np.eye(10)[test_label]

# for i in range(10):
#     print(train_label_onehot[i])
#[0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
# [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
# [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
# [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
# [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
# [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
# [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
# [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
# [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
# print(type(train_label_onehot))
# print(type(test_label_onehot))

#create  model
model = Sequential()


#create input layer and hiding layer
model.add(
    Input(
        shape=(784,)
    )
)
model.add(
    Dense(
        units=256,
        # input_dim = 784,
        #input_shpae 被編譯氣推薦使用 和input_dim有不同效能
        # input_shape=(784,) ,
        kernel_initializer='normal',
        activation='relu'
    )
)

#create output layer
model.add(
    Dense(
        units = 10,
        kernel_initializer='normal',
        activation='softmax'
    )
)

#set up the model to be training
model.compile(
    # loss='categorical', tensorflow欣版本更改參數
    # loss = losses.get("categorical_crossentropy")
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

#training the model
train_history = model.fit(
    x = train_feature_vector_normal,
    y = train_label_onehot,
    validation_split = 0.2,
    epochs=10,
    batch_size=200,
    verbose=2
)

#analysis efficiency
score = model.evaluate(
    test_feature_vector_normal,test_label_onehot
)
print("準確率"+str(score))

#use the model for prediction
prediction = model.predict(test_feature_vector_normal)



print(prediction.shape)#(10000, 10) one hot type
print(prediction[0])
# [6.1954182e-08
# 6.3216223e-09
# 1.7559838e-05
# 2.9526558e-04
# 2.0462201e-10
#  6.9039356e-09
# 2.0531690e-12
# 9.9968314e-01
# 1.0157860e-06
# 2.9755272e-06]

# 這些數字通常是在分類問題中使用 softmax 函數生成的概率分佈。
# 在你的例子中，prediction[0] 是一個長度為 10 的數組，
# 假設你有一個 10 類的分類問題。這些數字的和應該接近 1，
# 表示該樣本屬於每個類別的概率。

# numpy Guide of np.argmax()
# https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
i = 0
while(True):
    print("the value of model predictions = "
    +str(np.argmax(prediction[i])))

    show_test_image(i)

    i+=1
    if i==len(test_feature):
        break



