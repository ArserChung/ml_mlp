from keras.datasets import mnist
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Input

#------------------------main -------------------

# call data of mnist
(train_feature,train_label),(test_feature,test_label) = mnist.load_data()

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
score = model.evaluate(
    test_feature_vector_normal,test_label_onehot
)
print("準確率="+str(score))
model.save('Mnist_mlp_model.h5')
print("The model has already been saved")
del model