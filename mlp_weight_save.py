from keras.datasets import mnist
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Input

#這邊卓的code與儲存訓練模型相同---------------
(train_feature,train_label),(test_feature,test_label) = mnist.load_data()


train_feature_vector = train_feature.reshape(60000,28*28)
test_feature_vector = test_feature.reshape(10000,28*28)
train_feature_vector = train_feature_vector.astype('float32')
test_feature_vector = test_feature_vector.astype('float32')

print("變成"+str(train_feature_vector.shape))


train_feature_vector_normal = train_feature_vector/255
test_feature_vector_normal = test_feature_vector/255


train_label_onehot = np.eye(10)[train_label]
test_label_onehot = np.eye(10)[test_label]


model = Sequential()


model.add(
    Input(
        shape=(784,)
    )
)
model.add(
    Dense(
        units=256,
        kernel_initializer='normal',
        activation='relu'
    )
)


model.add(
    Dense(
        units = 10,
        kernel_initializer='normal',
        activation='softmax'
    )
)


model.compile(

    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

#_--------------------------------------相同訓練模型


#這邊著重在，當訓練資料很大，我們可以先把訓練好的權重
#先儲存起來 save_weigths() load_weight()

# Weight是模型的參數(但不包括模型)，所以不能用
#"mlp_model_weight_save.weights.h5"不能去做預測

train_history = model.fit(
    x = train_feature_vector_normal,
    y = train_label_onehot,
    validation_split = 0.2,
    epochs=1,  #參數 : 每次訓練次數改成只訓練1次
    batch_size=200,
    verbose=2
)

#回傳兩個，第一個忽略(list)，用_,跳過
_,score = model.evaluate(
    test_feature_vector_normal,test_label_onehot
)

print("準確率="+str(score))
#準確率=[0.224, 0.9375] 明顯發現準確率只有93%

# model.save_weights("檔名.weights.h5")
try :
    model.save_weights("mlp_model_weight_save.weights.h5")
except :
    print("儲存權重失敗")
else :
    print("權重儲存成功，正確率="+str(score*100)+"%")
    del model

#在mlp_load_retraing.py 載入，並繼續訓練模型權重
# 使其準確率增加


