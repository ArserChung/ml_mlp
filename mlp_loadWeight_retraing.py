from keras.datasets import mnist
import numpy as np
from keras.layers import Dense,Input
from keras.models import Sequential

# from keras.models import load_weight
#load_weight只能用模型載入權重，不能把load_weight()
#當作模型，載入預測

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

# model = load_wight()
# Weight是模型的參數(但不包括模型)，
# 所以不能用mlp_model_weight_save.weights.h5"不能
# 去做預測(他不是模型)

#要先宣告模型，再讓模型載入權重load_weight()
model = Sequential()

#在繼續訓練已經
# 加載權重的模型時，確保模型的架構與權重匹配是關鍵。

#1.重建模型架構
#2.加載權重
#3.繼續訓練

#注意事項-----------------------------------

# 架構必須匹配：模型架構在保存權重時必須與加載權重時的架構完全匹配。否則，模
# 型將無法正確加載權重，會導致錯誤或性能問題。

# 訓練配置：加載權重後，你可以根據需要重新編譯模型，設置優化器
# 、損失函數和評估指標。



# --------必須要先設置原始模型(參數不是模型)----------------
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
print("建置模型成功")

try :
    model.load_weights("mlp_model_weight_save.weights.h5")
except :
    print("模型載入權重失敗")
else :
    print("模型載入權重成功")

#繼續訓練
train_history = model.fit(
    x = train_feature_vector_normal,
    y = train_label_onehot,
    validation_split = 0.2,
    epochs=3, #每次在訓練3次
    batch_size=200,
    verbose=2
)

_,score = model.evaluate(
    test_feature_vector_normal,test_label_onehot
)
print("準確率為="+str(score))#0.9710000157356262變高了!

try:
    model.save("load_weight_new_model.h5")
except:
    print("儲存失敗")
else :
    print("load_weight_new_model.h5儲存成功")
    del model