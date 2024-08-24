from keras.datasets import mnist
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Input,Dropout
#插入Dropout()

(train_feature,train_label),(test_feature,test_label) = mnist.load_data()



train_feature_vertor = train_feature.reshape(60000,28*28)
train_feature_vertor = train_feature_vertor.astype('float32')
train_feature_vertor_normal = train_feature_vertor/255.0

test_feature_vector = test_feature.reshape(10000,28*28)
test_feature_vector = test_feature_vector.astype('float32')
test_feature_vector_normal = test_feature_vector/255.0

train_label_onehot = np.eye(10)[train_label]
test_label_onehot = np.eye(10)[test_label]

model = Sequential()

model.add(
    Input(
        shape=(784,)
    )
)

#建立第一個隱藏層數，神經元256
model.add(
    Dense(
        units=256,
        #kernel_initializer: 這個參數用於
        # 指定初始化層的權重矩陣的方式。
        kernel_initializer='normal',
        activation='relu'
    )
)

#將第一個隱藏層Dropout
#model.add(Dropout(放棄百分比))
#放棄百分比表示拋棄層中要放棄神經元的百分比
model.add(
    Dropout(0.2) #加入第一層20%的神經元進拋棄層
)

model.add( #第二層隱藏層，神經元128
    Dense(
        units=128,
        kernel_initializer='normal',
        activation='relu'
    )
)
model.add(
    Dropout(0.2)#加入第二層隱藏層神經元20%進拋棄層
)
model.add(
    Dense(
        units=10,
        kernel_initializer='normal',
        activation='softmax'
    )
)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy']
)

model.fit(
    x = train_feature_vertor_normal,
    y = train_label_onehot,
    validation_split = 0.2,
    epochs=1,  #參數 : 每次訓練次數改成只訓練1次
    batch_size=200,
    verbose=2
)

_,score = model.evaluate(
    test_feature_vector_normal,test_label_onehot
)
print(score)
try :
    model.save("new_model.h5")
except :
    print("file")
else:
    print("成功儲存模組，準確率="+str(score))
#準確率=0.9434833526611328 ，多一個隱藏層，且
#適當的加入拋棄層，可以提高準確率