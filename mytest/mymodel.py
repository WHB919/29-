# time: 2023/12 
# Author: whb
# 代码功能：定义原始模型

import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Dropout, Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import MaxPooling1D as MaxPooling
from tensorflow.python.keras.layers import Conv1D as Conv
from tensorflow.python.keras.layers import GlobalAveragePooling1D as GlobalAveragePooling
import resnet50_1D as resnet50

def myblock1(input, block_idx):
    kernel_size = ['None', 7, 5]
    filter_num = ['None', 128, 128]
    conv_stride = ['None', 1, 1]
    pool_size = ['None', 2]
    pool_stride = ['None', 1]
    act_func = 'relu'

    model = Conv(filters=filter_num[1], kernel_size=kernel_size[1], name='conv1_{}'.format(block_idx),
                 strides=conv_stride[1], padding='same', activation=act_func)(input)
    model = Conv(filters=filter_num[2], kernel_size=kernel_size[2], name='conv2_{}'.format(block_idx),
                 strides=conv_stride[2], padding='same', activation=act_func)(model)
    output = MaxPooling(pool_size=pool_size[1], strides=pool_stride[1], padding='same',
                        name='pool1_{}'.format(block_idx))(model)

    return output

# CNN网络，基于AlexNet
def mymodel1(inp_shape, NUM_CLASS):
    dense_layer_size = ['None', 256, 256, 128]
    act_func = ['None', 'relu', 'relu', 'relu']
    print(inp_shape)
    blockNum = 4
    input_data = Input(shape=inp_shape)

    for i in range(blockNum):
        idx = i + 1
        if 0 == i:
            x = myblock1(input_data, idx)
        else:
            x = myblock1(x, idx)

    x = GlobalAveragePooling()(x)

    x = Dense(dense_layer_size[1], name='dense1', activation=act_func[1])(x)
    x = Dense(dense_layer_size[2], name='dense2', activation=act_func[2])(x)
    x = Dense(dense_layer_size[3], name='dense3', activation=act_func[3])(x)

    out = Dense(NUM_CLASS, name='dense4', activation='softmax')(x)

    conv_model = Model(inputs=input_data, outputs=out)
    return conv_model

# CNN网络，基于resnet50
def mymodel2(inp_shape, NUM_CLASS):
    return resnet50.create_model(inp_shape, NUM_CLASS)

# CNN网络
def mymodel3(inp_shape, NUM_CLASS):
    input_data = Input(shape=inp_shape)
    conv1 = Conv(filters=32, kernel_size=3, padding="same", activation=tf.nn.relu)(input_data)
    pool1 = MaxPooling(pool_size=2, strides=2, padding="same")(conv1)
    conv2 = Conv(filters=64, kernel_size=3, padding="same", activation=tf.nn.relu)(pool1)
    pool2 = MaxPooling(pool_size=2, strides=2, padding="same")(conv2)
    conv3 = Conv(filters=128, kernel_size=2, padding="same", activation=tf.nn.relu)(pool2)
    conv4 = Conv(filters=128, kernel_size=2, padding="same", activation=tf.nn.relu)(conv3)
    pool3 = GlobalAveragePooling()(conv4)
    fc1 = Dense(units=64, activation=tf.nn.relu)(pool3)
    fc2 = Dense(units=NUM_CLASS, activation=None)(fc1)
    out = tf.nn.softmax(fc2)
    conv_model = Model(inputs=input_data, outputs=out)

    return conv_model

# RNN网络
def mymodel4(input_shape, NUM_CLASS):
    inputs = tf.keras.Input(shape=input_shape)
    # 嵌入层
    # x = tf.keras.layers.Embedding(input_dim=10000, output_dim=128, input_length=input_shape)(inputs)
    # # 注意力层
    # x = tf.keras.layers.Attention()([x, x])
    # GRU层
    x = tf.keras.layers.GRU(256, return_sequences=True)(inputs)
    x = tf.keras.layers.GRU(256, return_sequences=True)(x)
    x = tf.keras.layers.GRU(256, return_sequences=True)(x)
    # 全连接层
    x = GlobalAveragePooling()(x)
    outputs = tf.keras.layers.Dense(NUM_CLASS, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='attention_rnn_model')
    return model

# 全连接网络
def mymodel5(input_shape, NUM_CLASS):
    inputs = tf.keras.Input(shape=input_shape)
    x = Dense(128, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    outputs = Dense(NUM_CLASS,activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model