# time: 2023/12 
# Author: whb
# 代码功能：训练并测试模型

import os
import sys
import random
import tensorflow as tf

import config, load_data_csv
import mymodel as mymodel

from tensorflow.python.keras import optimizers

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = os.getenv('ROOT_DIR')
print(ROOT_DIR)
resDir = os.path.join(ROOT_DIR, 'resDir')
modelDir = os.path.join(resDir, 'modelDir')
os.makedirs(modelDir, exist_ok=True)


def loadMyData(opts):
    print('loading data...')
    dataOpts = load_data_csv.loadDataOpts(opts.input)
    train_x, train_y, test_x, test_y, NUM_CLASS = load_data_csv.loadData(dataOpts)

    if opts.normalize:
        train_x = load_data_csv.normalizeData(train_x)
        test_x = load_data_csv.normalizeData(test_x)

    return train_x, train_y, test_x, test_y, NUM_CLASS

def main(opts):
    # load data
    if 'neu' == opts.dataSource:
        train_x, train_y, test_x, test_y, NUM_CLASS = loadMyData(opts)
    else:
        raise NotImplementedError()
    
    shuf = [i for i in range(len(train_x))]
    random.shuffle(shuf)
    train_x = train_x[shuf]
    train_y = train_y[shuf]

    # import pdb
    # pdb.set_trace()
    shuf = [i for i in range(len(test_x))]
    random.shuffle(shuf)
    test_x, test_y = test_x[shuf], test_y[shuf]

    # setup params
    Batch_Size = 64
    # ----------------------------------
    Epoch_Num = 100
    # ----------------------------------
    saveModelPath = os.path.join(modelDir, 'best_model_{}.h5'.format('mymodel'))
    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=saveModelPath, monitor='val_accuracy', verbose=1,
                                                      save_best_only=True,
                                                      mode='max')
    earlyStopper = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=25)
    
    import datetime
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


    callBackList = [checkpointer, earlyStopper, tensorboard_callback]

    print('get the model and compile it...')
    inp_shape = (train_x.shape[1], train_x.shape[2])
    # inp_shape = (train_x.shape[1])

    #import pdb 
    #pdb.set_trace()
    print('input shape: {}'.format(inp_shape))
    model = mymodel.mymodel4(inp_shape, NUM_CLASS)
    model.summary()
    # # optimize = optimizers.adam_v2.Adam(0.001)
    # optimize = tf.keras.optimizers.SGD(0.0001, momentum=0.9)
    # # optimize = tf.keras.optimizers.experimental.SGD(0.0001, momentum=0.9)
    # model.compile(optimizer=optimize, loss='categorical_crossentropy', metrics=['accuracy'])

    # print('fit the model with data...')
    # history=model.fit(x=train_x, y=train_y,
    #           batch_size=Batch_Size,
    #           epochs=Epoch_Num,
    #           # verbose=opts.verbose,
    #           verbose=2,
    #           callbacks=callBackList,
    #           validation_data=(test_x, test_y))
    #           # validation_split=0.1,
    #           # shuffle=True)

    # 编译模型
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
    print('fit the model with data...')
    # 训练模型
    history=model.fit(x=train_x, y=train_y, epochs=5, steps_per_epoch=3750, verbose=2,
              callbacks=callBackList,validation_data=(test_x, test_y))

    print('test the trained model...')
    score, acc = model.evaluate(test_x, test_y, batch_size=Batch_Size, verbose=2)
    print('test acc is: ', acc)

    print('all test done!')
    results = [opts.location, acc]

    with open('results.txt', 'a') as filehandle:
        # for listitem in results:
        # filehandle.write('%s' % listitem)
        filehandle.write('%f\n' % acc)
    
    # print(history)

    file_suffix = opts.input.split('/')[-2]

    #绘制混淆矩阵
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 6))
    epochs = history.epoch
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history['loss'], label="loss")
    plt.plot(epochs, history.history['val_loss'], label="val_loss")
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['accuracy'], label="accuracy")
    plt.plot(epochs, history.history['val_accuracy'], label="val_accuracy")
    plt.title('Accuracy')
    plt.legend()
    plt.show()
    plt.savefig('result_plot_{}.png'.format(file_suffix), bbox_inches='tight')

    import numpy as np
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    predictions = model.predict(test_x)
    plt.figure(figsize=(15, 15))
    test_y = [np.argmax(ele) for ele in test_y]
    predictions = [np.argmax(ele) for ele in predictions]
    cf_matrix = confusion_matrix(test_y, predictions,normalize='true')
    ax = sns.heatmap(cf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'],
                                                                 yticklabels=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'])
    ax.title.set_text("Confusion Matrix")
    ax.set_xlabel("y_pred")
    ax.set_ylabel("y_true")
    plt.savefig('confusion_matrix_{}.png'.format(file_suffix), bbox_inches='tight')


if __name__ == '__main__':
    opts = config.parse_args(sys.argv)
    main(opts)
