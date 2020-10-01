import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

def main():
    data = pd.read_csv("E202-COMP7117-TD01-00 - classification.csv")
    feature = np.array([data.iloc[:, 1], data.iloc[:, 4], data.iloc[:, 5], data.iloc[:, 6], data.iloc[:, 7], data.iloc[:, 8]
                        , data.iloc[:, 9], data.iloc[:, 10]]).transpose()
    
    for i in range(1599):
        if (feature[i,2] == 'High'):
            feature[i,2] = 3
        elif (feature[i,2] == 'Medium'):
            feature[i,2] = 2
        elif (feature[i,2] == 'Low'):
            feature[i,2] = 1
        else: 
            feature[i,2] = 0

    for i in range(1599):
        if (feature[i,4] == 'Very High'):
            feature[i,4] = 0
        elif (feature[i,4] == 'High'):
            feature[i,4] = 3
        elif (feature[i,4] == 'Medium'):
            feature[i,4] = 2
        elif (feature[i,4] == 'Low'):
            feature[i,4] = 1

    for i in range(1599):
        if (feature[i,5] == 'Very Basic'):
            feature[i,5] = 3
        elif (feature[i,5] == 'Normal'):
            feature[i,5] = 2
        elif (feature[i,5] == 'Very Acidic'):
            feature[i,5] = 1
        else:
            feature[i,5] = 0

    pca = PCA(n_components = 4)
    pca = pca.fit(feature)
    result = pca.transform(feature)
    
    target = data[['quality']]
    one_hot = OneHotEncoder(sparse=False)
    target = one_hot.fit_transform(target)
    feat_train, feat_test, tar_train, tar_test = train_test_split(result, target, test_size=.1)
    feat_train, feat_val, tar_train, tar_val = train_test_split(feat_train, tar_train, test_size=.22)

    scaler = MinMaxScaler().fit(feat_train)

    feat_train = scaler.transform(feat_train)
    feat_test = scaler.transform(feat_test)

    layer = {
        'input': 4,
        'hidden': 10,
        'output': 5
    }

    weight = {
        'input-hidden': tf.Variable(tf.random_normal([layer['input'], layer ['hidden']])),
        'hidden-output': tf.Variable(tf.random_normal([layer['hidden'], layer ['output']]))
    }

    bias = {
        'input-hidden': tf.Variable(tf.random_normal([layer ['hidden']])),
        'hidden-output': tf.Variable(tf.random_normal([layer ['output']]))
    }

    feat_input = tf.placeholder(tf.float32, [None, layer['input']])
    tar_input = tf.placeholder(tf.float32, [None, layer['output']])
    
    saver = tf.train.Saver()

    def predict():
        y1 = tf.matmul(feat_input, weight['input-hidden']) + bias['input-hidden']
        
        y1 = tf.sigmoid(y1)

        y2 = tf.matmul(y1, weight['hidden-output']) + bias['hidden-output']

        return tf.sigmoid(y2)

    y_predict = predict()
    lr = .1

    loss = tf.reduce_mean(.5 * (tar_input - y_predict) **2 )

    optimizer = tf.train.GradientDescentOptimizer(lr)
    train = optimizer.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        epoch = 5000
        feed_train = {
            feat_input: feat_train,
            tar_input: tar_train
        }

        feed_val= {
            feat_input: feat_val,
            tar_input: tar_val
        }

        feed_val_1={
            feat_input: feat_val,
            tar_input: tar_val
        }
 
        for i in range(1, epoch + 1):
            sess.run(train, feed_dict = feed_train)

            if i % 100 == 0:
                result_loss = sess.run(loss, feed_dict = feed_train)
                print("epoch: {}    loss:{}" .format(i, result_loss))

            if i == 500:
                validation_loss_1 = sess.run(loss, feed_dict = feed_val)
                print("Validation Error:  {}" .format(validation_loss_1))
                saver.save(sess, save_path = "Classification/model.ckpt")
            elif i % 500 == 0:
                validation_loss_2 = sess.run(loss, feed_dict = feed_val_1)
                print("Validation Error:  {}" .format(validation_loss_2))
                if(validation_loss_2 < validation_loss_1):
                    saver.save(sess, save_path = "Classification/model.ckpt")

        feed_test = {
            feat_input: feat_test,
            tar_input: tar_test
        }

        match = tf.equal(tf.argmax(tar_input, axis = 1), tf.argmax(y_predict, axis = 1))
        accuracy = tf.reduce_mean(tf.cast(match, tf.float32))
        accuracy *= 100
        print("Accuracy: {}%" .format(sess.run(accuracy, feed_dict = feed_test)))


main()