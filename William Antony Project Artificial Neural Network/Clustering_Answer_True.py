import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

class SOM:
    def __init__(self, height, width, input_dimension):
        self.height = height
        self.width = width
        self.input_dimension = input_dimension
        self.location = [tf.to_float([y,x]) for y in range(height) for x in range(width)]
        self.weight = tf.Variable(tf.random_normal([width*height, input_dimension]))
        self.input = tf.placeholder(tf.float32, [input_dimension])

        best_matching_unit = self.get_bmu()

        self.updated_weight, self.rate_stacked = self.update_neighbour(best_matching_unit)
    
    def get_bmu(self):
        square_difference = tf.square(self.input - self.weight)
        distance = tf.sqrt(tf.reduce_mean(square_difference, axis=1))

        bmu_index = tf.argmin(distance)
        bmu_location = tf.to_float([tf.div(bmu_index, self.width) , tf.mod(bmu_index, self.width)])

        return bmu_location

    def update_neighbour(self, bmu):
        learning_rate = 0.1

        sigma = tf.to_float(tf.maximum(self.width, self.height)/2)

        square_difference = tf.square(self.location - bmu)
        distance = tf.sqrt(tf.reduce_mean(square_difference, axis=1))

        neighbour_strength = tf.exp(tf.div(tf.negative(tf.square(distance)), 2 * tf.square(sigma)))

        rate = neighbour_strength * learning_rate
        total_node = self.width * self.height
        rate_stacked = tf.stack([tf.tile(tf.slice(rate, [i], [1]), [self.input_dimension]) for i in range(total_node)])

        input_weight_difference = tf.subtract(self.input, self.weight)

        weight_difference = tf.multiply(rate_stacked, input_weight_difference)

        weight_new = tf.add(self.weight, weight_difference)

        return tf.assign(self.weight, weight_new), rate_stacked

    def train(self, dataset, num_of_epoch):
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            for i in range(num_of_epoch):
                print("epoch {}".format(i))
                for data in dataset:
                    sess.run(self.updated_weight, feed_dict={ self.input: data })
            
            cluster = [[] for i in range(self.height)]
            location = sess.run(self.location)
            weight = sess.run(self.weight)

            for i, loc in enumerate(location):
                print(i, loc[0])
                cluster[int(loc[0])].append(weight[i])
            
            self.cluster = cluster

def main():
    data = pd.read_csv("E202-COMP7117-TD01-00 - clustering.csv")
    raw_data = data[['SpecialDay', 'VisitorType', 'Weekend', 'ProductRelated', 'ExitRates']]

    for i in range(3632):
        if(raw_data.loc[i, 'SpecialDay'] == "HIGH"):
            raw_data.loc[i, 'SpecialDay'] = 2
        elif(raw_data.loc[i, 'SpecialDay'] == "NORMAL"):
            raw_data.loc[i, 'SpecialDay'] = 1
        elif(raw_data.loc[i, 'SpecialDay'] == "LOW"):
            raw_data.loc[i, 'SpecialDay'] = 0

    for i in range(3632):
        if(raw_data.loc[i, 'VisitorType'] == "Returning_Visitor"):
            raw_data.loc[i, 'VisitorType'] = 2
        elif(raw_data.loc[i, 'VisitorType'] == "New_Visitor"):
            raw_data.loc[i, 'VisitorType'] = 1
        elif(raw_data.loc[i, 'VisitorType'] == "Other"):
            raw_data.loc[i, 'VisitorType'] = 0

    for i in range(3632):
        if(raw_data.loc[i, 'Weekend'] == True):
            raw_data.loc[i, 'Weekend'] = 1
        elif(raw_data.loc[i, 'Weekend'] == False):
            raw_data.loc[i, 'Weekend'] = 0
    
    pca = PCA(n_components=3)
    pca = pca.fit(raw_data)
    result = pca.transform(raw_data)

    print(result.shape)
    print(result)

    height = 5
    width = 5
    input_dimension = 3
    som = SOM(height, width, input_dimension)

    som.train(result, 5000)

    plt.imshow(som.cluster)
    plt.show()

main()