__author__ = 'HyNguyen'
import theano
import theano.tensor as T
import numpy as np
from LayerClasses import MyConvLayer,FullConectedLayer,SoftmaxLayer
from tensorflow.examples.tutorials.mnist import input_data
import cPickle
import os
import sys
from lasagne.updates import adam
from sys import stderr


if __name__ == "__main__":

    mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

    X = T.dmatrix("X")  # data, presented as rasterized images
    Y = T.dmatrix("Y")
    mini_batch_size = 100
    filter_shape_encode = (20,1,5,5)
    filter_shape_decode = (1,20,5,5)
    rng = np.random.RandomState(23455)

    layer0_encode_input = X.reshape((mini_batch_size, 1, 28, 28))
    layer0_encode = MyConvLayer(rng,layer0_encode_input,image_shape=(mini_batch_size, 1, 28, 28),filter_shape=filter_shape_encode,border_mode="valid")

    layer1_input = layer0_encode.output.flatten(2)
    n_in = layer0_encode.output_shape[1] * layer0_encode.output_shape[2] * layer0_encode.output_shape[3]
    layer1 = FullConectedLayer(layer1_input,n_in,100)
    layer_classification =  SoftmaxLayer(input=layer1.output, n_in=100, n_out=10)


    err = layer_classification.error(Y)
    cost = layer_classification.negative_log_likelihood(Y) + 0.001*(layer0_encode.L2 + layer_classification.L2 + layer1.L2)
    params = layer0_encode.params + layer1.params + layer_classification.params

    updates = adam(cost,params)

    train_model = theano.function(inputs=[X,Y], outputs=[cost, err], updates=updates)
    valid_model = theano.function(inputs=[X,Y], outputs=[cost, err])
    predict_function = theano.function(inputs=[X], outputs=layer_classification.y_pred)


    counter = 0
    best_valid_err = 100
    early_stop = 20

    epoch_i = 0

    while counter < early_stop:
        epoch_i +=1
        batch_number = int(mnist.train.labels.shape[0]/mini_batch_size)
        train_costs = []
        train_errs = []
        for batch in range(batch_number):
            next_images, next_labels = mnist.train.next_batch(mini_batch_size)
            train_cost, train_err = train_model(next_images, next_labels)
            train_costs.append(train_cost)
            train_errs.append(train_err)
        #print >> stderr, "batch "+str(batch)+" Train cost: "+ str(train_cost)
        next_images, next_labels = mnist.validation.next_batch(mini_batch_size)
        valid_cost, val_err = valid_model(next_images, next_labels)
        if best_valid_err > val_err:
            best_valid_err = val_err
            print >> stderr, "Epoch "+str(epoch_i)+" Train cost: "+ str(np.mean(np.array(train_costs)))+ "Train mae: "+ str(np.mean(np.array(train_errs))) + " Validation cost: "+ str(valid_cost)+" Validation mae "+ str(val_err)  + ",counter "+str(counter)+ " __best__ "
            counter = 0
            with open("saveweight_caeclassification.bin", mode="wb") as f:
                cPickle.dump(params,f)
        else:
            counter +=1
            print >> stderr, "Epoch "+str(epoch_i)+" Train cost: "+ str(np.mean(np.array(train_costs)))+ "Train mae: "+ str(np.mean(np.array(train_errs))) + " Validation cost: "+ str(valid_cost)+" Validation mae "+ str(val_err)  + ",counter "+str(counter)

