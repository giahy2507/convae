__author__ = 'HyNguyen'

import theano
import theano.tensor as T
import numpy as np
from LayerClasses import MyConvLayer,FullConectedLayer,SoftmaxLayer
from tensorflow.examples.tutorials.mnist import input_data
from sys import stderr
import cPickle
import os
from scipy.misc import imread, imsave

if __name__ == "__main__":

    mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
    print >> stderr, "readed data"
    batch_size=100
    number_featuremaps = 20
    sentence_length = 28
    embed_size = 28
    learning_rate = 0.1

    image_shape = (batch_size,1,sentence_length,embed_size)

    filter_shape_encode = (20,1,5,28)
    filter_shape_decode = (1,20,5,28)

    rng = np.random.RandomState(23455)
    params_save = [None]*8

    if os.path.isfile("saveweight.bin"):
        with open("saveweight.bin",mode="rb") as f:
            params_save = cPickle.load(f)

    # minibatch)
    X = T.dmatrix("X")  # data, presented as rasterized images
    Y = T.dmatrix("Y")

    layer0_encode_input = X.reshape((batch_size, 1, 28, 28))
    layer0_encode = MyConvLayer(rng,layer0_encode_input,image_shape=image_shape,filter_shape=filter_shape_encode,border_mode="valid",activation = T.nnet.sigmoid, params=params_save[0:2])

    layer1_encode_input = layer0_encode.output.flatten(2)
    layer1_encode_input_shape = (batch_size,layer0_encode.output_shape[1] * layer0_encode.output_shape[2] * layer0_encode.output_shape[3])
    layer1_encode = FullConectedLayer(layer1_encode_input,layer1_encode_input_shape[1],100,activation = T.nnet.sigmoid, params=params_save[2:4])

    layer_hidden = FullConectedLayer(input=layer1_encode.output, n_in=100, n_out=50, activation= T.nnet.sigmoid)

    layer_classification =  SoftmaxLayer(input=layer_hidden.output, n_in=50, n_out=10)

    err = layer_classification.error(Y)

    cost = layer_classification.negative_log_likelihood(Y) + 0.001*(layer_classification.L2 + layer_hidden.L2)

    params = layer_hidden.params + layer_classification.params

    gparams = []

    for param in params:
        gparam = T.grad(cost, param)
        gparams.append(gparam)
    updates = []
    for param, gparam in zip(params, gparams):
        updates.append((param, param - learning_rate* gparam))

    train_model = theano.function(inputs=[X,Y], outputs=[cost, err], updates=updates)
    valid_model = theano.function(inputs=[X,Y], outputs=[cost, err])
    predict_function = theano.function(inputs=[X], outputs=layer_classification.y_pred)

    counter = 0
    best_valid_err = 100
    early_stop = 50

    epoch_i = 0

    while counter < early_stop:
        epoch_i +=1
        batch_number = int(mnist.train.labels.shape[0]/batch_size)
        train_costs = []
        train_errs = []
        for batch in range(batch_number):
            next_images, next_labels = mnist.train.next_batch(batch_size)
            train_cost, train_err = train_model(next_images, next_labels)
            train_costs.append(train_cost)
            train_errs.append(train_err)
        #print >> stderr, "batch "+str(batch)+" Train cost: "+ str(train_cost)
        next_images, next_labels = mnist.validation.next_batch(batch_size)
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










