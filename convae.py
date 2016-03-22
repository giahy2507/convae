__author__ = 'HyNguyen'

import theano
import theano.tensor as T
import numpy as np
from LayerClasses import MyConvLayer,FullConectedLayer
import cPickle
import os
import sys
from lasagne.updates import adam, sgd

def load_np_data(path, onehost = False):
    data = np.load(path)

    valid_size = int(data.shape[0]*0.8)
    shape = (-1, data.shape[1]*data.shape[2])

    X_train = data[:valid_size]
    X_valid = data[valid_size:]
    return X_train.reshape(shape), X_valid.reshape(shape)


class ConvolutionAutoEncoder(object):
    def __init__(self, layers, mini_batch_size, params = None, name = "CAE"):
        self.name = name
        self.layers = layers
        self.mini_batch_size = mini_batch_size

        if params is None:
            self.params = [param for layer in self.layers for param in layer.params]
        else:
            self.params = params
            for i in range(len(self.layers)):
                self.layers[i].w = params[i*2]
                self.layers[i].b = params[i*2 + 1]
        self.X = T.matrix("X")
        init_layer = self.layers[0]
        init_layer.set_input(self.x, self.mini_batch_size)
        for j in xrange(1, len(self.layers)):
            prev_layer, layer  = self.layers[j-1], self.layers[j]
            layer.set_inpt(prev_layer.output, self.mini_batch_size)
        self.output = self.layers[-1].output

    def train(self, X_train, X_valid, early_stop_count = 20, save_model_path = "model/saveweight.bin" , X_test = None):

        l2_norm_squared = sum([layer.L2 for layer in self.layers])
        mae = T.mean(T.sqrt(T.sum(T.sqr(self.layers[-1].output.flatten(2) - X), axis=1)), axis=0)
        cost = mae + l2_norm_squared
        updates = adam(cost,self.params)

        train_model = theano.function(inputs=[self.X], outputs=[cost, mae], updates=updates)
        valid_model = theano.function(inputs=[self.X], outputs=[cost, mae])

        num_training_batches = int(X_train.shape[0] / self.mini_batch_size)
        num_validation_batches = int(X_valid.shape[0] / self.mini_batch_size)

        counter = 0
        best_valid_err = 100
        early_stop = early_stop_count
        epoch_i = 0

        train_rand_idxs = list(range(0, X_train.shape[0]))
        valid_rand_idxs = list(range(0, X_valid.shape[0]))

        while counter < early_stop:
            epoch_i +=1
            train_costs = []
            train_errs = []

            valid_costs = []
            valid_errs = []

            np.random.shuffle(train_rand_idxs)
            for batch_i in range(num_training_batches):
                mnb_X = X_train[train_rand_idxs[batch_i*self.mini_batch_size: batch_i*self.mini_batch_size + self.mini_batch_size]]
                train_cost, train_err = train_model(mnb_X)
                train_costs.append(train_cost)
                train_errs.append(train_err)

            np.random.shuffle(valid_rand_idxs)
            for batch_i in range(num_validation_batches):
                mnb_X = X_train[train_rand_idxs[batch_i*self.mini_batch_size: batch_i*self.mini_batch_size + self.mini_batch_size]]
                valid_cost, valid_err = valid_model(mnb_X)
                valid_costs.append(valid_cost)
                valid_errs.append(valid_err)

            train_err = np.mean(np.array(train_errs))
            train_cost = np.mean(np.array(train_costs))
            val_err = np.mean(np.array(valid_errs))
            val_cost = np.mean(np.array(valid_costs))

            if best_valid_err > val_err:
                best_valid_err = val_err
                sys.stdout.write("Epoch "+str(epoch_i)+" Train cost: "+ str(train_cost)+ "Train mae: "+ str(train_err) + " Validation cost: "+ str(val_cost)+" Validation mae "+ str(val_err)  + ",counter "+str(counter)+ " __best__ \n")
                sys.stdout.flush()
                counter = 0
                with open(save_model_path, mode="wb") as f:
                    cPickle.dump(params,f)
            else:
                counter +=1
                sys.stdout.write("Epoch " + str(epoch_i)+" Train cost: "+ str(train_cost)+ "Train mae: "+ str(train_err) + " Validation cost: "+ str(val_cost)+" Validation mae "+ str(val_err)  + ",counter "+str(counter) + "\n")
                sys.stdout.flush()








if __name__ == "__main__":

    # mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

    batch_size=100
    number_featuremaps = 20
    sentence_length = 100
    embed_size = 100
    learning_rate = 0.1

    image_shape = (batch_size,1,sentence_length,embed_size)

    filter_shape_encode = (20,1,5,embed_size)
    filter_shape_decode = (1,20,5,embed_size)

    rng = np.random.RandomState(23455)
    params_save = [None]*8

    if os.path.isfile("model/saveweight.bin"):
        with open("model/saveweight.bin",mode="rb") as f:
            params_save = cPickle.load(f)

    # minibatch)
    X = T.dmatrix('X')  # data, presented as rasterized images

    layer0_encode_input = X.reshape((batch_size, 1, sentence_length, embed_size))
    layer0_encode = MyConvLayer(rng,layer0_encode_input,image_shape=image_shape,filter_shape=filter_shape_encode,border_mode="valid",activation = T.tanh, params=params_save[0:2])

    layer1_encode_input = layer0_encode.output.flatten(2)
    layer1_encode_input_shape = (batch_size,layer0_encode.output_shape[1] * layer0_encode.output_shape[2] * layer0_encode.output_shape[3])
    layer1_encode = FullConectedLayer(layer1_encode_input,layer1_encode_input_shape[1], 100, activation = T.tanh, params=params_save[2:4])

    layer1_decode_input_shape = (batch_size, 100)
    layer2_decode_input = layer1_encode.output
    layer2_decode = FullConectedLayer(layer2_decode_input, 100,layer1_encode_input_shape[1],activation = T.tanh,  params=params_save[4:6])

    layer3_decode_input = layer2_decode.output.reshape(layer0_encode.output_shape)
    layer3_decode = MyConvLayer(rng,layer3_decode_input, image_shape=layer0_encode.output_shape, filter_shape=filter_shape_decode, border_mode="full",activation = T.tanh,  params=params_save[6:8])

    mae = T.mean(T.sqrt(T.sum(T.sqr(layer3_decode.output.flatten(2) - X), axis=1)), axis=0)
    
    cost = mae + 0.001*(layer0_encode.L2 + layer1_encode.L2 + layer2_decode.L2 + layer3_decode.L2)

    params = layer0_encode.params + layer1_encode.params + layer2_decode.params + layer3_decode.params

    # Adam Optimizier
    updates = adam(cost,params)

    # SGD
    # updates = sgd(cost,params,0.1)

    # SGD Optimizier
    # gparams = []
    # for param in params:
    #     gparam = T.grad(cost, param)
    #     gparams.append(gparam)
    # updates = []
    # for param, gparam in zip(params, gparams):
    #     updates.append((param, param - learning_rate* gparam))

    train_model = theano.function(inputs=[X], outputs=[cost, mae],updates=updates)
    valid_model = theano.function(inputs=[X], outputs=[cost, mae])
    show_function = theano.function(inputs=[X], outputs=layer3_decode.output)
    predict_function = theano.function(inputs=[X], outputs=layer3_decode.output.flatten(2))

    counter = 0
    best_valid_err = 100
    early_stop = 20
    epoch_i = 0

    X_train, X_valid = load_np_data("vector/data_processed.npy")

    sys.stdout.write("Read data \n")

    train_rand_idxs = list(range(0, X_train.shape[0]))
    valid_rand_idxs = list(range(0, X_valid.shape[0]))

    X_test = X_train[:batch_size]
    X_pred = show_function(X_test)
    for i in range(batch_size):
        A = X_test[i].reshape(100,100)
        B = X_pred[i].reshape(100,100)
        print np.sqrt(np.sum(np.square(A - B)))

    #
    # while counter < early_stop:
    #     epoch_i +=1
    #     train_costs = []
    #     train_errs = []
    #
    #     valid_costs = []
    #     valid_errs = []
    #
    #     np.random.shuffle(train_rand_idxs)
    #     for start_idx in range(0, X_train.shape[0], batch_size):
    #         if start_idx + batch_size > X_train.shape[0]:
    #             break
    #         mnb_X = X_train[train_rand_idxs[start_idx: start_idx + batch_size]]
    #
    #         train_cost, train_err = train_model(mnb_X)
    #         train_costs.append(train_cost)
    #         train_errs.append(train_err)
    #
    #     np.random.shuffle(valid_rand_idxs)
    #     for start_idx in range(0, X_valid.shape[0], batch_size):
    #         if start_idx + batch_size > X_valid.shape[0]:
    #             break
    #         mnb_X = X_valid[valid_rand_idxs[start_idx: start_idx + batch_size]]
    #         valid_cost, valid_err = valid_model(mnb_X)
    #         valid_costs.append(valid_cost)
    #         valid_errs.append(valid_err)
    #
    #     train_err = np.mean(np.array(train_errs))
    #     train_cost = np.mean(np.array(train_costs))
    #     val_err = np.mean(np.array(valid_errs))
    #     val_cost = np.mean(np.array(valid_costs))
    #
    #     if best_valid_err > val_err:
    #         best_valid_err = val_err
    #         sys.stdout.write("Epoch "+str(epoch_i)+" Train cost: "+ str(train_cost)+ "Train mae: "+ str(train_err) + " Validation cost: "+ str(val_cost)+" Validation mae "+ str(val_err)  + ",counter "+str(counter)+ " __best__ \n")
    #         sys.stdout.flush()
    #         counter = 0
    #         with open("model/saveweight.bin", mode="wb") as f:
    #             cPickle.dump(params,f)
    #     else:
    #         counter +=1
    #         sys.stdout.write("Epoch " + str(epoch_i)+" Train cost: "+ str(train_cost)+ "Train mae: "+ str(train_err) + " Validation cost: "+ str(val_cost)+" Validation mae "+ str(val_err)  + ",counter "+str(counter) + "\n")
    #         sys.stdout.flush()








