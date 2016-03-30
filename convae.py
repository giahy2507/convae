__author__ = 'HyNguyen'

import theano
import theano.tensor as T
import numpy as np
from LayerClasses import MyConvLayer,FullConectedLayer
import cPickle
import os
import sys
# from lasagne.updates import adam,rmsprop,adadelta

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
        self.X = T.dmatrix("X")
        init_layer = self.layers[0]
        init_layer.set_input(self.X, self.X, self.mini_batch_size)
        for j in xrange(1, len(self.layers)):
            prev_layer, layer  = self.layers[j-1], self.layers[j]
            layer.set_input(prev_layer.output, prev_layer.output_dropout , self.mini_batch_size)
        self.output = self.layers[-1].output
        self.vector_sentence = self.layers[int(len(self.layers)/2)].input
        self.showfunction = theano.function([self.X], outputs=self.output)
        self.get_vector_function = theano.function([self.X], outputs=self.vector_sentence)

    def show(self, X_train):
        return self.showfunction(X_train)

    def load(self, filemodel = "model/CAE.model"):
        with open(filemodel, mode="rb") as f:
            self.params = cPickle.load(f)
            for i in range(len(self.layers)):
                self.layers[i].w = self.params[i*2]
                self.layers[i].b = self.params[i*2 + 1]

    def save(self, filemodel = "model/CAE.model"):
        with open(filemodel, mode="wb") as f:
            cPickle.dump(self.params,f)

    @classmethod
    def rebuild_for_testing(self, mini_batch_size, filemodel = "model/CAE.model" ):
        mini_batch_size=mini_batch_size
        number_featuremaps = 20
        sentence_length = 100
        embed_size = 100
        image_shape = (mini_batch_size,1,sentence_length,embed_size)
        filter_shape_encode = (number_featuremaps,1,5,embed_size)
        filter_shape_decode = (1,number_featuremaps,5,embed_size)
        rng = np.random.RandomState(23455)
        layer1 = MyConvLayer(rng,image_shape=image_shape,filter_shape=filter_shape_encode, border_mode="valid")
        layer2 = FullConectedLayer(n_in=layer1.output_shape[1] * layer1.output_shape[2] * layer1.output_shape[3],n_out=100)
        layer3 = FullConectedLayer(n_in=layer2.n_out, n_out=layer2.n_in)
        layer4 = MyConvLayer(rng,image_shape=layer1.output_shape, filter_shape=filter_shape_decode,border_mode="full")
        layers = [layer1,layer2,layer3,layer4]
        cae = ConvolutionAutoEncoder(layers,mini_batch_size)
        cae.load(filemodel)
        return cae

    def train(self, X_train, X_valid, early_stop_count = 20 , X_test = None):

        l2_norm_squared = 0.001*sum([layer.L2 for layer in self.layers])
        mae = T.mean(T.sqrt(T.sum(T.sqr(self.layers[-1].output.flatten(2) - self.X), axis=1)), axis=0)
        cost = mae + l2_norm_squared
        updates = adadelta(cost,self.params)
        # updates = adam(cost, self.params)

        self.train_model = theano.function(inputs=[self.X], outputs=[cost, mae], updates=updates)
        self.valid_model = theano.function(inputs=[self.X], outputs=[cost, mae])

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
                train_cost, train_err = self.train_model(mnb_X)
                train_costs.append(train_cost)
                train_errs.append(train_err)

            np.random.shuffle(valid_rand_idxs)
            for batch_i in range(num_validation_batches):
                mnb_X = X_train[train_rand_idxs[batch_i*self.mini_batch_size: batch_i*self.mini_batch_size + self.mini_batch_size]]
                valid_cost, valid_err = self.valid_model(mnb_X)
                valid_costs.append(valid_cost)
                valid_errs.append(valid_err)

            train_err = np.mean(np.array(train_errs))
            train_cost = np.mean(np.array(train_costs))
            val_err = np.mean(np.array(valid_errs))
            val_cost = np.mean(np.array(valid_costs))

            if val_err < best_valid_err:
                best_valid_err = val_err
                sys.stdout.write("Epoch "+str(epoch_i)+" Train cost: "+ str(train_cost)+ "Train mae: "+ str(train_err) + " Validation cost: "+ str(val_cost)+" Validation mae "+ str(val_err)  + ",counter "+str(counter)+ " __best__ \n")
                sys.stdout.flush()
                counter = 0
                with open("model/" + self.name +".model", mode="wb") as f:
                    cPickle.dump(self.params,f)
            else:
                counter +=1
                sys.stdout.write("Epoch " + str(epoch_i)+" Train cost: "+ str(train_cost)+ "Train mae: "+ str(train_err) + " Validation cost: "+ str(val_cost)+" Validation mae "+ str(val_err)  + ",counter "+str(counter) + "\n")
                sys.stdout.flush()


if __name__ == "__main__":

    mini_batch_size=100
    number_featuremaps = 20
    sentence_length = 100
    embed_size = 100
    image_shape = (mini_batch_size,1,sentence_length,embed_size)
    filter_shape_encode = (20,1,5,embed_size)
    filter_shape_decode = (1,20,5,embed_size)
    rng = np.random.RandomState(23455)
    X_train, X_valid = load_np_data("vector/data_processed.npy")
    print("X_train.shape: ", X_train.shape)
    # layer1 = MyConvLayer(rng,image_shape=image_shape,filter_shape=filter_shape_encode, border_mode="valid")
    # layer2 = FullConectedLayer(n_in=layer1.output_shape[1] * layer1.output_shape[2] * layer1.output_shape[3],n_out=100)
    # layer3 = FullConectedLayer(n_in=layer2.n_out, n_out=layer2.n_in)
    # layer4 = MyConvLayer(rng,image_shape=layer1.output_shape, filter_shape=filter_shape_decode,border_mode="full")
    # layers = [layer1,layer2,layer3,layer4]
    # cae = ConvolutionAutoEncoder(layers, mini_batch_size)
    # cae.train(X_train,X_valid)










