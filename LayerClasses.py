import numpy as np
from theano.tensor.nnet import conv
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
# from tensorflow.examples.tutorials.mnist import input_data
from theano.tensor import shared_randomstreams


class MyConvLayer(object):
    def __init__(self, rng, image_shape, filter_shape, border_mode = "valid", activation = T.tanh, params = [None, None]):
        self.image_shape = image_shape
        self.filter_shape = filter_shape
        self.output_shape = (image_shape[0],filter_shape[0],image_shape[2]-filter_shape[2]+1,image_shape[3]-filter_shape[3]+1)
        self.activation = activation
        self.border_mode = border_mode
        assert image_shape[1] == filter_shape[1]
        fan_in = np.prod(filter_shape[1:])
        fan_out = filter_shape[0] * np.prod(filter_shape[2:])
        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        if params[0] == None:
            self.W = theano.shared(np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX),
                                   borrow=True)
            self.b = theano.shared(value=b_values, borrow=True)
        else:
            self.W, self.b = params[0], params[1]

        self.params = [self.W, self.b]
        self.L1 = abs(self.W).sum()
        self.L2 = (self.W**2).sum()

    def set_input(self, input, input_dropout, mini_batch_size):
        self.input = input.reshape(self.image_shape)
        self.conv_out = conv.conv2d(input=self.input, filters=self.W, filter_shape=self.filter_shape, image_shape=self.image_shape, border_mode=self.border_mode)
        self.output = self.activation(self.conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output_dropout = self.output

class FullConectedLayer(object):
    def __init__(self, n_in, n_out, activation = T.tanh, p_dropout = 0.5,  params = [None,None]):
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        self.p_dropout = p_dropout
        if params[0] == None:
            self.W = theano.shared(value= np.asarray(np.random.rand(n_in,n_out)/np.sqrt(n_in+1),dtype=theano.config.floatX),
                                   name = "W",
                                   borrow=True)
            self.b = theano.shared(value= np.asarray(np.random.rand(n_out,) ,dtype=theano.config.floatX),
                                   name ="b",
                                   borrow=True
            )
        else:
            self.W, self.b = params[0], params[1]
        self.params = [self.W, self.b]
        self.L1 = abs(self.W).sum()
        self.L2 = (self.W**2).sum()

    def set_input(self, input, input_dropout, mini_batch_size):
        self.input = input.flatten(2)
        self.output = self.activation(T.dot(self.input,self.W) + self.b)
        self.inpt_dropout = dropout_layer(input_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation(T.dot(self.inpt_dropout, self.W) + self.b)

class SoftmaxLayer(object):
    def __init__(self , n_in, n_out, params=[None, None]):
        if params[0] == None:
            self.W = theano.shared(value= np.asarray(np.random.rand(n_in,n_out)/np.sqrt(n_in+1),dtype=theano.config.floatX),
                                   name = "W",
                                   borrow=True)
            self.b = theano.shared(value= np.asarray(np.random.rand(n_out,) ,dtype=theano.config.floatX),
                                   name ="b",
                                   borrow=True
            )
        else:
            self.W, self.b = params[0], params[1]
        self.n_in = n_in
        self.n_out = n_out
        # parameters of the model
        self.params = [self.W, self.b]
        self.L1 = abs(self.W).sum()
        self.L2 = (self.W**2).sum()

    def set_input(self, input):
        self.input = input.flatten(2)
        self.p_y_given_x = T.nnet.softmax(T.dot(self.input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.output = self.y_pred

    def negative_log_likelihood(self,y):
        return -T.mean(T.log(self.p_y_given_x)*y)

    def predict(self):
        return self.y_pred

    def error(self,y):
        y = T.argmax(y,1)
         # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


def dropout_layer(layer, p_dropout):
    srng = shared_randomstreams.RandomStreams(np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    return layer*T.cast(mask, theano.config.floatX)

def mask_k_maxpooling(variable, variable_shape ,axis, k):
    """
    Params:
    variable:   tensor2D
    axis:       get k_max_pooling in axis'th dimension
    k:          k loop  --> k max value
    ------
    Return:
    mask : tensor2D
        1: if in position k_max
        0: else
        ex variable:
            1 2 3              0 0 1
            2 7 1      --->    0 1 0
            1 2 1              0 1 0
    """
    min = -999999999

    variable_tmp = variable
    mask = T.zeros(variable_shape, dtype=theano.config.floatX)
    for i in range(k):
        max_idx = T.argmax(variable_tmp,axis=axis)
        if axis == 0:
            mask = T.set_subtensor(mask[max_idx,range(0,variable_shape[1])],1)
            variable_tmp = T.set_subtensor(variable_tmp[max_idx,range(0,variable_shape[1])],min)
        elif axis == 1:
            mask = T.set_subtensor(mask[range(0,variable_shape[0]),max_idx],1)
            variable_tmp = T.set_subtensor(variable_tmp[range(0,variable_shape[0]),max_idx],min)
    return mask

class MyConvPoolLayer(object):

    def __init__(self, rng, input, image_shape, filter_shape, k_pool_size, activation = T.tanh):

        self.input = input

        self.image_shape = image_shape

        self.filter_shape = filter_shape

        self.k_pool_size = k_pool_size

        assert image_shape[1] == filter_shape[1]

        fan_in = np.prod(filter_shape[1:])
        fan_out = filter_shape[0] * np.prod(filter_shape[2:])

        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(np.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)

        self.b = theano.shared(value=b_values, borrow=True)

        self.conv_out = conv.conv2d(input=input, filters=self.W, filter_shape=filter_shape, image_shape=image_shape, border_mode="valid")

        self.mask_input = self.conv_out.flatten(2)


        # loi cho nay
        shape_afconv = (image_shape[0],image_shape[1],image_shape[2]-filter_shape[2]+1,image_shape[3]-filter_shape[3]+1)

        self.mask_k_maxpooling_2D = mask_k_maxpooling(self.mask_input,(image_shape[0],shape_afconv[1]*shape_afconv[2]*shape_afconv[3]),axis=1,k=k_pool_size)

        self.mask_k_maxpooling_4D = self.mask_k_maxpooling_2D.reshape(shape_afconv)

        self.output = activation(self.mask_k_maxpooling_4D * self.conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        self.params = [self.W, self.b]

class MyUnPoolDeconvLayer(object):
    def __init__(self, rng, input, mask_k_maxpooling_4D, input_shape, filter_shape, activation = T.tanh):

        self.input = input
        # mask4D        (batch_size, n_chnnel. wifth, height
        self.mask_k_maxpooling_4D = mask_k_maxpooling_4D
        # input_shape: (batch_size, n_channel, width, height)   e.g: (1,20,24,24)
        self.input_shape = input_shape
        # filter_shape: (n_kenel, n_channel, width, height)     e.g: (1,20,5,5)
        self.filter_shape = filter_shape

        assert input_shape[1] == filter_shape[1]

        unpool_out =  input * mask_k_maxpooling_4D

        fan_in = np.prod(filter_shape[1:])

        fan_out = filter_shape[0] * np.prod(filter_shape[2:])

        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(np.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        self.conv_out = conv.conv2d(input=unpool_out, filters=self.W, filter_shape=filter_shape, image_shape=input_shape, border_mode="full")

        self.output = self.conv_out + self.b.dimshuffle('x', 0, 'x', 'x')

        # co su dung activation o day ko, vi sai activation se ve doan -1:1, nhieu luc gia tri cua vector tu > 1
        # self.ouput = activation(self.conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.params = [self.W, self.b]

class LenetConvPoolLayer(object):
    def __init__(self, rng, input, image_shape, filter_shape, poolsize , border_mode ='valid' , activation = T.tanh):
        self.input = input
        self.image_shape = image_shape
        self.filter_shape = filter_shape
        self.poolsize = poolsize

        assert image_shape[1] == filter_shape[1]
        self.input = input

        fan_in = np.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /
                   np.prod(poolsize))
        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(np.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)

        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

         # convolve input feature maps with filters
        self.conv_out = conv.conv2d(input=input, filters=self.W, filter_shape=filter_shape, image_shape=image_shape, border_mode=border_mode)

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=self.conv_out, ds=poolsize, ignore_border=True)

        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        self.params = [self.W, self.b]

if __name__ == "__main__":

    print("main")
    # mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
    # # mnist.train.images.shape  :   (55000, 784)
    # # mnist.train.labels        :   (55000) --> [list label ...]
    #
    # # next_images, next_labels = mnist.train.next_batch(100)
    # # tuple: images, label      :   (100, 784) , (100, 10)
    #
    # nkerns=[20, 50]
    # batch_size=100
    # rng = np.random.RandomState(23455)
    # # minibatch)
    # x = T.dmatrix('x')  # data, presented as rasterized images
    # y = T.dmatrix('y')  # labels, presented as 1D vector of [int] labels
    #
    # # construct the logistic regression class
    # # Each MNIST image has size 28*28
    # layer0_input = x.reshape((-1, 1, 28, 28))
    #
    # layer0 = LenetConvPoolLayer(rng, input=layer0_input,
    #         image_shape=(batch_size, 1, 28, 28),
    #         filter_shape=(nkerns[0], 1, 5, 5), poolsize=(2, 24))
    #
    # layer2_input = layer0.output.flatten(2)
    #
    # layer2 = FullConectedLayer(input=layer2_input, n_in=nkerns[0] * 12 * 1, n_out=500, activation=T.tanh)
    #
    # classifier = SoftmaxLayer(input=layer2.ouput, n_in=500, n_out=10)
    #
    # cost = classifier.negative_log_likelihood(y)
    #
    # error = classifier.error(y)
    #
    # params = layer0.params + layer2.params + classifier.params
    #
    # gparams = []
    # for param in params:
    #     gparam = T.grad(cost, param)
    #     gparams.append(gparam)
    #
    # updates = []
    # # given two list the zip A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4] of
    # # same length, zip generates a list C of same size, where each element
    # # is a pair formed from the two lists :
    # #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    # for param, gparam in zip(params, gparams):
    #     updates.append((param, param - 0.1 * gparam))
    #
    # train_model = theano.function(inputs=[x,y], outputs=[cost,error,layer0.output, layer2_input],updates=updates)
    #
    # counter = 0
    # best_valid_err = 100
    # early_stop = 20
    #
    # batch_size = 100
    #
    # epoch_i = 0
    #
    # while counter < early_stop:
    #     epoch_i +=1
    #     batch_number = int(mnist.train.labels.shape[0]/batch_size)
    #     for batch in range(batch_number):
    #         next_images, next_labels = mnist.train.next_batch(100)
    #         train_cost, train_error, layer0_out, layer2_in = train_model(next_images, next_labels)
    #         print layer0_out.shape, layer2_in.shape
    #         # print train_cost, train_error
    #     next_images, next_labels = mnist.validation.next_batch(100)
    #     valid_cost, valid_error,_,_ = train_model(next_images, next_labels)
    #     if best_valid_err > valid_error:
    #         best_valid_err = valid_error
    #         print "Epoch ",epoch_i, " Validation cost: ", valid_cost, " Validation error: " , valid_error ," ",counter , " __best__ "
    #         counter = 0
    #     else:
    #         counter +=1
    #         print "Epoch ",epoch_i, " Validation cost: ", valid_cost, " Validation error: " , valid_error ," ",counter



