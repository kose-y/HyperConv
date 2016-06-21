import cPickle
import gzip
import os
import sys
import time

import numpy
import scipy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression
from mlp import HiddenLayer, relu
import cPickle

def load_mat_patch(patchsize = 1, datfile='KSC_norm.mat',datvar='KSC_norm',labelfile='KSC_gt.mat',labelvar='KSC_gt', samplevar=None, clip=1300, load_all=False):
    ''' load the dataset
    
    :type patchsize: int
    :param patchsize: size of patch to load. 0 for 1x1, 1 for 3x3, 2 for 5x5, 3 for 7x7, etc. 
    
    :type datfile: string
    :param datafile: the path of mat file that  contains the image data
    
    :type datvar: string
    :param datvar: the variable name of image data in the mat file
    
    :type labelfile: string
    :param labelfile: the path of mat file that contains label data
    
    :type labelvar: string
    :param labelvar: the variable name of label data in the mat file
    
    :type load_all: bool
    :param load_all: whether to load the whole image
    '''

    print '...loading data'
    mat = scipy.io.loadmat(datfile)[datvar]
    #mat = numpy.minimum(mat,65535.) / 65535. #normalization
    #mat = mat
    label = scipy.io.loadmat(labelfile)[labelvar]
    
    if not samplevar:
        nzidx = numpy.transpose(numpy.nonzero(label))
        #shuffle nonzero indices
        numpy.random.shuffle(nzidx)
        
        #do something for sample index if necessary
    else: # we already have shuffled order of nonzero indices in labelfile
        nzidx = scipy.io.loadmat(labelfile)[samplevar-1]
    print mat.shape
    print nzidx.shape
    
    xpos = nzidx[:,0]
    ypos = nzidx[:,1] 
    mat_small = mat[(xpos[0]-patchsize):(xpos[0]+patchsize+1), (ypos[0]-patchsize):(ypos[0]+patchsize+1),:]
    
    for i in xrange(1,nzidx.shape[0]):
        mat_small_new = mat[(xpos[i]-patchsize):(xpos[i]+patchsize+1), (ypos[i]-patchsize):(ypos[i]+patchsize+1),:]
        mat_small = numpy.concatenate((mat_small, mat_small_new), axis=2)
        #print mat_small.shape
    #mat_small = mat[(xpos-patchsize):(xpos+patchsize+1), (ypos-patchsize):(ypos+patchsize+1)]
    patchwidth = 2 * patchsize+1
    mat_small = mat_small.reshape((patchwidth,patchwidth,nzidx.shape[0],mat.shape[2]))
    mat_small = numpy.transpose(mat_small, (2,3,0,1))
    print mat_small.shape

    mat_small = mat_small.reshape(nzidx.shape[0], mat.shape[2]*patchwidth**2 )
    label_small = label[nzidx[:,0], nzidx[:,1]]-1
    
    n_train = int(numpy.floor(mat_small.shape[0] * 0.6))
    n_valid = int(numpy.floor(mat_small.shape[0] * 0.2))
    n_test  = mat_small.shape[0] - n_train - n_valid

    #all_set = mat.reshape(mat.shape[0]*mat.shape[1], mat.shape[2])
    #numpy.random.shuffle(all_set)
    #all_label = label#.reshape(-1)#dummy
    #all_set = (all_set[:all_set.shape[0]/100], all_label)

    train_set = mat_small[:n_train], label_small[:n_train]
    valid_set = mat_small[n_train:n_train+n_valid], label_small[n_train:n_train+n_valid]
    test_set  = mat_small[n_train+n_valid:], label_small[n_train + n_valid:]
    print numpy.max(train_set[1])

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    #all_set_x, all_set_y = shared_dataset(all_set)
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)
    #print all_set_x.shape
    print train_set_x.shape
    print test_set_x.shape
    print valid_set_x.shape

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y),]# (all_set_x, all_set_y) ]
    return rval

class ConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2), activation=T.tanh):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

    :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))

        if activation ==relu:
            self.W = theano.shared(numpy.asarray(
            rng.normal(0, 0.1, size=filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)
        else:

            self.W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)


        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape)

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=poolsize, ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = activation(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]


def evaluate_cnns(data = None, learning_rate=0.01, n_epochs=20000,
                    dataset='mnist.pkl.gz',
                    nkerns=[500], n_hidden = 300, batch_size=60, out=None, activation=T.tanh, save_name=""):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    rng = numpy.random.RandomState(23453)
    if not data:
        datasets = load_mat_patch()
    else:
        datasets = data

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    
    valid_batch_size = n_valid_batches
    test_batch_size  = n_test_batches
    
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ishape = (3, 3)  # this is the size of MNIST images

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size,28*28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 176, 3, 3))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
    # maxpooling reduces this further to (24/2,24/2) = (12,12)
    # 4D output tensor is thus of shape (batch_size,nkerns[0],12,12)
    layer0 = ConvPoolLayer(rng, input=layer0_input,
            image_shape=(batch_size, 176, 3, 3),
            filter_shape=(nkerns[0], 176, 2, 2), poolsize=(2, 2), activation=activation)

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
    # maxpooling reduces this further to (8/2,8/2) = (4,4)
    # 4D output tensor is thus of shape (nkerns[0],nkerns[1],4,4)
#    layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
#            image_shape=(batch_size, nkerns[0], 12, 12),
#            filter_shape=(nkerns[1], nkerns[0], 5, 5), poolsize=(2, 2))

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (20,32*4*4) = (20,512)
#    layer2_input = layer1.output.flatten(2)
    layer2_input = layer0.output.flatten(2)
    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(rng, input=layer2_input, n_in=nkerns[0] * 1 * 1 ,
                         n_out=n_hidden, activation=activation)

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=n_hidden, n_out=13)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function([index], layer3.errors(y),
             givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]})

    validate_model = theano.function([index], layer3.errors(y),
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size]})

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params +  layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i],grads[i]) pairs.
    updates = []
    for param_i, grad_i in zip(params, grads):
        updates.append((param_i, param_i - learning_rate * grad_i))

    train_model = theano.function([index], cost, updates=updates,
          givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        print epoch
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 104 == 0:
                print 'training @ iter = ', iter
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                      (epoch, minibatch_index + 1, n_train_batches, \
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss or epoch % 100 == 0:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                    if this_validation_loss <= best_validation_loss:
                    # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter

                    # test it on the test set
                        test_losses = [test_model(i) for i in xrange(n_test_batches)]
                        test_score = numpy.mean(test_losses)
                        print(('     epoch %i, minibatch %i/%i, test error of best '
                            'model %f %%') %
                            (epoch, minibatch_index + 1, n_train_batches,
                            test_score * 100.))

            if patience <= iter/52:
                #done_looping = True
                #break
                pass

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i,'\
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    if out: 
        out.write('%d\t%d\t%f\t%f\n'% (nkerns[0], n_hidden, best_validation_loss*100., test_score*100.))

    f = file(save_name, 'wb')
    obj = (layer0, layer2, layer3)
    cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    
    
if __name__ == '__main__':
#    load_mat_patch()
    numpy.random.seed(13576)
    data = load_mat_patch(datfile='KSC_norm.mat',datvar='KSC_norm')
    #f = open('cnn_rslt.txt', 'w')
    #for i in xrange(10):
    #    a = (i+1)*100
    #    for j in xrange(i):
    #        b = (j+1)*100
    #        print a,b

    print 100, 40
    evaluate_cnns(data=data, nkerns=[100], n_hidden=40, save_name="saved_cnn_100_40")


    #f.close()


def experiment(state, channel):
    evaluate_cnns(state.learning_rate, dataset=state.dataset)
