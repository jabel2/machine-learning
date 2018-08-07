import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

get_ipython().run_line_magic('matplotlib', 'inline')

# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()


if __name__ == '__main__':
    # Example of a picture
    index = 10
    plt.imshow(train_set_x_orig[index])
    print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")
    print (classes)

if __name__ == '__main__':
    print('the shape of train_set_x_org', train_set_x_orig.shape)

    # The number of training examples is a 1 d array with 209 examples
    print('the shape of train_set_y: ', train_set_y.shape)
    print('the shape of test_set_x_orig', test_set_x_orig.shape)

    #the number of test examples is a 1 d array with 50 examples
    print('the shape of test_set_y', test_set_y.shape)


m_train = np.shape(train_set_y)[1]
m_test = np.shape(test_set_y)[1]

#since the shape of train_set_x_orig is (209, 64, 64, 3), then the number of num_pix is either the second or third shape value
#which is indexed as 1 or 2
num_px = train_set_x_orig.shape[1]

if __name__ == '__main__':
    print ("Number of training examples: m_train = " + str(m_train))
    print ("Number of testing examples: m_test = " + str(m_test))
    print ("Height/Width of each image: num_px = " + str(num_px))
    print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print ("train_set_x shape: " + str(train_set_x_orig.shape))
    print ("train_set_y shape: " + str(train_set_y.shape))
    print ("test_set_x shape: " + str(test_set_x_orig.shape))
    print ("test_set_y shape: " + str(test_set_y.shape))

# Reshape the training and test examples
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

if __name__ == '__main__':
    print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
    print ("train_set_y shape: " + str(train_set_y.shape))
    print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
    print ("test_set_y shape: " + str(test_set_y.shape))
    print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))

train_set_x_orig = train_set_x_flatten/255.
test_set_x_orig = test_set_x_flatten/255.
