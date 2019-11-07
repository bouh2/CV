
# Famous classifier to recognize cats

from dnn_utils import * # L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
from scipy import ndimage
from skimage import transform




#=======================================================================================================================

plt.rcParams['figure.figsize'] = (5.0, 4.0) # default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


train_x_orig, train_y, test_x_orig, test_y, classes = load_dataset()

# Sample pic
'''
index = 12
plt.imshow(train_x_orig[index])
print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")
plt.show()
'''

m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

# Reshape the training and test samples
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Normalization
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.



layers_dims = [12288, 40, 10, 5, 1] #  5-layer model
parameters = L_layer_model(train_x, train_y, layers_dims, learning_rate=0.03, num_iterations = 2500, print_cost = True)

pred_train = predict(train_x, train_y, parameters)
pred_test = predict(test_x, test_y, parameters)
print_mislabeled_images(classes, test_x, test_y, pred_test)


#sample test
my_image = "dog0.png"
my_label_y = [0] # true label


fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = transform.resize(image, (num_px,num_px,3), mode="reflect").reshape((1, num_px*num_px*3)).T

my_predicted_image = predict(my_image, my_label_y, parameters)

plt.imshow(image)
plt.title ("y = " + str(np.squeeze(my_predicted_image)) + ", " + str(len(layers_dims)) + "-layer model prediction is a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
plt.show()