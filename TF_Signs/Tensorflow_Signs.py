
# Recognition of hand signs from 0 to 5

from tf_utils import * # 3-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX
from skimage import transform
from scipy import ndimage

#=======================================================================================================================

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

#'''
# Sample pic
index = 0
plt.imshow(X_train_orig[index])
plt.title("Sign of " + str(np.squeeze(Y_train_orig[:, index])))
plt.show()
#'''

# Reshape the training and test samples
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
# Normalization
X_train = X_train_flatten/255.
X_test = X_test_flatten/255.
# Convert labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)


parameters = model(X_train, Y_train, X_test, Y_test, num_epochs=1300)


my_image = "thumbs_up.png"
fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = transform.resize(image, (64,64,3), mode="reflect").reshape((1, 64*64*3)).T

my_image_prediction = predict(my_image, parameters)
plt.title("Prediction: it's a sign of " + str(np.squeeze(my_image_prediction)))
plt.imshow(image)
plt.show()