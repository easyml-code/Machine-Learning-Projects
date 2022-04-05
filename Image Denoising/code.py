#Importing Libraries
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Conv2DTranspose, Conv2D, Input

#Putting Image Pixel Values in form of 28x28 Array
# We don't need y_train and y_test
(x_train, _), (x_test, _) = fashion_mnist.load_data()

# These Steps are used for generating images in grid format as a plot {For comparison between Original and Noised images}
fig, axs = plt.subplots(5, 10)  
fig.tight_layout(pad=-1)
plt.gray()
a = 0 
for i in range(5): 
  for j in range(10): 
    axs[i, j].imshow(tf.squeeze(x_test[a])) 
    axs[i, j].xaxis.set_visible(False) 
    axs[i, j].yaxis.set_visible(False) 
    a = a + 1 

# Normalising the RGB values between 0 and 1 {For Computaional Efficiency and Model Reliability}    
x_train = x_train.astype('float32') / 255. 
x_test = x_test.astype('float32') / 255.


# At this stage we are adding a fourth Dimension is used to ensure that we are using greyscale images
x_train =  x_train[..., tf.newaxis] 
x_test = x_test[..., tf.newaxis]

#Creation of the noisy version of the MNIST Dataset
#We will add randomly generated value to each array item

noise_factor = 0.4
x_train_noisy= x_train + noise_factor * tf.random.normal(shape=x_train.shape) 
x_test_noisy = x_test + noise_factor* tf.random.normal(shape=x_test.shape)

# But these values may not be within 0 and 1 for that we will tf.clip_by_value method

x_train_noisy = tf.clip_by_value(x_train_noisy, clip_value_min=0., clip_value_max=1.) 
x_test_noisy = tf.clip_by_value(x_test_noisy, clip_value_min=0., clip_value_max=1.)


#We are using 'Model Subclassing' method to build our model. It is fully customizable
class NoiseReducer(tf.keras.Model): 
  def __init__(self):
                                                                                                            
    super(NoiseReducer, self).__init__() 
#We need to call the initialized encoder model which takes the images as input
    self.encoder = tf.keras.Sequential([ 
      Input(shape=(28, 28, 1)), 
      Conv2D(16, (3,3), activation='relu', padding='same', strides=2), 
      Conv2D(8, (3,3), activation='relu', padding='same', strides=2)]) 
#calling the initialized decoder model which takes the output of the encoder model (encoded) as input    
    self.decoder = tf.keras.Sequential([ 
      Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'), 
      Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'), 
      Conv2D(1, kernel_size=(3,3), activation='sigmoid', padding='same')]) 
  
  def call(self, x): 
    encoded = self.encoder(x) 
    decoded = self.decoder(encoded) 
    return decoded

autoencoder = NoiseReducer()

autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(x_train_noisy, x_train, epochs=10, shuffle=True, validation_data=(x_test_noisy, x_test))


encoded_imgs=autoencoder.encoder(x_test_noisy).numpy()
decoded_imgs=autoencoder.decoder(encoded_imgs)

n = 10 
plt.figure(figsize=(20, 7))
plt.gray()
for i in range(n): 
  # display original + noise 
  bx = plt.subplot(3, n, i + 1) 
  plt.title("original + noise") 
  plt.imshow(tf.squeeze(x_test_noisy[i])) 
  bx.get_xaxis().set_visible(False) 
  bx.get_yaxis().set_visible(False) 
  
  # display reconstruction 
  cx = plt.subplot(3, n, i + n + 1) 
  plt.title("reconstructed") 
  plt.imshow(tf.squeeze(decoded_imgs[i])) 
  cx.get_xaxis().set_visible(False) 
  cx.get_yaxis().set_visible(False) 
  
  # display original 
  ax = plt.subplot(3, n, i + 2*n + 1) 
  plt.title("original") 
  plt.imshow(tf.squeeze(x_test[i])) 
  ax.get_xaxis().set_visible(False) 
  ax.get_yaxis().set_visible(False) 

plt.show()
