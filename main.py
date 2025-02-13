#Importing all the required modules
import tensorflow as tf
import numpy as np
import keras
import pandas as pd
import matplotlib.pyplot as plt
#Importing the datset
fmnist=tf.keras.datasets.fashion_mnist
(training_images,training_labels),(test_images,test_labels)=fmnist.load_data()
index=100
print(f'Label:z{training_labels[index]}')
print(f'\nImage pixel array:\n{training_images[index]}')
plt.imshow(training_images[index])
plt.colorbar()
plt.show()
training_images=training_images/255.0
test_images=test_images/255.0
model=tf.keras.models.Sequential([tf.keras.Input(shape=(28,28)),
                                  tf.keras.layers.Flatten(),
                                  tf.keras.layers.Dense(128,activation=tf.nn.relu),
                                  tf.keras.layers.Dense(10,activation=tf.nn.softmax)])
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self,epoch,logs={}):
    if logs['loss']<0.4:
      print('Loss is too low so cancelling training')
      self.model.stop_trainig=True
model.compile(optimizer=tf.optimizers.Adam(),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(training_images,training_labels,epochs=5,callbacks=[myCallback()])
model.evaluate(test_images,test_labels)
print(test_labels[21])
test=np.array([test_images[21]])
plt.imshow(test[0])
plt.colorbar()
plt.show()
out=model.predict(test)
print(np.argmax(out))
