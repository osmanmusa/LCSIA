import tensorflow as tf

class MyModel(tf.keras.Model):

  def __init__(self):
    super(MyModel, self).__init__()
    self.save_var = tf.Variable(1.0, name='save-var')
    self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu, input_shape=(3,))
    self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
    self.dropout = tf.keras.layers.Dropout(0.5)

  def call(self, inputs, training=False):
    x = self.dense1(inputs)
    if training:
      x = self.dropout(x, training=training)
    return self.dense2(x)

# class CustomLayer(tf.keras.layers.Layer):
#   def __init__(self, a):
#     self.var = tf.Variable(a, name="var_a")

model = MyModel()

model.save_var.assign(1010.101)
# model.save_weights('my_model.h5', overwrite=True, save_format='h5')
# print("Saved model to disk")



# loaded_model = MyModel()
# loaded_model.load_weights('my_model.h5')
# print("Loaded model from disk")
# print(loaded_model.save_var)

path = './experiments/m250_n500_k0.0_p0.1_s40/LITA_T2_bg_t_s_0/ckpt'
model.save_weights(path)
print("Saved model to disk")

loaded_model2 = MyModel()
loaded_model2.load_weights(path)
stop = 1

# layer = CustomLayer(5)
# layer_ckpt = tf.train.Checkpoint(layer=layer).save("custom_layer")



# import os
#
# import tensorflow as tf
# from tensorflow import keras
#
# (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
#
# train_labels = train_labels[:1000]
# test_labels = test_labels[:1000]
#
# train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
# test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0
#
# print(tf.version.VERSION)
#
# # Define a simple sequential model
# def create_model():
#   model = tf.keras.models.Sequential([
#     keras.layers.Dense(512, activation='relu', input_shape=(784,)),
#     keras.layers.Dropout(0.2),
#     keras.layers.Dense(10)
#   ])
#
#   model.compile(optimizer='adam',
#                 loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
#                 metrics=['accuracy'])
#
#   return model
#
# # Create a basic model instance
# model = create_model()
#
# # Display the model's architecture
# model.summary()
#
#
#
# checkpoint_path = "training_1/cp.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)
#
# # Create a callback that saves the model's weights
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                  save_weights_only=True,
#                                                  verbose=1)
#
# # # Train the model with the new callback
# # model.fit(train_images,
# #           train_labels,
# #           epochs=10,
# #           validation_data=(test_images,test_labels),
# #           callbacks=[cp_callback])  # Pass callback to training
#
# cp_callback.on_train_begin()
#
# # Create a basic model instance
# model = create_model()
#
# # Evaluate the model
# loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
# print("Untrained model, accuracy: {:5.2f}%".format(100*acc))
#
# # Loads the weights
# model.load_weights(checkpoint_path)
#
# # Re-evaluate the model
# loss,acc = model.evaluate(test_images,  test_labels, verbose=2)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))