from ann_visualizer.visualize import ann_viz
import tensorflow as tf

model = tf.keras.models.load_model('visualize.h5')

ann_viz(model, title="My first neural network")