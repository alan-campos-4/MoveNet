import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'# turns off different numerical values due to rounding errors 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # enables more tf instructions in operations
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Num CPUs Available: ", len(tf.config.list_physical_devices('CPU')))


#### Logging
print("\n\n--- Logging device placement ---")

tf.debugging.set_log_device_placement(True)
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)
print(c)


#### Manual device placement
print("\n\n--- Manual device placement ---")

tf.debugging.set_log_device_placement(True)
with tf.device('/CPU:0'):
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)
print(c)



