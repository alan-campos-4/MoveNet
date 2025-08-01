import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
infer = module.signatures['serving_default']

# Float32 spec
spec = tf.TensorSpec((1, 256, 256, 3), tf.float32)
func = tf.function(lambda x: infer(x))
concrete = func.get_concrete_function(spec)

frozen = convert_variables_to_constants_v2(concrete)
tf.io.write_graph(frozen.graph, "models/movenet_frozen", "frozen_movenet.pb", as_text=False)
