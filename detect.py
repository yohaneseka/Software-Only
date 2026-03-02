import os
import json
# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #only show error message
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags
from absl.flags import FLAGS
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

flags.DEFINE_string(
    "model", "/home/microscope/malaria/MalaScope/fold_5.keras", "path to saved model"
)
flags.DEFINE_integer('size',128,'resize images to')
flags.DEFINE_string(
    "input_path",
    "/home/microscope/malaria/MalaScope/extracted_cell",
    "path to input cell folder",
)
flags.DEFINE_string(
    "output_path",
    "/home/microscope/malaria/MalaScope/predicted_cell",
    "path to output folder",
)
flags.DEFINE_float("score_thr", 0.25, "prediction score threshold")

class GradCAM:
    def __init__(self, model, classIdx, layerName=None):
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName
        if self.layerName is None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

    def compute_heatmap(self, image, eps=1e-8):
        gradModel = tf.keras.models.Model(
            inputs=self.model.input,
            outputs=[self.model.get_layer(self.layerName).output, self.model.output])

        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            loss = predictions[:, self.classIdx]

        grads = tape.gradient(loss, convOutputs)

        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads

        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))

        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")
        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha, colormap=cv2.COLORMAP_JET):
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
        return (heatmap, output)


def main(_argv):
    config=ConfigProto()
    config.gpu_options.allow_growth=True
    session=InteractiveSession(config=config)

    target_size=FLAGS.size
    input_dir=FLAGS.input_path
    output_dir=FLAGS.output_path
    threshold = FLAGS.score_thr

    model = tf.keras.models.load_model(FLAGS.model)

    for count, filename in enumerate(os.listdir(input_dir)):
        image_path = os.path.join(input_dir, filename)
        original_image = cv2.imread(image_path)
        image = cv2.resize(original_image, (target_size, target_size))
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_data = rgb_image.astype('float32') / 255.0
        image_data = np.expand_dims(image_data,axis=0)

        predictions = model.predict(image_data)
        i = np.argmax(predictions[0])
        predicted_label = "Parasitized" if predictions < threshold else "Uninfected"

        print(f"{filename} predicted: {predicted_label} with score {predictions}")

        max_params = 0
        layer_selected = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                params = layer.count_params()
                if params > max_params:
                    max_params = params
                    layer_selected = layer.name

        icam = GradCAM(model, i, layer_selected)
        heatmap = icam.compute_heatmap(image_data)
        heatmap = cv2.resize(heatmap, (target_size, target_size))
        (heatmap, output) = icam.overlay_heatmap(heatmap, rgb_image, alpha=0.7)

        os.makedirs(os.path.join(output_dir, predicted_label), exist_ok=True)
        cell_heatmap=os.path.join(output_dir,predicted_label,filename)
        cv2.imwrite(cell_heatmap, output)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
