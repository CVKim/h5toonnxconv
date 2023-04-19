import tensorflow as tf
import tf2onnx
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import tensorflow.keras.applications as app

SIZE_X = 260
SIZE_Y = 260

# data augmentation
img_augmentation = Sequential(
    [
        # layers.RandomRotation(factor=0.1),
        # layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        # layers.RandomFlip(),
        # layers.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)

def build_model(num_classes, input_width, input_height, expand_output=False): 
    inputs = layers.Input(shape=(input_height, input_width, 3), name = "data")    
    x = img_augmentation(inputs)
    model = app.EfficientNetB2(include_top=False, input_tensor=x, weights="imagenet")
    
    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.3
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)
    
    if expand_output:
        outputs = tf.expand_dims(outputs, axis=1)
        outputs = tf.expand_dims(outputs, axis=1)
    
    print("output shape : ", outputs.shape)

    model = tf.keras.Model(inputs, outputs, name="EfficientNet")

    return model

# h5 model load
test_seedmodel = './230417_EfficientNet_epoch063_acc0.8744.h5'
new_input_shape = (None, SIZE_Y, SIZE_X, 3)

model = build_model(32, SIZE_X, SIZE_Y, True)
model.load_weights(test_seedmodel)

# config acc
# model_prev = tf.keras.models.load_model(test_seedmodel)
# config = model_prev.get_config()
# config['layers'][0]['config']['batch_input_shape'] = new_input_shape
# model = tf.keras.models.Model.from_config(config)

in_shape = model.inputs[0].shape.as_list()
in_shape[0] = 1

# output name change
model.output_names[0] = 'output'

spec = (tf.TensorSpec(in_shape, tf.float32, name="data"),)        
output_path = test_seedmodel[:-3]+'_______.onnx'
tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)      
print("Conveting End...!")

