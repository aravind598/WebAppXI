from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import base64
import json
import logging
from PIL import Image
import io
from azureml.core.model import Model


np.set_printoptions(suppress=True)

def init():
    #global model
    global model
    global tflite_inter
    #model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    try:
        model = load_model(Model.get_model_path("mymodel"))
        tflite_inter = tf.lite.Interpreter(model_path=Model.get_model_path("my_model.tflite"))
        tflite_inter.allocate_tensors()
    except Exception as e:
        print(str(e))
        return str(e)

def run(info):
    #model.eval()
    try:
        logging.info("Request received")
        jsonInfer = json.loads(info)
        inference = jsonInfer['inference']
        bytestr = base64.b64decode(str(inference))
        
        #img = Image.open(io.BytesIO(bytestr))
        
        img = tf.io.decode_image(bytestr, channels=3, dtype=tf.dtypes.float32)
        img = tf.image.resize(img, [224,224])
        img = tf.cast(tf.expand_dims(img, axis=0), tf.dtypes.float32)
        classes = ["Fruit", "Dog", "Person", "Car", "Motorbike", "Flower", "Cat"]
        classes = ["Car", "Cat", "Dog", "Flower", "Fruit", "Motorbike", "Person"]
        # run the inference
        prediction = model.predict(img)
        print(classes[prediction.argmax()])
        logging.info("Request processed")
        
        shape = (1,224,224,3)
        data = np.ndarray(shape, dtype=np.uint8)
        # Replace this with the path to your image
        image = Image.open(io.BytesIO(bytestr)).convert('RGB')
        #resize the image to a 224x224 with the same strategy as in TM2:
        #resizing the image to be at least 224x224 and then cropping from the center
        #img_shape=224
        #size = (img_shape, img_shape)
        #image = ImageOps.fit(image, size, Image.ANTIALIAS)
        #turn the image into a numpy array
        image_array = np.asarray(image)
        # Normalize the image
        normalized_image_array = np.uint8(image_array)
        # Load the image into the array
        data[0] = normalized_image_array
        
        input_details = tflite_inter.get_input_details()
        output_details = tflite_inter.get_output_details()
        tflite_inter.set_tensor(input_details[0]['index'], data)
        tflite_inter.invoke()
        output_data = tflite_inter.get_tensor(output_details[0]['index'])
        colab_classes = ['Airplane', 'Bird', 'Car', 'Cat', 'Dog', 'Flower', 'Fruit', 'Motorbike', 'Person']
        return str(colab_classes[output_data.argmax()]) + str(classes[prediction.argmax()])

    except Exception as e:
        error = str(e)
        print(error)
        return error