import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import io

def prepare(bytestr, img_shape=224, rescale=False, expand_dims=False):
    img = tf.io.decode_image(bytestr, channels=3, dtype=tf.dtypes.float32)
    #img = tf.image.resize(img, [img_shape, img_shape])
    if rescale:
        img = img/255.
        img = img.numpy()
    else:
        pass
    if expand_dims:
        return tf.cast(tf.expand_dims(img, axis=0), tf.dtypes.float32)
    else:
        return img.numpy()

def prediction(model, pred):
    prednumpyarray = model.predict(pred)
    print(prednumpyarray.shape)
    predarray = tf.keras.applications.efficientnet.decode_predictions(prednumpyarray, top=5)
    return predarray


def prepare_my(bytestr, img_shape=224):
    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    # Replace this with the path to your image
    image = Image.open(io.BytesIO(bytestr))
    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (img_shape, img_shape)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array
    return data


def prediction_my(model, pred):
    classes = ["Fruit", "Dog", "Person", "Car", "Motorbike", "Flower", "Cat"]
    # run the inference
    prediction = model.predict(pred)
    print(classes[prediction.argmax()])
    return classes[prediction.argmax()]



"""
def our_image_classifier(image):
    '''
            Function that takes the path of the image as input and returns the closest predicted label as output
            '''
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)
    # Load the model
    model = tensorflow.keras.models.load_model(
        'model/name_of_the_keras_model.h5')
    # Determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    # Resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    # Turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (
        image_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array
    labels = {0: "Class 0", 1: "Class 1", 2: "Class 2",
              3: "Class 3", 4: "Class 4", 5: "Class 5"}
    # Run the inference
    predictions = model.predict(data).tolist()
    best_outcome = predictions[0].index(max(predictions[0]))
    print(labels[best_outcome])
    return labels[best_outcome]"""
