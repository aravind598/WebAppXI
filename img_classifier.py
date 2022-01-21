import tensorflow as tf
from PIL import Image
import numpy as np
import io
from contextlib import contextmanager, redirect_stdout
from io import StringIO


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


def prediction_my(model, pred):
    classes = ["Fruit", "Dog", "Person", "Car", "Motorbike", "Flower", "Cat"]
    # run the inference
    prediction = model.predict(pred)
    #print(classes[prediction.argmax()])
    return classes[prediction.argmax()]

def prepare_my(bytestr: bytes, shape = (1,224,224,3) ):
    """[summary]

    Args:
        bytestr (bytes): [image bytestr from read]
        shape (tuple, optional): [description]. Defaults to (1,224,224,3).

    Returns:
        ndarray: [Output the data in the form of [1,224,224,3] ]]
    """    
    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape, dtype=np.float32)
    # Replace this with the path to your image
    image = Image.open(io.BytesIO(bytestr)).convert('RGB')
    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    #size = (img_shape, img_shape)
    #image = ImageOps.fit(image, size, Image.ANTIALIAS)
    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array
    return data



#@st.experimental_singleton # cache the function so predictions aren't always redone (Streamlit refreshes every click)
def make_prediction(model, image):
    """
    Takes an image and uses model (a trained TensorFlow model) to make a
    prediction. Using the EfficientNet

    Returns:
     image (preproccessed)
     pred_class (prediction class from class_names)
     pred_conf (model confidence)
    """
    image_array = prepare(image,expand_dims=True)
    image_pred = prediction(model,image_array)
    return str(image_pred)


def make_my_prediction(my_model,image):
    """
    Takes an image and uses model (a trained TensorFlow model) to make a
    prediction. Using My Model

    Returns:
     image (preproccessed)
     pred_class (prediction class from class_names)
     pred_conf (model confidence)
    """
    #my_model = tf.keras.models.load_model("mymodel")
    image_array = prepare_my(image)
    image_pred = prediction_my(my_model,image_array)
    return str(image_pred)

def prediction_my(model, pred):
    """[Prediction using my model]

    Args:
        model ([type]): [description]
        pred ([type]): [description]

    Returns:
        [type]: [description]
    """
    classes = ["Car","Cat","Dog", "Flower", "Fruit", "Motorbike", "Person"]
    # run the inference
    prediction = model.predict(pred)
    #print(classes[prediction.argmax()])
    return classes[prediction.argmax()]

def getOutput(interpreter, input_data, input_details=None, output_details=None, colab=False):
    """[Get output from interpreter using tflite models]

    Args:
        interpreter ([type]): [description]
        input_data ([type]): [description]

    Returns:
        [type]: [description]
    """
    #
    if not input_details or not output_details:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
    #print(input_details)
    #print(output_details)
    # Test the model on random input data.
    #input_shape = input_details[0]['shape']
    #input_data = np.array(np.random.random_sample(input_shape), dtype=np.uint8)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    if colab:
        colab_classes = ['Airplane', 'Bird', 'Car', 'Cat', 'Dog', 'Flower', 'Fruit', 'Motorbike', 'Person']
        return colab_classes[output_data.argmax()]
    else:
        classes = ["Car","Cat","Dog", "Flower", "Fruit", "Motorbike", "Person"]
        return classes[output_data.argmax()]
    
    #print(output_data)
    

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



"""
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
"""

@contextmanager
def st_capture(output_func):
    with StringIO() as stdout, redirect_stdout(stdout):
        old_write = stdout.write

        def new_write(string):
            ret = old_write(string)
            output_func(stdout.getvalue())
            return ret
        
        stdout.write = new_write
        yield

def createserver():
    from azureml.core import Workspace
    from azureml.core import Workspace
    from azureml.core.webservice import AciWebservice, LocalWebservice
    from azureml.core.webservice import Webservice
    from azureml.core.model import InferenceConfig, Model
    from azureml.core.environment import Environment
    from azureml.core import Workspace
    from azureml.core.conda_dependencies import CondaDependencies
    try:
        ws = Workspace.get( name="azworkspace",
                            subscription_id='763d1b5b-6b77-4ca2-b25e-c2f3774a3d2a',
                            resource_group='azureresource')

        myenv = Environment(name="myenv")
        conda_dependencies = CondaDependencies()

        conda_dependencies.add_conda_package("python==3.9")

        #conda_dependencies.add_conda_package("scikit-learn") 
        #conda_dependencies.add_conda_package("tensorflow") 
        #conda_dependencies.add_conda_package("pip") 
        conda_dependencies.add_pip_package("tensorflow")
        conda_dependencies.add_pip_package("numpy")
        conda_dependencies.add_pip_package("pillow") 
        #conda_dependencies.add_pip_package("azureml-core")
        #conda_dependencies.add_pip_package("keras")
        conda_dependencies.add_pip_package("torch")
        conda_dependencies.add_pip_package("torchvision")
        conda_dependencies.add_pip_package("torchaudio")
        conda_dependencies.add_pip_package("timm")

        myenv.python.conda_dependencies = conda_dependencies
        
        m = Model.register(#model_path= r"C:\Users\Aravind\OneDrive - Nanyang Technological University\CZ4171\models\mymodel", #model_path= r"C:\Users\Aravind\OneDrive - Nanyang Technological University\CZ4171\models\enetd0"
                        model_path = "mymodel",
                        model_name = "mymodel",
                        description = "Custom Keras Model with 7 classes" , 
                        workspace = ws)
            # Create inference configuration based on the environment definition and the entry script
        #myenv = Environment.from_conda_specification(name="env", file_path=r"C:\Users\Aravind\Downloads\env.yml")

        #myenv.python.conda_dependencies = myenvironment

        inference_config = InferenceConfig(entry_script="entry_script.py", environment=myenv)
        # Create a local deployment, using port 8890 for the web service endpoint
        deployment_config = LocalWebservice.deploy_configuration(port=8890)
        # Deploy the service
        service = Model.deploy(
            ws, "ef0", [m], inference_config, deployment_config, overwrite=True, show_output=True)
        # Wait for the deployment to complete
        service.wait_for_deployment(True)
        # Display the port that the web service is available on
        print(service.port)
        return service
    except Exception as e:
        print(str(e))
        try:
            service.delete()
        except:
            pass
