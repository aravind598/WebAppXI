from functools import cache
import streamlit as st
#from streamlit_tensorboard import st_tensorboard
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import io
import requests
from img_classifier import getOutput, prepare_my, make_my_prediction, make_prediction
import traceback
import copy
import time
import base64
import json
#import cv2
#from img_classifier import our_image_classifier
# import firebase_bro

# Just making sure we are not bothered by File Encoding warnings
st.set_option('deprecation.showfileUploaderEncoding', False)

global model; global my_model; global tflite_colab; 
#global tflite_model_uint8; global tflite_model 
#global picture
global uri
global sentence
uri = None
sentence = None

@st.experimental_singleton
@cache
def call_model(model_path):
    """[Call my model and the EfficientNet model]

    Args:
        model_path ([type]): [description]

    Returns:
        [type]: [description]
    """
    model = tf.keras.models.load_model(model_path)
    model.make_predict_function()
    #model.summary()
    return model

@st.experimental_singleton
@cache
def call_efficient(model_path):
    model = tf.keras.models.load_model(model_path)
    model.make_predict_function()
    #model.summary()
    return model

@st.experimental_singleton
@cache
def call_interpreter(model_path):
    """[Call tflite interpreter]

    Args:
        model_path ([type]): [description]

    Returns:
        [type]: [Returns an interpreter]
    """
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


@st.experimental_singleton
@cache
def call_colab(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter




#@st.experimental_memo
#@st.cache
#@cache
def cache_image(image_byte: bytes, azure = False, img_shape: int = 224) -> bytes:
    """[Cache the image and makes the image smaller before doing stuff]

    Args:
        image_byte (bytes): [Get from reading the file/image using image.read() or file.read()]
        img_shape (int, optional): [size of the default prediction/image tensor i.e needs to be 224x224 in image size for all my models]. Defaults to 224.

    Returns:
        bytes: [return a new bytes object that is smaller/faster to interpret]
    """
    byteImgIO = io.BytesIO()
    image = Image.open(io.BytesIO(image_byte)).convert('RGB')
    #print(image.size)  
    size = (img_shape, img_shape)
    
    # Maintain Aspect Ratio of images by adding padding of black bars
    image = ImageOps.pad(image, size, Image.ANTIALIAS)
    # Cut images to size by cropping into them
    #image = ImageOps.fit(image, size, Image.ANTIALIAS)
    
    #print(image.size)
    
    #Lower the image size by decreasing its quality
    image.save(byteImgIO, format = "JPEG", optimize=True,quality = 80)
   
   #If the azure variable is True then dump the data as encoded utf-8 json for sending to the server 
    if azure:
        img_byte = byteImgIO.getvalue()  # bytes
        img_base64 = base64.b64encode(img_byte)  # Base64-encoded bytes * not str
        img_str = img_base64.decode('latin-1')  # str
        data = {"inference": img_str}
        return bytes(json.dumps(data),encoding='utf-8')
    
    byteImgIO.seek(0)
    image = byteImgIO.read()
    return image

@cache
def main():
    """
    [Main function all the UI is here]
    """
    # Metadata for the web app
    st.set_page_config(
    page_title = "WebApp",
    layout = "wide",
    page_icon= ":dog:",
    initial_sidebar_state = "collapsed",
    )

    with st.spinner("The magic of our AI is starting.... Please Wait"):
        model = call_efficient("enetd0")
        my_model = call_model("FinalTeachingModel")
        tflite_model = call_interpreter(model_path="FinalTeachingModel/model_unquant.tflite")
        #tflite_model_uint8 = call_interpreter(model_path="mymodel/teaching_quant.tflite")
        #tflite_colab = call_colab(model_path="mymodel/efficientlite0.tflite")
    

    #st.balloons()

    #Sidebar selection
    choose_model = st.sidebar.selectbox(
        "Pick model you'd like to use",
        ("Model 1 (Custom Model)",  # original 10 classes
         "Model 2 (EfficientNet)",  # original 10 classes + donuts
         "Model 3 (Colab)") )
         #"Model 4 (Quantised Model)",
         #"Model 5 (UnQuantised Model)")  11 classes (same as above) + not_food class
    #)
    
    menu = ['Home', 'Stats', 'Contact', 'Feedback']
    choice = st.sidebar.selectbox("Menu", menu)
        

    if choice == "Home":
        # Let's set the title of our awesome web app
        st.title("Aravind's Application")
        # Now setting up a header text
        #st.subheader("By Your Cool Dev Name")
        
        #Expander 1
        my_expandering = st.expander(label='Model URL Input')
        with my_expandering:
            try:
                uri = st.text_input('Enter Azure ML Inference URL here:')
            except:
                uri = ""
        if uri:
            st.write("Azure ML Inference URL: " + uri)
        
        #Expander 1.5
        #QR code input but need to manually copy and paste into the above line to store the uri
        #my_expanders = st.expander(label="QR Code Input")
        checking_list = ["http", "/score"]
        a = """
        with my_expanders:
            QR_file = st.file_uploader("Input QR Code", type=["jpg", "png", "jpeg"])
            qrCodeDetector = cv2.QRCodeDetector()
            try:
                if st.button("Altair") and QR_file:
                    nparr = np.frombuffer(QR_file.read(), np.uint8)
                    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    decodedText, _ , _ = qrCodeDetector.detectAndDecode(img_np)
                    decodedText = str(decodedText).strip()
                    if all(x in decodedText for x in checking_list):
                        uri = decodedText
                        st.success("Azure ML Url is at: " + decodedText)
                    else:
                        st.error(f"URI of {decodedText} is not valid")
            except Exception as e:
                st.error("Failure" + str(e))
                uri = ""
        """
        #Expander 2
        #Changed from this
        #sentence = st.text_input('Input your sentence here:')
        # To this using an expander
        my_expander = st.expander(label='Inference for images on the internet:')
        image_bytes = None
        with my_expander:
            sentence = st.text_input('Input your image url here:') 
            if sentence:
                try:
                    response = requests.get(sentence)
                    # = Image.open(io.BytesIO(response.content))
                    image_bytes = response.content
                    #st.write(str(sentence))
                except Exception as e:
                    st.error("Exception occured due to url not having image " + str(e))
                    image = None
                    #st.error()
                    


        #Expander 3
        # Option to upload an image file with jpg,jpeg or png extensions
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])
        
        #Copy images if not error
        uploaded_file1 = copy.copy(uploaded_file)
        uploaded_copy = copy.copy(uploaded_file)
        image1 = copy.copy(image_bytes)
        #picture1 = copy.copy(picture)

        jsonImage = None
        if uploaded_file is not None:
            upload = uploaded_file.read()
            jsonImage = cache_image(image_byte=upload, azure=True)
        elif image_bytes is not None:
            jsonImage = cache_image(image_byte=image1, azure=True)
            

            
        if uploaded_file is not None:
            placeholder = st.image(uploaded_file1.read(),use_column_width=True)
        elif image_bytes is not None:
            placeholder = st.image(image1,use_column_width=True)
        #elif picture is not None:
            #placeholder = st.image(picture1.read(),use_column_width=True)
        else:
            pass
        
        #azpredictbut = st.button("Azure ML Predict")
        try:
            if uri:
                if all(x in uri for x in checking_list):
                    #st.write(str(uriparts in uri for uriparts in checking_list))
                    if st.button("Azure ML Predict"):
                            if uploaded_file is not None or image_bytes is not None:
                                t = time.time()
                                if jsonImage:
                                    headers = {"Content-Type": "application/json"}
                                    response = requests.post(uri, data=jsonImage, headers=headers)
                                    label = str(response.text)
                                    #label = azure_prediction(jsonImage, uri)
                                else:
                                    label = "error"
                                    st.error("JsonImage error")
                                    
                                st.success(label)
                                st.success("Time taken is : " + str(time.time()- t))
                            
                            else:
                                st.error("Can you please upload an image 🙇🏽‍♂️")
                    else:
                        pass#st.error("Button not pressed")
                else:
                    uri = ""
                    st.error("URL is not correct/valid or empty. Did you include the /score? ")
            else:
                pass
                #st.error("URL is empty.")
        except Exception as e:
            st.error('Service unavailable -> Is the Server Running?')
            st.error(str(e))
            
            

        # When the user clicks the predict button
        if st.button("Local Prediction"):
            start = time.time()
        # If the user uploads an image
            if uploaded_file is not None or image_bytes is not None:
                
                if uploaded_file:
                    # Opening our image
                    #placeholder = st.image(copy.copy(uploaded_file).read(),use_column_width=True)
                    single_image = uploaded_copy.read()
                    image = cache_image(image_byte = single_image)
                    #input_data = prepare_my_uint8(image)
                    #print(type(image))
                    #image = Image.open(uploaded_file
                    
                
                #Predict using the image link
                elif image_bytes:
                    image = cache_image(image_byte = image_bytes)
                    #input_data = prepare_my_uint8(image)
                else:
                    st.error("Error")
                # # Send our image to database for later analysis
                # firebase_bro.send_img(image)
                # Let's see what we got
                st.image(image,use_column_width=True)
                st.write("")
                try:
                    with st.spinner("The magic of our AI has started...."):
                            #label = our_image_classifier(image)
                        if choose_model == "Model 1 (Custom Model)":
                            #model = call_model()
                            label=make_my_prediction(my_model,image,colab=True)
                            t = time.time() - start
                        
                        elif choose_model == "Model 2 (EfficientNet)":
                            #my_model = call_my_model()
                            label=make_prediction(model,image)
                            t = time.time() - start
                            #time.sleep(8)
                        elif choose_model == "Model 3 (Colab)":
                            label = getOutput(tflite_model, prepare_my(image), colab=True)
                            t = time.time() - start
                            
                            #label=getOutput(tflite_colab,prepare_my(image,colab=True),colab=True)
                            #label = "Nothing here"
                            #t = time.time() - start
                            
                        #elif choose_model == "Model 4 (Quantised Model)":
                        #    label = "Nothing here"
                        #   #label = getOutput(tflite_model_uint8, input_data)
                        #    t = time.time() - start
                        #   
                        #elif choose_model == "Model 5 (UnQuantised Model)":
                        #    #input_data = prepare_my_uint8(image)
                        #   label = getOutput(tflite_model, prepare_my(image),colab=True)
                        #    t = time.time() - start
                        
                        else:
                            #TODO 
                            label = "Not yet done"
                            t = time.time() - start
                    if placeholder:
                        placeholder.empty()
                    st.success("We predict this image to be: "+ label)
                    st.success("Time Taken "+ str(t))
                    #rating = st.slider("Do you mind rating our service?",1,10)
                except Exception as e:
                    st.error(e)
                    st.error(traceback.format_exc())
                    st.error("We apologize something went wrong 🙇🏽‍♂️")
            else:
                st.error("Can you please upload an image 🙇🏽‍♂️")
        
        if st.button("Clear Screen"):
            placeholder.empty()
            
    elif choice == "Contact":
        # Let's set the title of our Contact Page
        st.title('Get in touch')
        def display_team(name,path,affiliation="",email=""):
            '''
            Function to display picture,name,affiliation and name of creators
            '''
            team_img = Image.open(path)

            st.image(team_img, width=350, use_column_width=False)
            st.markdown(f"## {name}")
            st.markdown(f"#### {affiliation}")
            st.markdown(f"###### Email {email}")
            st.write("------")

        display_team("Your Awesome Name", "./assets/profile_pic.png","Your Awesome Affliation","hello@youareawesome.com")

    elif choice == "Stats":
        # Let's set the title of our About page
        st.title('Tensorboard Stats of the Run')
        st.markdown('## Train')
        #st_tensorboard(logdir= "logs/train/", port=5011, width=1080)
        st.markdown('## Validation')
        #st_tensorboard(logdir= "logs/validation/", port=5011, width=1080)
        # A function to display the company logo
        def display_logo(path):
            company_logo = Image.open(path)
            st.image(company_logo, width=350, use_column_width=False)

        # Add the necessary info
        display_logo("./assets/profile_pic.png")
        st.markdown('## Objective')
        st.markdown("Write your company's objective here.")
        st.markdown('## More about the company.')
        st.markdown("Write more about your country here.")

    elif choice == "Feedback":
        # Let's set the feedback page complete with a form
        st.title("Feel free to share your opinions :smile:")

        first_name = st.text_input('First Name:')
        last_name = st.text_input('Last Name:')
        user_email = st.text_input('Enter Email: ')
        feedback = st.text_area('Feedback')

        # When User clicks the send feedback button
        if st.button('Send Feedback'):
            # # Let's send the data to a Database to store it
            # firebase_bro.send_feedback(first_name, last_name, user_email, feedback)

            # Share a Successful Completion Message
            st.success("Your feedback has been shared!")

if __name__ == "__main__":
    main()
