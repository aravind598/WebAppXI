import streamlit as st
from streamlit_tensorboard import st_tensorboard
import numpy as np
import time
import tensorflow as tf
<<<<<<< Updated upstream
from PIL import Image
from img_classifier import prediction, prepare
=======
from PIL import Image, ImageOps
import io
import requests
from img_classifier import getOutput, prepare_my, make_my_prediction, make_prediction, createserver, st_capture
>>>>>>> Stashed changes
import traceback
#from img_classifier import our_image_classifier
# import firebase_bro

# Just making sure we are not bothered by File Encoding warnings
st.set_option('deprecation.showfileUploaderEncoding', False)

<<<<<<< Updated upstream
global models
#model = tf.keras.models.load_model()
=======
global model; global my_model; global tflite_colab; global tflite_model_uint8; global tflite_model 

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
    model.summary()
    return model

@st.experimental_singleton
@cache
def call_efficient(model_path):
    model = tf.keras.models.load_model(model_path)
    model.make_predict_function()
    model.summary()
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
def call_uint8(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


@st.experimental_singleton
@cache
def call_colab(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


@st.experimental_memo
@st.cache
@cache
def prepare_my_uint8(bytestr, shape = (1,224,224,3) ):
    """[prepare the data for uint8 inference]
>>>>>>> Stashed changes

@st.experimental_singleton # cache the function so predictions aren't always redone (Streamlit refreshes every click)
def make_prediction(image):
    """
    Takes an image and uses model (a trained TensorFlow model) to make a
    prediction.

    Returns:
     image (preproccessed)
     pred_class (prediction class from class_names)
     pred_conf (model confidence)
    """
    model = tf.keras.models.load_model("enetd0")
    image_array = prepare(image,expand_dims=True)
    image_pred = prediction(model,image_array)
    return str(image_pred)

def main():
    # Metadata for the web app
    st.set_page_config(
    page_title = "Title of the webpage",
    layout = "centered",
    page_icon= ":shark:",
    initial_sidebar_state = "collapsed",
    )
<<<<<<< Updated upstream
    choose_model = st.sidebar.selectbox(
    "Pick model you'd like to use",
    ("Model 1", # original 10 classes
     "Model 2 (11 food classes)", # original 10 classes + donuts
     "Model 3 (11 food classes + non-food class)") # 11 classes (same as above) + not_food class
    )
    menu = ['Home', 'About', 'Contact', 'Feedback']
=======

    with st.spinner("The magic of our AI is starting.... Please Wait"):
        model = call_efficient("enetd0")
        my_model = call_model("mymodel")
        tflite_model = call_interpreter(model_path="mymodel/model_unquant.tflite")
        tflite_model_uint8 = call_uint8(model_path="mymodel/model.tflite")
        tflite_colab = call_colab(model_path="mymodel/my_model.tflite")
    

        
    choose_model = st.sidebar.selectbox(
    "Pick model you'd like to use",
    ("Model 1 (Custom Model)", # original 10 classes
     "Model 2 (EfficientNet)", # original 10 classes + donuts
     "Model 3 (Colab)",
     "Model 4 (Quantised Model)",
     "Model 5 (UnQuantised Model)") # 11 classes (same as above) + not_food class
    )
    
    menu = ['Home', 'Stats', 'Contact', 'Feedback']
>>>>>>> Stashed changes
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        # Let's set the title of our awesome web app
        st.title('Title of your Awesome App')
        # Now setting up a header text
        st.subheader("By Your Cool Dev Name")
<<<<<<< Updated upstream
=======
        
        if st.button("Create Server"):
            service = st_capture(createserver())
            if st.button("Delete Server"):
                pass
                
        #sentence = st.text_input('Input your sentence here:')
        sentence = st.text_input('Input your image url here:') 
        image = None
        if sentence:
            try:
                response = requests.get(sentence)
                image = Image.open(io.BytesIO(response.content))
                image = response.content
                #st.write(str(sentence))
            except Exception as e:
                st.error("Exception occured due to url not having image " + str(e))
                image = None
                #st.error()
>>>>>>> Stashed changes
        # Option to upload an image file with jpg,jpeg or png extensions
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])
        
        # When the user clicks the predict button
        if st.button("Predict"):
        # If the user uploads an image
            if uploaded_file is not None:
                # Opening our image
                image = uploaded_file.read()
                print(type(image))
                #image = Image.open(uploaded_file)
                # # Send our image to database for later analysis
                # firebase_bro.send_img(image)
                # Let's see what we got
                st.image(image,use_column_width=True)
                st.write("")
                try:
<<<<<<< Updated upstream
                    #with st.spinner("The magic of our AI has started...."):
                        #label = our_image_classifier(image)
                    label=make_prediction(image)
                        #time.sleep(8)
                    st.success("We predict this image to be: "+label)
=======
                    with st.spinner("The magic of our AI has started...."):
                            #label = our_image_classifier(image)
                        if choose_model == "Model 1 (Custom Model)":
                            #model = call_model()
                            label=make_my_prediction(my_model,image)
                            t = time.time() - start
                        elif choose_model == "Model 2 (EfficientNet)":
                            #my_model = call_my_model()
                            label=make_prediction(model,image)
                            t = time.time() - start
                            #time.sleep(8)
                        elif choose_model == "Model 3 (Colab)":
                            label=getOutput(tflite_colab,prepare_my(image),colab=True)
                            t = time.time() - start
                        elif choose_model == "Model 4 (Quantised Model)":
                            label = getOutput(tflite_model_uint8, input_data)
                            t = time.time() - start
                        elif choose_model == "Model 5 (UnQuantised Model)":
                            #input_data = prepare_my_uint8(image)
                            label = getOutput(tflite_model, prepare_my(image))
                            t = time.time() - start
                        else:
                            #TODO 
                            label = "Not yet done"
                            t = time.time() - start
                    if placeholder:
                        placeholder.empty()
                    st.success("We predict this image to be: "+ label)
                    st.success("Time Taken "+ str(t))
>>>>>>> Stashed changes
                    #rating = st.slider("Do you mind rating our service?",1,10)
                except Exception as e:
                    st.error(e)
                    st.error(traceback.format_exc())
                    st.error("We apologize something went wrong 🙇🏽‍♂️")
            else:
                st.error("Can you please upload an image 🙇🏽‍♂️")

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
        st_tensorboard(logdir= "logs/train/", port=5011, width=1080)
        st.markdown('## Validation')
        st_tensorboard(logdir= "logs/validation/", port=5011, width=1080)
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
