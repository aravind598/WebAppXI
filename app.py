from functools import cache
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import io
from img_classifier import getOutput, prepare_my, prepare_my_uint8, make_my_prediction, make_prediction
import traceback
import copy
#from img_classifier import our_image_classifier
# import firebase_bro

# Just making sure we are not bothered by File Encoding warnings
st.set_option('deprecation.showfileUploaderEncoding', False)

global model
global my_model

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
    return model

@st.experimental_singleton
@cache
def call_interpreter(model_path):
    """[Call tflite interpreter]

    Args:
        model_path ([type]): [description]

    Returns:
        [type]: [description]
    """
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


@st.experimental_memo
@st.cache
@cache
def cache_image(image_byte: bytes, img_shape: int = 224) -> bytes:
    """[Cache the image and makes the image smaller before doing stuff]

    Args:
        image_byte (bytes): [Get from reading the file/image using image.read() or file.read()]
        img_shape (int, optional): [size of the default prediction/image tensor i.e needs to be 224x224 in image size for all my models]. Defaults to 224.

    Returns:
        bytes: [return a new bytes object that is smaller/faster to interpret]
    """
    byteImgIO = io.BytesIO()
    image = Image.open(io.BytesIO(image_byte)).convert('RGB')   
    size = (img_shape, img_shape)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image.save(byteImgIO, format = "JPEG", optimize=True,quality = 70)

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
    page_title = "Title of the webpage",
    layout = "centered",
    page_icon= ":shark:",
    initial_sidebar_state = "collapsed",
    )

    with st.spinner("The magic of our AI has started...."):
        model = call_model("enetd0")
        my_model = call_model("mymodel")
        tflite_model = call_interpreter(model_path="mymodel/model_unquant.tflite")
        tflite_model_uint8 = call_interpreter(model_path="mymodel/model.tflite")
    
    choose_model = st.sidebar.selectbox(
    "Pick model you'd like to use",
    ("Model 1 (Custom Model)", # original 10 classes
     "Model 2 (EfficientNet)", # original 10 classes + donuts
     "Model 3 (11 food classes + non-food class)",
     "Model 4 (Quantised Model)",
     "Model 5 (UnQuantised Model)") # 11 classes (same as above) + not_food class
    )
    
    menu = ['Home', 'About', 'Contact', 'Feedback']
    choice = st.sidebar.selectbox("Menu", menu)
    


    if choice == "Home":
        # Let's set the title of our awesome web app
        st.title('Title of your Awesome App')
        # Now setting up a header text
        st.subheader("By Your Cool Dev Name")
        # Option to upload an image file with jpg,jpeg or png extensions
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])
        
        if uploaded_file is not None:
            placeholder = st.image(copy.copy(uploaded_file).read(),use_column_width=True)
            
        # When the user clicks the predict button
        if st.button("Predict"):
        # If the user uploads an image
            if uploaded_file is not None:
                # Opening our image
                #placeholder = st.image(copy.copy(uploaded_file).read(),use_column_width=True)
                single_image = uploaded_file.read()
                image = cache_image(image_byte = single_image)
                input_data = prepare_my_uint8(image)
                #print(type(image))
                #image = Image.open(uploaded_file)
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
                            label=make_my_prediction(my_model,image)
                        elif choose_model == "Model 2 (EfficientNet)":
                            #my_model = call_my_model()
                            label=make_prediction(model,image)
                            #time.sleep(8)
                        elif choose_model == "Model 4 (Quantised Model)":
                            label = getOutput(tflite_model_uint8, input_data)
                            pass
                        elif choose_model == "Model 5 (UnQuantised Model)":
                            #input_data = prepare_my_uint8(image)
                            label = getOutput(tflite_model, prepare_my(image))
                            pass
                        else:
                            pass
                    placeholder.empty()
                    st.success("We predict this image to be: "+ label)
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

    elif choice == "About":
        # Let's set the title of our About page
        st.title('About us')

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
