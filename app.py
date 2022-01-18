from functools import cache
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import io
from img_classifier import prediction, prepare
import traceback
#from img_classifier import our_image_classifier
# import firebase_bro

# Just making sure we are not bothered by File Encoding warnings
st.set_option('deprecation.showfileUploaderEncoding', False)

global model
global my_model

@st.experimental_singleton
@cache
def call_model():
    model = tf.keras.models.load_model("enetd0")
    return model


@st.experimental_singleton
@cache
def call_my_model():
    my_model = tf.keras.models.load_model("mymodel")
    return my_model



#@st.experimental_singleton # cache the function so predictions aren't always redone (Streamlit refreshes every click)
def make_prediction(model, image):
    """
    Takes an image and uses model (a trained TensorFlow model) to make a
    prediction.

    Returns:
     image (preproccessed)
     pred_class (prediction class from class_names)
     pred_conf (model confidence)
    """
    image_array = prepare(image,expand_dims=True)
    image_pred = prediction(model,image_array)
    return str(image_pred)

#@st.cache
def make_my_prediction(my_model,image):
    """
    Takes an image and uses model (a trained TensorFlow model) to make a
    prediction.

    Returns:
     image (preproccessed)
     pred_class (prediction class from class_names)
     pred_conf (model confidence)
    """
    #my_model = tf.keras.models.load_model("mymodel")
    image_array = prepare_my(image)
    image_pred = prediction_my(my_model,image_array)
    return str(image_pred)

def prepare_my(bytestr, img_shape=224):
    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    # Replace this with the path to your image
    image = Image.open(io.BytesIO(bytestr)).convert('RGB')
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
    #print(classes[prediction.argmax()])
    return classes[prediction.argmax()]

@st.experimental_memo
@st.cache
@cache
def cache_image(image_byte):
    byteImgIO = io.BytesIO()
    image = Image.open(io.BytesIO(image_byte)).convert('RGB')
    image.save(byteImgIO, format = "JPEG", optimize=True,quality = 70)

    byteImgIO.seek(0)
    image = byteImgIO.read()
    
    return image

@cache
def main():
    # Metadata for the web app
    st.set_page_config(
    page_title = "Title of the webpage",
    layout = "centered",
    page_icon= ":shark:",
    initial_sidebar_state = "collapsed",
    )

    model = call_model()
    my_model = call_my_model()
    
    choose_model = st.sidebar.selectbox(
    "Pick model you'd like to use",
    ("Model 1 (Custom Model)", # original 10 classes
     "Model 2 (11 food classes)", # original 10 classes + donuts
     "Model 3 (11 food classes + non-food class)") # 11 classes (same as above) + not_food class
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
        
        
        # When the user clicks the predict button
        if st.button("Predict"):
        # If the user uploads an image
            if uploaded_file is not None:
                # Opening our image
                single_image = uploaded_file.read()
                image = cache_image(image_byte = single_image)
                #print(type(image))
                #image = Image.open(uploaded_file)
                # # Send our image to database for later analysis
                # firebase_bro.send_img(image)
                # Let's see what we got
                st.image(image,use_column_width=True)
                st.write("")
                try:
                    #with st.spinner("The magic of our AI has started...."):
                        #label = our_image_classifier(image)
                    if choose_model != "Model 1 (Custom Model)":
                        #model = call_model()
                        label=make_prediction(model,image)
                        st.success("We predict this image to be: "+label)
                    else:
                        #my_model = call_my_model()
                        labels=make_my_prediction(my_model,image)
                        st.success("We predict this image to be: "+ labels)
                        #time.sleep(8)
                    
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
