import streamlit as st
import tensorflow as tf
import numpy as np
import yaml
import streamlit_authenticator as  stauth
from yaml.loader import SafeLoader

#authenticator
with open('./config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

#Prehashing all plain text passwords once
#stauth.Hasher.hash_passwords(config['credentials'])
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# model
def model_predict(test_image):
    model = tf.keras.models.load_model('trained_model.h5')
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(64,64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    predictions = model.predict(input_arr)
    return np.argmax(predictions)
    

st.sidebar.title("FreshGuard")
app_mode  = st.sidebar.selectbox(" Choose Page",["Log-in","Register","Dashobord","recipes","Recipe suggestion"])

#login page
if(app_mode=="Log-in"):
    authenticator.login()
    
#Register Page
if(app_mode=="Register"):
    email,username,name = authenticator.register_user(captcha=False)
    if(email):
        st.success('User registered successfully')
        with open('../config.yaml', 'w') as file:
            yaml.dump(config, file, default_flow_style=False, allow_unicode=True)


#upload image page
if (app_mode=="Recipe suggestion"):
    st.header("upload image")
    test_image = st.file_uploader("Upload Image",type=["jpg","png","jpeg"])
   
    if(st.button("Show image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
       st.write("test the image!!!")
       result_index=model_predict(test_image)
       with open("labels.txt") as f:
           content = f.readlines()
       label=[]

       for i in content:
           label.append(i[:-1])
       st.success(label[result_index])
       
