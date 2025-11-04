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
    
print('Hello')
print(st.session_state)

options=["Log-in","Register","Dashboard","Recipes","Recipe suggestion"]

st.sidebar.title("FreshGuard")
index=0
if(st.session_state.get("next_page")!=None):
    index=st.session_state.next_page
    del st.session_state.next_page
    st.session_state["app_mode"]=options[index]
    print('Hello1')
app_mode  = st.sidebar.selectbox(" Choose Page",options, key="app_mode")
print(st.session_state)
print('hi')
print(st.sidebar)

def ensure_login():
    if not st.session_state.authentication_status:
        st.session_state["next_page"]=0
        del st.session_state.app_mode
        st.rerun()

#login page
if(app_mode=="Log-in"):
    image = './app/static/bg1.png'
    css = f'''
        <style>
        .stApp {{
            background-image: url({image});
            background-size: cover;
        }}
        .stForm {{
            background-color: forestgreen;
        }}
    </style>
    '''
    st.markdown(css, unsafe_allow_html=True)
    
    def redirect(arg):
        st.session_state["next_page"]=2
        del st.session_state.app_mode
        st.rerun()
    authenticator.login(callback=redirect)
    if(st.session_state.authentication_status):
        
        authenticator.logout()
    
#Register page
if(app_mode=="Register"):
    #adding some CSS for the image background
    image = './app/static/bg1.png'
    css = f'''
    <style>
        .stApp {{
            background-image: url({image});
            background-size: cover;
        }}
        .stForm {{
            background-color: forestgreen;
        }}
    </style>
    '''
    st.markdown(css, unsafe_allow_html=True)
    
    email,username,name = authenticator.register_user(captcha=False)
    if(email):
        st.success('User registered successfully')
        with open('./config.yaml', 'w') as file:
            yaml.dump(config, file, default_flow_style=False, allow_unicode=True)
        st.session_state["next_page"]=0
        del st.session_state.app_mode
        print('Redirecting')
        print(st.session_state)
        st.rerun()

        
#Dashboard
if(app_mode=="Dashboard"):
    ensure_login()
    def redirect(index):
        st.session_state["next_page"]=index
        del st.session_state.app_mode
    st.button("Recipes", on_click=redirect, args=[3])
    st.button("Recipe Suggestions", on_click=redirect, args=[4])
    
    #adding some CSS for the image background
    image = './app/static/bg1.png'
    css = f'''
    <style>
        .stApp {{
            background-image: url({image});
            background-size: cover;
        }}
    </style>
    '''
    st.markdown(css, unsafe_allow_html=True)

#Recipes
if(app_mode=="Recipes"):
    #adding some CSS for the image background
    image = './app/static/bg2.png'
    css = f'''
    <style>
        .stApp {{
            background-image: url({image});
            background-size: cover;
        }}
    </style>
    '''
    st.markdown(css, unsafe_allow_html=True)
    
    ensure_login()

#upload image page
if (app_mode=="Recipe suggestion"):
    #adding some CSS for the image background
    image = './app/static/bg2.png'
    css = f'''
    <style>
        .stApp {{
            background-image: url({image});
            background-size: cover;
        }}
    </style>
    '''
    st.markdown(css, unsafe_allow_html=True)
    
    ensure_login()
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
       
