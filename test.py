import streamlit as st
import tensorflow as tf
import numpy as np
import bcrypt
from database import Database
from db_config import DB_CONFIG
import requests 
# for the token part
import re


import user_dao
from user_dao import register_user,get_calorie,login_user,update_user_info

import recipe_dao
from recipe_dao import get_saved_recipes, save_recipe,  delete_saved_recipe

#from calorie
from datetime import date

#API CALL for recipes
CLIENT_ID = st.secrets["CLIENT_ID"]
CLIENT_SECRET = st.secrets["CLIENT_SECRET"]


def get_todays_calorie():
    return f"calories_{date.today().isoformat()}"


# model
def model_predict(test_image):
    model = tf.keras.models.load_model("trained_model.h5", compile=False)
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(64,64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

#database connection
def get_database():
    try:
     db = Database(
        host=DB_CONFIG['host'],
        user=DB_CONFIG['user'],
        password=DB_CONFIG['password'],
        database=DB_CONFIG['database']
     )
   
    
     success = db.connection.is_connected()
     if success:
        print("Connection successful")
        return db
     else:
        print("Connection failed")
        return None
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return None
    
# recipes
def get_token():
    url = "https://oauth.fatsecret.com/connect/token"
    data = {
        "grant_type": "client_credentials",
        "scope": "basic",
    }

    resp = requests.post(url, data=data, auth=(CLIENT_ID, CLIENT_SECRET))
    resp.raise_for_status()
    return resp.json()["access_token"]


def get_foods(query=None, page_number=0, max_results=20, include_recipe_details=True):
    token = get_token()

    search_url = "https://platform.fatsecret.com/rest/recipes/search/v3"
    headers = {
        "Authorization": f"Bearer {token}",
    }
    
    if max_results < 1:
        max_results = 1
    if max_results > 50:
        max_results = 50

    # Default search recipes for chicken
    if not query or not query.strip():
        search_expression = "chicken"
    else:
        search_expression = query.strip()

    params = {
        "search_expression": search_expression,
        "page_number": page_number,
        "max_results": max_results,
        "format": "json",
    }

    resp = requests.get(search_url, headers=headers, params=params)
    resp.raise_for_status()
    data = resp.json()

    if "error" in data:
        msg = data["error"].get("message", "Unknown API error")
        st.error(f"API error: {msg}")
        return [], 0

    recipes_root = data.get("recipes") or {}
    recipe_list = recipes_root.get("recipe")

    
    if isinstance(recipe_list, dict):
        recipe_list = [recipe_list]
    elif recipe_list is None:
        recipe_list = []

    total_results = recipes_root.get("total_results", len(recipe_list))
    try:
        total_results = int(total_results)
    except Exception:
        total_results = len(recipe_list)

    results = []

    for r in recipe_list:
        recipe_id = r.get("recipe_id")
        name = r.get("recipe_name", "Unknown")
        description = r.get("recipe_description", "") or ""
        image_url = r.get("recipe_image", "")
        recipe_url = r.get("recipe_url", "") if isinstance(r.get("recipe_url", ""), str) else ""

        ingredients_text = ""
        instructions_text = ""
        calories_val = "N/A"   
       
        if include_recipe_details and recipe_id:
            try:
                detail_url = "https://platform.fatsecret.com/rest/recipe/v2"
                detail_params = {
                    "recipe_id": recipe_id,
                    "format": "json",
                }
                detail_resp = requests.get(detail_url, headers=headers, params=detail_params)
                detail_resp.raise_for_status()
                detail_data = detail_resp.json()

                recipe_obj = detail_data.get("recipe") or {}

                # recipe descrpition
                detailed_desc = recipe_obj.get("recipe_description") or ""
                if detailed_desc:
                    description = detailed_desc

                
                # calories
                serving_sizes = recipe_obj.get("serving_sizes")
                first_serving = None

                if isinstance(serving_sizes, dict):
                    raw_serving = serving_sizes.get("serving")
                    if isinstance(raw_serving, list) and raw_serving:
                        first_serving = raw_serving[0]
                    elif isinstance(raw_serving, dict):
                        first_serving = raw_serving

                if isinstance(first_serving, dict):
                    c = first_serving.get("calories")
                    if c is not None:
                        calories_val = str(c)

                # ingredients
                ing_list = []
                ingredients_section = recipe_obj.get("ingredients")

                if isinstance(ingredients_section, dict):
                    raw_ings = ingredients_section.get("ingredient", [])
                    if isinstance(raw_ings, dict):
                        raw_ings = [raw_ings]
                    elif isinstance(raw_ings, str):
                        raw_ings = [raw_ings]
                else:
                    raw_ings = []

                for ing in raw_ings:
                    if isinstance(ing, dict):
                        text = ing.get("ingredient_description") or ing.get("food_name") or ""
                    else:
                        text = str(ing)
                    if text:
                        ing_list.append(text)

                ingredients_text = "\n".join(ing_list)

                # instructions
                dir_list = []
                directions_section = recipe_obj.get("directions")

                if isinstance(directions_section, dict):
                    raw_dirs = directions_section.get("direction", [])
                    if isinstance(raw_dirs, dict):
                        raw_dirs = [raw_dirs]
                    elif isinstance(raw_dirs, str):
                        raw_dirs = [raw_dirs]
                else:
                    raw_dirs = []

                for d in raw_dirs:
                    if isinstance(d, dict):
                        num = d.get("direction_number", "")
                        ddesc = d.get("direction_description", "")
                        if num:
                            dir_list.append(f"{num}. {ddesc}")
                        else:
                            dir_list.append(ddesc)
                    else:
                        dir_list.append(str(d))

                instructions_text = "\n".join(dir_list)

            except Exception as e:
               
                print(f"Error loading details for recipe {recipe_id}: {e}")

        
        results.append({
            "title": name,
            "calories": calories_val,
            "description": description,
            "url": recipe_url,
            "image": image_url,
            "ingredients": ingredients_text,
            "instructions": instructions_text,
        })

    return results, total_results


#log out
def logout():
    try:
        db = get_database()
        cursor = db.connection.cursor()
        update_query = "UPDATE users SET logged_in = FALSE WHERE id = %s"
        cursor.execute(update_query, (st.session_state["user_id"],))
        db.connection.commit()
    except:
        pass
    
    # Clear session
    for key in list(st.session_state.keys()):
        del st.session_state[key]

    st.session_state["next_page"] = 0
    


print(st.session_state)

if not st.session_state.get("logged_in"):
    options = ["Log-in", "Register"]
else:
    st.sidebar.markdown(f"### Welcome, **{st.session_state['username']}**")
    options = ["Dashboard", "Recipes", "Recipe suggestion"]
    st.sidebar.button("Logout", on_click=logout)

st.sidebar.title("FreshGuard")
index=0
if(st.session_state.get("next_page")!=None):
    index=st.session_state.next_page
    del st.session_state.next_page
    st.session_state["app_mode"]=options[index]

app_mode  = st.sidebar.selectbox(" Choose Page",options, key="app_mode")


def ensure_login():
    if not st.session_state.get("logged_in"):
        st.session_state["next_page"] = 0
        del st.session_state["app_mode"]
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
    st.title("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        db = get_database()
        if db is None:
            st.error("Unable to connect to the database. Please try again later.")
        else:
            try:
                cursor = db.connection.cursor(dictionary=True)
               
                query = "SELECT id, username, password FROM users WHERE username = %s"
                cursor.execute(query, (username,))
                user = cursor.fetchone()

                if user is None:
                    st.error("User does not exist")
               
                else:
                    stored_password = user['password'].encode() 
                    if isinstance(stored_password, str):
                        stored_hash = stored_password.encode()

                    # Encode entered password
                    entered_password = password.encode()
                    if bcrypt.checkpw(entered_password, stored_password):
                            st.success("Login successful!")

                            # Update logged in flag in DB
                            update_query = "UPDATE users SET logged_in = TRUE WHERE id = %s"
                            cursor.execute(update_query, (user['id'],))
                            db.connection.commit()

                            # Start session
                            st.session_state["logged_in"] = True
                            st.session_state["username"] = user["username"]
                            st.session_state["user_id"] = user["id"]
                            st.session_state["next_page"] = 2

                              # Ensure new mode loads
                            del st.session_state["app_mode"]
                            st.rerun()

                    else:
                        st.error("Incorrect password.")

            except Exception as e:
                st.error(f"Database error: {e}")

     # REGISTER PAGE
if app_mode == "Register":

    st.title("Create an account")

    username   = st.text_input("Username")
    email      = st.text_input("Email")
    first_name = st.text_input("First Name")
    last_name  = st.text_input("Last Name")
    password   = st.text_input("Password", type="password")
    confirm    = st.text_input("Confirm Password", type="password")

    gender = st.selectbox("Gender", ["Male", "Female"])
    weight = st.number_input("Weight (kg)", step=0.1)
    height = st.number_input("Height (cm)", step=0.1)
    if st.button("Register"):

        if password != confirm:
            st.error("Passwords do not match")
        else:
           result = register_user(username, email, first_name, last_name, password, gender, weight, height)
           if result == "success":
                st.success("Registration successful! Please login.")
                st.session_state["next_page"] = 0
                del st.session_state["app_mode"]
                st.rerun()

           elif result == "username_exists":
                st.error("Username already taken.")

           elif result == "email_exists":
                st.error("Email already in use.")


           
   
#Dashboard
if(app_mode=="Dashboard"):
    ensure_login()
   
    db = get_database()
    cursor = db.connection.cursor(dictionary=True)
    cursor.execute("""
        SELECT username, email, first_name, last_name, gender, weight, height
        FROM users WHERE id = %s
    """, (st.session_state["user_id"],))
    profile = cursor.fetchone()

    username = profile["username"]
    email = profile["email"]
    first_name = profile["first_name"]
    last_name = profile["last_name"]
    gender = profile["gender"]
    weight = float(profile["weight"])
    height = float(profile["height"])
    # calculate daily calories
    if weight > 0 and height > 0:
     if gender == "Male":
            bmr = (10 * weight) + (6.25 * height) + 5
     else:
            bmr = (10 * weight) + (6.25 * height) - 161

     daily_calories = bmr * 1.375

    else:
        daily_calories = 0
        
    today_calorie = get_todays_calorie()
    calories_consumed = st.session_state.get(today_calorie, 0)
    calories_left = max(daily_calories - calories_consumed, 0.0)

    # edit user info
    if st.session_state.get("editing_user_info", False):

        st.subheader("Update Your Profile")

        new_username = st.text_input("Username", username)
        new_email = st.text_input("Email", email)
        new_first = st.text_input("First Name", first_name)
        new_last = st.text_input("Last Name", last_name)

        new_gender = st.selectbox("Gender", ["Male", "Female"],
                                  index=(0 if gender == "Male" else 1))
        new_weight = st.number_input("Weight (kg)", step=0.1, value=weight)
        new_height = st.number_input("Height (cm)", step=0.1, value=height)

        if st.button("Save Changes"):
            update_user_info(
                st.session_state["user_id"],
                new_username,
                new_email,
                new_first,
                new_last,
                new_gender,
                new_weight,
                new_height
            )
            st.success("Information updated successfully!")
            st.session_state["editing_user_info"] = False
            st.rerun()
    else:
        #edit button
         if st.button("Edit User Information"):
           st.session_state["editing_user_info"] = True
           st.rerun()
#display the profile information
         with st.container():
          st.markdown("""
        <div style="
            background-color: #f0f0f0;
            padding: 20px;
            margin-bottom: 10px;
            border-radius: 12px;
            width: 350px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        ">
            <h4 style="margin-bottom: 10px;">Your Profile</h4>
            <p><b>Username:</b> """ + username + """</p>
            <p><b>Email:</b> """ + email + """</p>
            <p><b>Name:</b> """ + first_name + " " + last_name + """</p>
            <p><b>Gender:</b> """ + gender + """</p>
            <p><b>Weight:</b> """ + str(weight) + """ kg</p>
            <p><b>Height:</b> """ + str(height) + """ cm</p>
            
    """, unsafe_allow_html=True)
         with st.container():
               st.markdown(f"""
        <div style="
            background-color: #f0f0f0;
            padding: 10px;
            border-radius: 12px;
            width: 300px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        ">
         <h4 style="margin-bottom: 10px;">Calorie count</h4>
          <p><b>Daily Calories:</b> {daily_calories:.0f} cal</p>
          <p><b>Eaten Today:</b> {calories_consumed:.0f} cal</p>
          <p><b>Remaining:</b> {calories_left:.0f} cal</p>
        </div>
         """, unsafe_allow_html=True)
         

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

    #saved recipes
    st.subheader("Your Saved Recipes")

    saved_recipes = get_saved_recipes(st.session_state["user_id"])

    if not saved_recipes:
        st.caption("You haven't saved any recipes yet.")
    else:
        for r in saved_recipes:
            with st.container():
                st.markdown(
                    f"""
                    <div style="
                        background-color: #ffffff;
                        padding: 12px 16px;
                        border-radius: 12px;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                        margin-bottom: 10px;
                        max-width: 500px;
                    ">
                        <h4 style="margin: 0 0 4px 0;">{r['title']}</h4>
                        <p style="margin: 0;"><b>Calories:</b> {r['calories']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                with st.expander("View details"):
                 if r.get("description"):
                    st.markdown("**Description**")
                    st.write(r["description"])

                 if r.get("ingredients"):
                    st.markdown("**Ingredients**")
                    st.write(r["ingredients"])

                 if r.get("instructions"):
                    st.markdown("**Instructions**")
                    st.write(r["instructions"])

                 st.caption(f"Saved on: {r['created_at']}")
                 if st.button("Delete recipe", key=f"delete_saved_{r['id']}"):
                    ok = delete_saved_recipe(r["id"], st.session_state["user_id"])
                    if ok:
                     st.success("Recipe deleted.")
                    else:
                     st.error("Could not delete recipe.")
                    st.rerun()

#Recipes
if (app_mode == "Recipes"):
    ensure_login()

    # background
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

    st.title("Recipes")

    # Initiaize the session state variables
    if "recipes_query" not in st.session_state:
        st.session_state["recipes_query"] = ""
    if "recipes_page" not in st.session_state:
        st.session_state["recipes_page"] = 0

    #
    default_query = st.session_state.get("recipes_query") or st.session_state.get("predicted_ingredient", "")

    query = st.text_input(
        "Search recipes:",
        value=default_query
    )

    # Buttons for search and reset the recipe page
    col1, col2, col3 = st.columns(3)
    with col1:
        search_clicked = st.button("Search")
    with col2:
        reset_clicked = st.button("Reset")
    with col3:
        st.write("") 

    
    if search_clicked:
        st.session_state["recipes_query"] = query.strip()
        st.session_state["recipes_page"] = 0

    if reset_clicked:
        st.session_state["recipes_query"] = ""
        st.session_state["recipes_page"] = 0

    current_query = st.session_state["recipes_query"]
    current_page = st.session_state["recipes_page"]
    max_per_page = 50

    # get foods 
    try:
        foods, total_results = get_foods(
            query=current_query if current_query else None,
            page_number=current_page,
            max_results=max_per_page,
            include_recipe_details=True
        )
    except Exception as e:
        st.error(f"Error fetching recipes: {e}")
        foods, total_results = [], 0

    # recipe search
    if current_query:
        st.subheader(f"Showing results for: '{current_query}' (page {current_page + 1})")
    else:
        st.subheader(f"Showing all recipes")

    # if no result
    if not foods:
        st.info("No recipes found for this page.")
    else:
        # Display recipe if there is result
        for idx, food in enumerate(foods):
            with st.container():
                
                img_html = ""
                if food.get("image"):
                    img_html = (
                        f"<img src='{food['image']}' "
                        "width='50' height='50' "
                        "style='border-radius: 8px; margin-right: 12px; object-fit: cover;' />"
                    )

                
                card_html = f"""<div style="background-color: #ffffff;
padding: 12px 16px;
border-radius: 12px;
box-shadow: 0 2px 8px rgba(0,0,0,0.08);
margin-bottom: 10px;
display: flex;
align-items: center;">
{img_html}
<div>
  <h4 style="margin: 0 0 4px 0;">{food['title']}</h4>
  <p style="margin: 0;"><b>Calories:</b> {food['calories']}</p>
</div>
</div>"""

                st.markdown(card_html, unsafe_allow_html=True)

              
                with st.expander(f"More about {food['title']}"):
                    if food.get("description"):
                        st.markdown("**Description**")
                        st.write(food["description"])
                    else:
                        st.write("No description available.")

                    if food.get("ingredients"):
                        st.markdown("**Ingredients**")
                        st.write(food["ingredients"])

                    if food.get("instructions"):
                        st.markdown("**Instructions**")
                        st.write(food["instructions"])

                   
                # Save recipe button 
                if st.button(
                    "Save recipe",
                    key=f"save_recipe_{current_page}_{idx}"
                ):
                    save_recipe(
                        st.session_state["user_id"],
                        food["title"],
                        str(food["calories"]),
                        food.get("description", ""),
                        food.get("ingredients", ""),
                        food.get("instructions", "")
                    )
                    st.success("Recipe saved!")

                try:
                    cal_val = float(food["calories"])
                except:
                    cal_val = None

                if cal_val is not None:
                    if st.button(
                        "Add to today's calorie count",
                        key=f"add_cal_{current_page}_{idx}"
                    ):
                        today_cal = get_todays_calorie()
                        current = st.session_state.get(today_cal, 0.0)
                        st.session_state[today_cal] = current + cal_val
                        st.success(f"Added {cal_val:.0f} cal to today's total!")
                else:
                    st.caption("Calories not available for this recipe.")

    # recipe pages
    total_pages = (total_results // max_per_page) + (1 if total_results % max_per_page else 0)
    if total_pages == 0:
        total_pages = 1

    col_prev, col_page, col_next = st.columns(3)
    with col_prev:
        if st.button("Previous page") and current_page > 0:
            st.session_state["recipes_page"] -= 1
            st.rerun()
    with col_page:
        st.write(f"Page {current_page + 1} of {total_pages}")
    with col_next:
        if st.button("Next page") and (current_page + 1) < total_pages:
            st.session_state["recipes_page"] += 1
            st.rerun()


#upload image page
if (app_mode=="Recipe suggestion"):
    ensure_login()
    
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
    
    #ensure_login()
    st.header("Upload image to get recipe suggestion")
    test_image = st.file_uploader("Upload Image",type=["jpg","png","jpeg"])
   
    if(st.button("Show image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
       if test_image is None:
            st.error("Please upload an image first.")
       else:
        result_index=model_predict(test_image)
        with open("labels.txt") as f:
           content = f.readlines()
        label=[]

       for i in content:
           label.append(i[:-1])
       ingredient = label[result_index]
       st.success(f"Detected ingredient: {ingredient}")
       
       st.session_state["predicted_ingredient"] = ingredient
       st.session_state["recipes_query"] = ingredient
       st.session_state["recipes_page"] = 0  

           
       st.session_state["next_page"] = 1   

            
       if "app_mode" in st.session_state:
                del st.session_state["app_mode"]
       st.rerun()
