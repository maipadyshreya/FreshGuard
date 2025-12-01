import bcrypt
from database import Database
from db_config import DB_CONFIG

def get_db():
    return Database(**DB_CONFIG)


# register user
def register_user(username, email, first_name, last_name, password, gender, weight, height):
    db = get_db()

    # check if the username or if email already exists
    user_query = "SELECT id FROM users WHERE username = %s"
    email_query = "SELECT id FROM users WHERE email = %s"

    if db.fetch_one(user_query, (username,)):
        return "username_exists"

    if db.fetch_one(email_query, (email,)):
        return "email_exists"

    # hash password to store in db
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    # inseert new user
    insert_query = """
        INSERT INTO users (username, email, first_name, last_name, password, gender, weight, height)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    db.execute_query(insert_query, (username, email, first_name, last_name, hashed, gender, weight, height))

    db.close()
    return "success"

# login
def login_user(username, password):
    db = get_db()

    query = "SELECT id, password FROM users WHERE username = %s"
    row = db.fetch_one(query, (username,))

    if not row:
        return None   

    stored_hash = row[1].encode() if isinstance(row[1], str) else row[1]

    if bcrypt.checkpw(password.encode(), stored_hash):
        return True

    return False

#Get calorie information for user
def get_calorie(user_id):
    db = get_db()

    query = "SELECT gender, weight, height FROM users WHERE id = %s"
    row = db.fetch_one(query, (user_id,))

    db.close()
    return row

#update user information
def update_user_info(user_id, username, email, first_name, last_name, gender, weight, height):
    db = get_db()
    query = """
        UPDATE users 
        SET username = %s, email = %s, first_name = %s, last_name = %s,
            gender = %s, weight = %s, height = %s
        WHERE id = %s
    """
    db.execute_query(query, (username, email, first_name, last_name,
                             gender, weight, height, user_id))
    db.close()
    return True