from database import Database
from db_config import DB_CONFIG

def get_database():
    try:
        db = Database(
            host=DB_CONFIG['host'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            database=DB_CONFIG['database']
        )
        if db.connection.is_connected():
            print("[recipe_dao] DB connection successful")
            return db
        else:
            print("[recipe_dao] DB connection failed")
            return None
    except Exception as e:
        print(f"[recipe_dao] Database connection error: {e}")
        return None


def save_recipe(user_id, title, calories, description, ingredients, instructions):
    db = get_database()
    if db is None:
        print("[recipe_dao] Could not connect to DB to save recipe.")
        return

    try:
        cursor = db.connection.cursor()
        insert_query = """
            INSERT INTO saved_recipes
                (user_id, title, calories, description, ingredients, instructions)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        cursor.execute(
            insert_query,
            (user_id, title, calories, description, ingredients, instructions)
        )
        db.connection.commit()
    except Exception as e:
        print(f"[recipe_dao] Error saving recipe: {e}")
    finally:
        try:
            cursor.close()
        except:
            pass


def get_saved_recipes(user_id):
    """Fetch all saved recipes for this user."""
    db = get_database()
    if db is None:
        print("[recipe_dao] Could not connect to DB to load recipes.")
        return []

    try:
        cursor = db.connection.cursor(dictionary=True)
        select_query = """
            SELECT id,
                   title,
                   calories,
                   description,
                   ingredients,
                   instructions,
                   created_at
            FROM saved_recipes
            WHERE user_id = %s
            ORDER BY created_at DESC
        """
        cursor.execute(select_query, (user_id,))
        rows = cursor.fetchall()
        return rows
    except Exception as e:
        print(f"[recipe_dao] Error loading recipes: {e}")
        return []
    finally:
        try:
            cursor.close()
        except:
            pass


def delete_saved_recipe(recipe_id, user_id):
    db = get_database()
    if db is None:
        print("[recipe_dao] Could not connect to DB to delete recipe.")
        return False

    try:
        cursor = db.connection.cursor()
        delete_query = """
            DELETE FROM saved_recipes
            WHERE id = %s AND user_id = %s
        """
        cursor.execute(delete_query, (recipe_id, user_id))
        db.connection.commit()
        return cursor.rowcount > 0  
    except Exception as e:
        print(f"[recipe_dao] Error deleting recipe: {e}")
        return False
    finally:
        try:
            cursor.close()
        except:
            pass
