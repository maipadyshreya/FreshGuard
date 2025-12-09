from database import Database
from db_config import DB_CONFIG
from datetime import date

def get_db():
    return Database(
        host=DB_CONFIG['host'],
        user=DB_CONFIG['user'],
        password=DB_CONFIG['password'],
        database=DB_CONFIG['database']
    )

# Add meal for calorie tracking

def add_meal(user_id, meal_name, calories, meal_type, notes=""):
    db = get_db()
    cursor = db.connection.cursor()

    query = """
        INSERT INTO calorie_log (user_id, meal_name, calories, meal_type, notes, log_date)
        VALUES (%s, %s, %s, %s, %s, %s)
    """

    cursor.execute(query, (user_id, meal_name, calories, meal_type, notes, date.today()))
    db.connection.commit()
    return True


def get_user_meals(user_id, log_date=None):
    db = get_db()
    cursor = db.connection.cursor(dictionary=True)

    if log_date is None:
        log_date = date.today()

    query = """
        SELECT id, meal_name, calories, meal_type, notes, created_at
        FROM calorie_log
        WHERE user_id = %s AND log_date = %s
        ORDER BY created_at DESC
    """

    cursor.execute(query, (user_id, log_date))
    return cursor.fetchall()


def delete_meal(meal_id, user_id):
    db = get_db()
    cursor = db.connection.cursor()
    query = "DELETE FROM calorie_log WHERE id = %s AND user_id = %s"
    cursor.execute(query, (meal_id, user_id))
    db.connection.commit()
    return True

def get_total_calories_today(user_id):
    db = get_db()
    cursor = db.connection.cursor()

    query = """
        SELECT COALESCE(SUM(calories), 0)
        FROM calorie_log
        WHERE user_id = %s AND log_date = CURDATE()
    """

    cursor.execute(query, (user_id,))
    result = cursor.fetchone()
    return result[0] if result else 0



