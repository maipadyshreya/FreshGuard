import mysql.connector
from mysql.connector import Error
import bcrypt

class Database:
    def __init__(self, host, user, password, database):
        self.connection = None
        try:
            self.connection = mysql.connector.connect(
                host=host,
                user=user,
                password=password,
                database=database
            )
            if self.connection.is_connected():
                print("Connected to the database")
        except Error as e:
            print(f"Error connecting to database: '{e}'")

    def execute_query(self, query, params=None):
        """Executes any SQL query and returns a buffered cursor."""
        if self.connection is None:
            print("Error: No database connection is available.")
            return None

        try:
            cursor = self.connection.cursor(buffered=True)
            cursor.execute(query, params)

            # Only commit write operations
            write_ops = ("insert", "update", "delete", "create", "drop", "alter")
            if query.strip().lower().startswith(write_ops):
                self.connection.commit()

            return cursor

        except Error as e:
            print(f"Error executing query: '{e}'")
            return None

    def fetch_one(self, query, params=None):
        cursor = self.execute_query(query, params)
        return cursor.fetchone() if cursor else None

    def fetch_all(self, query, params=None):
        cursor = self.execute_query(query, params)
        return cursor.fetchall() if cursor else None

    def close(self):
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("Database connection closed")
