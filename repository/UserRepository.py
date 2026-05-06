import os
import sqlite3
from domain.User import User
from repository.RepositoryException import RepositoryException


class UserRepository:
    def __init__(self, db_path):
        self.db_path = db_path
        try:
            with sqlite3.connect(self.db_path) as con:
                cursor = con.cursor()
                cursor.execute('''
                               CREATE TABLE IF NOT EXISTS users
                               (
                                   id       INTEGER PRIMARY KEY AUTOINCREMENT,
                                   username TEXT UNIQUE NOT NULL,
                                   email    TEXT UNIQUE NOT NULL,
                                   password TEXT        NOT NULL
                               )
                               ''')
                con.commit()
                print("User table has been created")
        except sqlite3.Error as e:
            raise RepositoryException(f"Database initialization failed: {str(e)}")

    def login_user(self, username, password):
        with sqlite3.connect(self.db_path) as con:
            cursor = con.cursor()
            cursor.execute('''
                           SELECT id, username, email, password
                           FROM users
                           WHERE username = ?
                             AND password = ?
                           ''', (username, password))

            row = cursor.fetchone()
            if row is None:
                return None

            user = User()

            user.id = row[0]
            user.username = row[1]
            user.email = row[2]
            user.password = row[3]

            return user

    def create(self, user):
        try:
            with sqlite3.connect(self.db_path) as con:
                cursor = con.cursor()
                cursor.execute('''
                               INSERT INTO users (username, email, password)
                               VALUES (?, ?, ?)
                               ''', (user.username, user.email, user.password))
                con.commit()

                user.id = cursor.lastrowid
                return user
        except sqlite3.IntegrityError:
            raise RepositoryException("Username or Email already exists.")
        except sqlite3.Error as e:
            raise RepositoryException(f"User creation failed: {str(e)}")