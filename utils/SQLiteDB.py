import sqlite3 as sql
import re


def createTableIfNotExist():
    sqlConnection = sql.connect(r"SQLiteDb\userData.db")
    print(sqlConnection)

    sqlConnection.execute("""
                        CREATE TABLE IF NOT EXISTS users (
                            id integer primary key autoincrement,
                            username text not null,
                            password text not null
                        );
                    """)


# cursor = sqlConnection.execute("""SELECT name FROM sqlite_master WHERE type='table';""")
# print(cursor.fetchall())

def insertUser(username, password):
    con = sql.connect("SQLiteDb\\userData.db")
    cur = con.cursor()
    cur.execute("INSERT INTO users (username,password) VALUES (?,?)", (username, password))
    con.commit()
    con.close()


def retrieveUsers():
    con = sql.connect("SQLiteDb\\userData.db")
    cur = con.cursor()
    cur.execute("SELECT username, password FROM users")
    users = cur.fetchall()
    con.close()
    return users


def registerUsers(username, email, password):
    con = sql.connect("SQLiteDb\\userData.db")
    cursor = con.cursor()
    cursor.execute('SELECT * FROM accounts WHERE username = % s', (username,))
    account = cursor.fetchone()
    if account:
        msg = 'Account already exists !'
    elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
        msg = 'Invalid email address !'
    elif not re.match(r'[A-Za-z0-9]+', username):
        msg = 'Username must contain only characters and numbers !'
    elif not username or not password or not email:
        msg = 'Please fill out the form !'
    else:
        cursor.execute('INSERT INTO accounts VALUES (NULL, % s, % s, % s)', (username, password, email,))
        con.commit()
        con.close()
        msg = 'You have successfully registered !'
    return msg


def retrieveUsersWithUsername(username):
    con = sql.connect("SQLiteDb\\userData.db")
    cur = con.cursor()
    cur.execute("SELECT * FROM users WHERE username = % s', (username)")
    users = cur.fetchall()
    con.close()
    return users


def executeQuery(query: str):
    con = sql.connect("SQLiteDb\\userData.db")
    cur = con.cursor()
    users = ""
    try:
        cur.execute(query)
        users = cur.fetchall()
    except:
        print("Kindly correct your query")
    con.close()
    return users
