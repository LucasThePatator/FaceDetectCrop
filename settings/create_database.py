import sqlite3

conn = sqlite3.connect('labels.db')
query = (''' CREATE TABLE IF NOT EXISTS LABELS
            (FOLDER           TEXT    NOT NULL,
            NAME        CHAR(50) NOT NULL
            );''')
conn.execute(query)
conn.close()