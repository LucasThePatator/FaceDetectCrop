import sqlite3
import generate_csv

def insert_folder(folder, name):
    conn = sqlite3.connect('labels.db')
    conn.execute("INSERT INTO LABELS (FOLDER,NAME) \
VALUES (?,?)", (folder,name))
    conn.commit()
    conn.close()

def delete_folder_by_name(folder):
    conn = sqlite3.connect('labels.db')
    conn.execute("DELETE from LABELS where NAME = ?",(folder,))
    conn.close()

def edit_folder_by_name(folder, name):
    conn = sqlite3.connect('labels.db')
    conn.execute("UPDATE LABELS set FOLDER = ? where NAME = ?", (folder, name))
    conn.commit()
    conn.close()

def retrieve_labels():
    results = []
    conn = sqlite3.connect('labels.db')
    cursor = conn.execute("SELECT folder, name from LABELS")
    # records are tuples and need to be converted into an array, whatever the fuck that means
    for row in cursor:
        results.append(list(row))
    return results