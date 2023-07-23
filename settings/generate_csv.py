import database_interface
import csv

def get_labels():
    label_records = database_interface.retrieve_labels()
    return label_records

def create():
    labels_array = get_labels()
    headings = ['Folder', 'Name']

    file = open('labels.csv', 'w', encoding='UTF8', newline='')
    writer = csv.writer(file)

    # write the header
    writer.writerow(headings)

    # write multiple rows
    writer.writerows(labels_array)
    file.close()