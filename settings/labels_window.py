import PySimpleGUI as sg
import database_interface

def get_labels():
    labels = database_interface.retrieve_labels()
    return labels

def create():
    labels_array = get_labels()
    headings = ['Folder', 'Name']

    labels_window_layout = [
        [sg.Table(values=labels_array, headings=headings, max_col_width=35,
                    auto_size_columns=True,
                    display_row_numbers=True,
                    justification='right',
                    num_rows=10,
                    key='-TABLE-',
                    row_height=35,
                    tooltip='Labels info')]
    ]

    labels_window = sg.Window("Labels Information Window", 
    labels_window_layout, modal=True)

    while True:
        event, values = labels_window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        if event == '-TABLE-':
            selected_index = values['-TABLE-'][0]
            selected_row = labels_array[selected_index]
            popup_message = "Folder: " + selected_row[0] + "\n" + "Name: " + selected_row[1]
            sg.popup(popup_message)
            print(selected_row)
                    
    labels_window.close()