import PySimpleGUI as sg
import labels_window
import database_interface
import generate_csv
import validation

sg.theme('greentan')
sg.set_options(font=("Microsoft JhengHei", 10))

layout = [[sg.Text("Enter folder name:"), sg.Input(key='-FOLDER-', do_not_clear=True, size=(20, 1))],
          [sg.Text("Enter person name:"), sg.Input(key='-NAME-', do_not_clear=True, size=(20, 1))],
          [sg.Button('Submit Update'), sg.Button('Show Table'), sg.Button('Generate CSV'), sg.Exit()]]

window = sg.Window("Submit Update", layout)

headings = ['Folder, Name']

while True:
    event, values = window.read()
    if event in (sg.WIN_CLOSED, 'Exit'):
        break
    elif event == 'Submit Update':
        validation_result = validation.validate(values)
        if validation_result["is_valid"]:
            database_interface.insert_folder(values['-FOLDER-'], values['-NAME-'])
            sg.popup("Update submitted!")
        else:
            error_message = validation.generate_error_message(validation_result["values_invalid"])
            sg.popup(error_message)
    elif event == 'Show Table':
        labels_window.create()
    elif event == 'Generate CSV':
        generate_csv.create()