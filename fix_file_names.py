import os

def rename_files(folder_path, replacements, file_type):
    files = os.listdir(folder_path)

    for file_name in files:
        file_path = os.path.join(folder_path, file_name)

        # Check if the file is a regular file (not a directory)
        if os.path.isfile(file_path) and file_type in file_name:
            new_file_name = file_name
            for search_string, replace_string in replacements.items():
                new_file_name = new_file_name.replace(search_string, replace_string)
                
            new_file_path = os.path.join(folder_path, new_file_name)
            try:
                os.rename(file_path, new_file_path)
            except FileNotFoundError and FileExistsError:
                new_file_name += ' (2)'
                new_file_path = os.path.join(folder_path, new_file_name)
                os.rename(file_path, new_file_path)
            print(f"Renamed file: {file_name} -> {new_file_name}")

setting = 'LSTM_New_CorrH_11F_timeF_Unscaled_240Seq_0Label_16Batch_1e-06LR_256Hidden_2LayerDim'
folder_path = './env_results/' + setting + '/'

do_checkpoints = True
do_env_results = True

replacements = {
    'Epochs': 'Ep',
    '_mse': '',
    '_best_loss':'',
    '0.5SS': 'SS',
    'Small':'SB',
    '_ActA':'',
    'dilate_all':'d_all'
}

# rename_files(folder_path, replacements, file_type)

if do_env_results:
    file_type = 'pkl'
    try:
        for folder in os.listdir('./env_results/'):
            rename_files('./env_results/' + folder + '/', replacements, file_type)
    except Exception as e:
        pass

if do_checkpoints:
    file_type = 'pth'
    try:
        for folder in os.listdir('./checkpoints/'):
            rename_files('./checkpoints/' + folder + '/', replacements, file_type)
    except Exception as e:
        pass

def delete_files_with_string_in_name(directory, string):
    """
    Deletes files with a specific string in their names within a directory and its subdirectories.
    
    :param directory: The root directory to start the search.
    :param string: The string to search for in the file names.
    """
    for root, _, files in os.walk(directory):
        for file in files:
            if string in file:
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Deleted file: {file_path}")


directory = "./env_results/"
string_to_search = "1Ep"

delete_files_with_string_in_name(directory, string_to_search)