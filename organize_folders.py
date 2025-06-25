import os
import shutil
import stat
import time

def on_rm_error(func, path, exc_info):
    """
    Error handler for shutil.rmtree.
    Changes the file to be writable and then tries to unlink it.
    """
    print(f"Error removing file: {path}")
    try:
        os.chmod(path, stat.S_IWRITE)
        os.unlink(path)
    except Exception as e:
        print(f"Exception occurred while trying to remove {path}: {e}")

def is_directory_empty(path):
    """ Check if a directory is empty """
    try:
        return not os.listdir(path)
    except OSError as e:
        print(f"Error accessing directory: {path}, {e}")
        return False

def safe_rmtree(path):
    """ Safely remove a directory tree """
    if os.path.exists(path) and os.path.isdir(path):
        try:
            shutil.rmtree(path, onerror=on_rm_error)
            time.sleep(1)  # Brief pause to let the file system update
        except Exception as e:
            print(f"Exception occurred while trying to remove directory {path}: {e}")

# Delete empty directories in './checkpoints/'
checkpoint_all = list(os.walk('./checkpoints/'))
for path, _, _ in checkpoint_all:
    if is_directory_empty(path):
        safe_rmtree(path)

# Delete folders in './checkpoints/' not present in './args/'
args_folders = set(os.listdir('./args/'))
checkpoint_folders = set(os.listdir('./checkpoints/'))
for folder in checkpoint_folders - args_folders:
    safe_rmtree(os.path.join('./checkpoints/', folder))

# Delete specific directories if they don't correspond to folders in './checkpoints/'
checkpoint_folders = set(os.listdir('./checkpoints/'))
directories = ['./results/', './args/', './scalers/']

for directory in directories:
    for folder in os.listdir(directory):
        if folder not in checkpoint_folders:
            safe_rmtree(os.path.join(directory, folder))

#%% Move old folders

def move_folders(source_dir, target_dir, keywords):
    """
    Move folders that contain any of the specified keywords in their name
    from source_dir to target_dir.
    """
    # Ensure the target directory exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Iterate over the folders in the source directory
    for folder_name in os.listdir(source_dir):
        folder_path = os.path.join(source_dir, folder_name)

        # Check if the current item is a folder and if it contains any keyword
        if os.path.isdir(folder_path) and any(keyword in folder_name for keyword in keywords):
            # Move the folder to the target directory
            shutil.move(folder_path, os.path.join(target_dir, folder_name))
            print(f"Moved '{folder_name}' to '{target_dir}'.")

# Example usage
dir_names = ['results', 'args', 'scalers', 'checkpoints', 'env_results']
target_dir_root = 'C:/Users/esmaeel.mohammadi/Desktop/RecaP/Old Models'
keywords = ['New_IP', 'IP', 'P_and_Me', 'Phosphorous', 'DLinear', 'NLinear', 'Autoformer', 'Transformer', 'Informer']

for dir_name in dir_names:
    source_directory = './' + dir_name + '/'
    target_directory = target_dir_root + '/' + dir_name
    move_folders(source_directory, target_directory, keywords)