import os

def print_directory_contents(directory_path, depth=0):
    try:
        # List all the files and subdirectories in the given directory
        contents = os.listdir(directory_path)

        # Print the contents of the directory at the current depth
        print("    " * depth + f"Contents of directory '{directory_path}':")
        for item in contents:
            item_path = os.path.join(directory_path, item)
            print("    " * (depth + 1) + item)

            # If the item is a directory, recursively call the function for that directory
            if os.path.isdir(item_path) and depth < 4:
                print_directory_contents(item_path, depth + 1)

    except FileNotFoundError:
        print(f"Directory '{directory_path}' not found.")
    except PermissionError:
        print(f"Permission denied for directory '{directory_path}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Replace '/path/to/your/directory' with the actual directory path you want to start from
directory_path = '/path/to/your/directory'
print_directory_contents(directory_path)
