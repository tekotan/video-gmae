# Writing a Python script to search for core dumps and remove them every second

import os
import time

def search_and_remove_core_dumps():
    # Directory to search for core dumps
    search_directory = '/private/home/jathushan/3D/video_gmae/'

    while True:
        # Searching for core dump files in the directory
        for files in os.listdir(search_directory):
            if files.startswith('core'):
                full_path = os.path.join(search_directory, files)
                print(f'Removing core dump file: {full_path}')
                os.remove(full_path)

        # Wait for 1 second before next iteration
        time.sleep(5)

if __name__ == "__main__":
    search_and_remove_core_dumps()


