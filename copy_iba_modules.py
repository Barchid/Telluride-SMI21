"""
Script used to copy all developed iba_modules into the right directory inside the NRP container.

The script will look into the "iba_modules" folder inside the current repository and move it into
the /home/bbpnrsoa/nrp/src/GazeboRosPackages/src/iba_multimodule_experiment folder
"""
import os
import shutil

SOURCE_DIR = 'iba_modules'
TARGET_DIR = '/home/bbpnrsoa/nrp/src/GazeboRosPackages/src/iba_multimodule_experiment/scripts'

if __name__ == '__main__':
    print(f'COPYING MODULE SCRIPTS FROM [{SOURCE_DIR}] of this repository to the iba module location: {TARGET_DIR}')
    
    for file in os.listdir(SOURCE_DIR):
        source = os.path.join(SOURCE_DIR, file)
        if not os.path.isfile(source):
            print(f'Entry f{source} is not a file. Continue to next file.')

        shutil.copy(source, os.path.join(TARGET_DIR, file))
        print(f'Script {source} -----> {os.path.join(TARGET_DIR, file)}')
