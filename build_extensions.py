import argparse
import subprocess
import os


parser = argparse.ArgumentParser(description='Build C/C++ extensions')
parser.add_argument('--enable-openmp', action='store_true')
args = parser.parse_args()


if args.enable_openmp:
    cmake_script_path = os.path.join(os.getcwd(), 'pckg', 'pygpc_extensions') 
    cmake_build_path = 'build'
    pygpc_root_path = os.path.join(os.getcwd(), 'pygpc')
    generator_command = ['cmake', '-B' + cmake_build_path, '-H' + cmake_script_path,
               '-DPYGPC_PACKAGE_PATH=' + pygpc_root_path]
    subprocess.run(generator_command)
    build_command = ['cmake', '--build', cmake_build_path] 
    subprocess.run(build_command)
