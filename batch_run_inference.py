#!/usr/bin/env python3

import subprocess

def execute_command(command):
    """Execute a shell command."""
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate()
    if process.returncode != 0:
        print(f"Error executing {command}. Error: {err.decode('utf-8')}")
    else:
        print(out.decode('utf-8'))

if __name__ == "__main__":

    commands = [
        "python3 run_inference.py -models adept -datasets all",
        "python3 run_inference.py -models llava -datasets hatful_memes",
        "python3 run_inference.py -models openflamingo -datasets all",
        "python3 run_inference.py -models llava -datasets all",
        "python3 run_inference.py -models blip2 -datasets all",
        "python3 run_inference.py -models instructblip -datasets all"
    ]

    for command in commands:
        execute_command(command)


'''

chmod +x batch_run_inference.py
./batch_run_inference.py

'''