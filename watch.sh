#!/bin/bash

# Make sure a file was provided
if [ $# -eq 0 ]; then
    echo "No arguments supplied. Please provide the path to the file to watch."
    exit 1
fi

watchexec -w $1 -r "crystal $1 && echo '\nDone\n'"
