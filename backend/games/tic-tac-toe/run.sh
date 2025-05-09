#!/bin/bash

# Change to a safe directory first
cd /app

# Check if numpy is properly installed
if ! python3 -c "import numpy" &> /dev/null; then
    echo "NumPy not working properly. Reinstalling..."
    pip uninstall -y numpy
    pip install --force-reinstall numpy==1.26.4
fi

# Now run the actual script
python3 /mnt/ahmed/2nd_Year/Second_Term/MP/Project/final\ esp\ integration/docker-it/backend/games/tic-tac-toe/play.py "$@"
