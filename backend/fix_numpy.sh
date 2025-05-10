#!/bin/bash

# Fix NumPy import issue
echo "Fixing NumPy import issues..."

# Install the correct NumPy version
pip uninstall -y numpy
pip install --force-reinstall numpy==1.26.4

# Check if we're in a numpy source directory
if [ -d "numpy" ] && [ -f "numpy/__init__.py" ]; then
    echo "Warning: Current directory contains a numpy package!"
    echo "Moving out of potential numpy source directory..."
    cd /app
fi

# Add a helper script to prevent this issue
cat > /usr/local/bin/run-fixed-python <<EOF
#!/bin/bash
# Ensure we're not in a numpy source directory
if [ -d "\$(pwd)/numpy" ] && [ -f "\$(pwd)/numpy/__init__.py" ]; then
    cd /app
fi
# Run Python with the provided arguments
python3 "\$@"
EOF

chmod +x /usr/local/bin/run-fixed-python

echo "Fixed environment setup. Please run your Python scripts using 'run-fixed-python' instead of 'python3'"
echo "Example: run-fixed-python games/tic-tac-toe/play.py"
echo ""
echo "Alternatively, ensure you are not in a directory containing a numpy package when running Python"
