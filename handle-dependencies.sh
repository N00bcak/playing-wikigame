# Check for Python version 3.10
if [[ "$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')" != "3.10" ]]; then
    echo "Python 3.10 is required. Please install and use Python 3.10."
    exit 1
fi

# Install required packages
# Can't use requirements.txt because of some shenanigans with deepspeed, flash-attn, and oat.
python -m pip install --upgrade pip
python -m pip install wheel 
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
python -m pip install vllm==0.8.4
python -m pip install oat-llm --no-build-isolation

# Install search dependencies
python -m pip install gem-llm[search]

# TODO: Replace with gem when the build hits PyPI
# git clone git@github.com:N00bcak/gem.git
cd ../gem
git checkout wikigame
python -m pip install -e .
cd ../playing-wikigame

# Fix versioning issues
python -m pip install protobuf==3.20.0 numpy==1.26.4 sympy==1.13.1