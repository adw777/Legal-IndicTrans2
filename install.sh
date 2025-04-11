#/bin/bash

root_dir=$(pwd)
echo "Setting up the environment in the $root_dir"

# --------------------------------------------------------------
#          create and activate the virtual environment
# --------------------------------------------------------------
echo "Creating a virtual environment with python3"
conda create -n legalTrans python=3.9 -y
conda activate legalTrans

echo "Installing all the dependencies"
conda install pip
python3 -m pip install --upgrade pip

python3 -m pip install -r requirements.txt
# --------------------------------------------------------------
#                   PyTorch Installation
# --------------------------------------------------------------
python3 -m pip install torch --extra-index-url https://download.pytorch.org/whl/cu118


# --------------------------------------------------------------
#               Install additional utility packages
# --------------------------------------------------------------
python3 -m pip install nltk sacremoses pandas regex mock transformers>=4.33.2 mosestokenizer
python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
python3 -m pip install bitsandbytes scipy accelerate datasets flash-attn>=2.1


# --------------------------------------------------------------
#               Sentencepiece for tokenization
# --------------------------------------------------------------
# build the cpp binaries from the source repo in order to use the command line utility
# source repo: https://github.com/google/sentencepiece
python3 -m pip install sentencepiece

python3 -m pip install peft

python3 -m pip install indic-nlp-library"

# -----------------------------------------------------------------
#       Install IndicTrans2 tokenizer and its dependencies
# -----------------------------------------------------------------
git clone https://github.com/VarunGumma/IndicTransToolkit
cd IndicTransToolkit
python3 -m pip install --editable ./
cd $root_dir

echo "Setup completed!"
