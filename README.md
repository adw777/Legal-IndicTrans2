# Legal English to Hindi Translator

This application translates legal English text to Hindi using LegalLoRA-IndicTrans2, a fine-tuned version of [AI4Bharat's IndicTrans2](https://huggingface.co/ai4bharat/indictrans2-en-indic-1B) optimized for the legal domain.

## Model

The app uses:
- Base model: [ai4bharat/indictrans2-en-indic-1B](https://huggingface.co/ai4bharat/indictrans2-en-indic-1B)
- Fine-tuned LoRA: [axondendriteplus/LegalLoRA-IndicTrans2-en_indic](https://huggingface.co/axondendriteplus/LegalLoRA-IndicTrans2-en_indic)

## Features

- Translate legal English text to Hindi
- Specifically optimized for legal terminology and phrasing
- User-friendly interface with example prompts

## Supported Language

- Hindi (hi)

## Setup Instructions

### Method 1: Using `install.sh`

1. **Clone the repository**:
   ```bash
   git clone https://github.com/adw777/Legal-IndicTrans2.git
   cd Legal-IndicTrans2
   ```

2. **Run the installation script**:
   ```bash
   bash install.sh
   ```

This script will create a conda virtual environment, install all necessary dependencies, and set up the project for you.

### Method 2: Using `requirements.txt`

1. **Clone the repository**:
   ```bash
   git clone https://github.com/adw777/Legal-IndicTrans2.git
   cd Legal-IndicTrans2
   ```

2. **Create and activate a virtual environment**:
   ```bash
   conda create -n legalTrans python=3.9 -y
   conda activate legalTrans
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install additional packages** (if not included in `requirements.txt`):
   ```bash
   python3 -m pip install torch --extra-index-url https://download.pytorch.org/whl/cu118
   python3 -m pip install nltk sacremoses pandas regex mock transformers>=4.33.2 mosestokenizer bitsandbytes scipy accelerate datasets flash-attn>=2.1 sentencepiece peft indic-nlp-library
   ```

5. **Download NLTK data**:
   ```bash
   python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
   ```

### Running the Application

- **Inference**: To test the model, run:
  ```bash
  python inference.py
  ```

- **User Interface**: To launch the UI, run:
  ```bash
  streamlit run streamlit_app.py
  ```

Make sure you have CUDA GPU support enabled for optimal performance.

## Limitations

- Very technical or domain-specific legal terms might not translate perfectly.
- Maximum input length is limited to avoid memory issues.

## Acknowledgments

This app is based on AI4Bharat's IndicTrans2 model and uses the LegalLoRA adaptation for the legal domain.