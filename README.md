---
title: LegalLoRA-IndicTrans2-en_indic
emoji: ⚖️
colorFrom: blue
colorTo: green
sdk: streamlit
python_version: '3.9'
sdk_version: 1.44.1
suggested_hardware: cpu-upgrade
suggested_storage: medium
app_file: streamlit_app.py
license: mit
---
# Legal English to Hindi Translator

This application translates legal English text to Hindi using LegalLoRA-IndicTrans2, a fine-tuned version of [AI4Bharat's IndicTrans2](https://huggingface.co/ai4bharat/indictrans2-en-indic-1B) optimized for the legal domain.

## Model

The app uses:
- Base model: [ai4bharat/indictrans2-en-indic-1B](https://huggingface.co/ai4bharat/indictrans2-en-indic-1B)
- Fine-tuned LoRA: [axondendriteplus/LegalLoRA-IndicTrans2-en_indic](https://huggingface.co/axondendriteplus/LegalLoRA-IndicTrans2-en_indic)

## Features

- Translate legal English text to Hindi
- Specifically optimized for legal terminology and phrasing
- Advanced generation options (beam search, sampling, temperature)
- User-friendly interface with example prompts

## Supported Language

- Hindi (hi)

## Usage

1. Enter your legal English text in the input box
2. Click "Translate"
3. Copy the Hindi translated text from the output box

## Advanced Options

- **Maximum Length**: Control the maximum length of the generated translation
- **Use Sampling**: Toggle between deterministic (beam search) and probabilistic generation
- **Temperature**: Control randomness in generation (higher = more creative)
- **Number of Beams**: Adjust beam search width for deterministic generation

## Limitations

- Very technical or domain-specific legal terms might not translate perfectly
- Maximum input length is limited to avoid memory issues

## Acknowledgments

This app is based on AI4Bharat's IndicTrans2 model and uses the LegalLoRA adaptation for the legal domain.