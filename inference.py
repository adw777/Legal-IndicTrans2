import sys
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
from IndicTransToolkit.processor import IndicProcessor
from nltk import sent_tokenize
from mosestokenizer import MosesSentenceSplitter

# Model paths
BASE_MODEL = "ai4bharat/indictrans2-en-indic-1B"
LORA_MODEL = "axondendriteplus/LegalLoRA-IndicTrans2-en_indic"

# Configuration
BATCH_SIZE = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SRC_LANG = "eng_Latn"
TGT_LANG = "hin_Deva"

# FLORES language code mapping to 2 letter ISO language code
flores_codes = {
    "eng_Latn": "en",
    "hin_Deva": "hi",
}

def split_sentences(input_text, lang):
    """Split text into sentences based on language"""
    if lang == "eng_Latn":
        # For English, use both NLTK and Moses splitters and pick the one with fewer sentences
        with MosesSentenceSplitter(flores_codes[lang]) as splitter:
            sents_moses = splitter([input_text])
        sents_nltk = sent_tokenize(input_text)
        if len(sents_nltk) < len(sents_moses):
            input_sentences = sents_nltk
        else:
            input_sentences = sents_moses
        input_sentences = [sent.replace("\xad", "") for sent in input_sentences]
    else:
        # For non-English languages, use the default NLTK tokenizer
        input_sentences = sent_tokenize(input_text)
    
    return input_sentences

def initialize_model_and_tokenizer():
    """Initialize the base model with LoRA fine-tuning"""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    
    print("Loading base model...")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="auto" if DEVICE == "cuda" else None,
    )
    
    print("Applying LoRA weights...")
    model = PeftModel.from_pretrained(base_model, LORA_MODEL).to(DEVICE)
    
    if DEVICE == "cuda":
        model = model.half()  # Convert to FP16 for faster inference on GPU
    
    model.eval()
    print("Model loaded successfully!")
    
    return tokenizer, model

def batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip):
    """Translate a batch of sentences"""
    translations = []
    for i in range(0, len(input_sentences), BATCH_SIZE):
        batch = input_sentences[i : i + BATCH_SIZE]

        # Preprocess the batch and extract entity mappings
        batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)

        # Tokenize the batch and generate input encodings
        inputs = tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(DEVICE)

        # Generate translations using the model
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=512,
                num_beams=5,
                num_return_sequences=1,
            )

        # Decode the generated tokens into text
        with tokenizer.as_target_tokenizer():
            generated_tokens = tokenizer.batch_decode(
                generated_tokens.detach().cpu().tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

        # Postprocess the translations, including entity replacement
        translations += ip.postprocess_batch(generated_tokens, lang=tgt_lang)

        del inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return translations

def translate_paragraph(input_text, src_lang, tgt_lang, model, tokenizer, ip):
    """Translate a full paragraph by splitting it into sentences and rejoining"""
    input_sentences = split_sentences(input_text, src_lang)
    translated_text = batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip)
    return " ".join(translated_text)

def translate_legal_text(input_text):
    """Main function to translate legal English text to Hindi"""
    # Initialize IndicProcessor for preprocessing and postprocessing
    ip = IndicProcessor(inference=True)
    
    # Initialize model and tokenizer
    tokenizer, model = initialize_model_and_tokenizer()
    
    # Translate the input text
    hindi_translation = translate_paragraph(
        input_text, SRC_LANG, TGT_LANG, model, tokenizer, ip
    )
    
    return hindi_translation

def translate_file(input_file_path, output_file_path=None):
    """Translate text from a file and optionally save to another file"""
    with open(input_file_path, 'r', encoding='utf-8') as file:
        input_text = file.read()
    
    translation = translate_legal_text(input_text)
    
    if output_file_path:
        with open(output_file_path, 'w', encoding='utf-8') as file:
            file.write(translation)
        print(f"Translation saved to {output_file_path}")
    
    return translation

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    
    # Check if file paths are provided as command-line arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "translation_output.txt"
        
        translation = translate_file(input_file, output_file)
    else:
        # Example legal text for demonstration
        legal_text = """Ex-parte arbitral awards – Enforcement by employee, when denial of the authenticity of the arbitration agreement by employer – Service dispute by the employee against the State Government and the government hospital where he was employed as regards age of superannuation"""
        
        print("\nTranslating legal text from English to Hindi...\n")
        
        translation = translate_legal_text(legal_text)
        
        print(f"Original English text:\n{legal_text}\n")
        print(f"Hindi translation:\n{translation}")

# python example.py 
# for translating a complete .txt file - python example.py input.txt output.txt