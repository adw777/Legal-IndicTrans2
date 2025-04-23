# run @ /IndicTransToolkit
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
import torch
import uvicorn
import io
import base64
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
from IndicTransToolkit.processor import IndicProcessor
from nltk import sent_tokenize
import platform
import nltk
from typing import List, Optional
# Import PyPDF2 for PDF handling
import PyPDF2

# Ensure NLTK data is downloaded
nltk.download('punkt', quiet=True)

# Model paths
BASE_MODEL = "ai4bharat/indictrans2-en-indic-1B"
# LORA_MODEL = "axondendriteplus/LegalLoRA-IndicTrans2-en_indic"
# best one
LORA_MODEL = "axondendriteplus/Legal-IndicTrans2-en_indic"

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

app = FastAPI(title="Legal English to Hindi Translation API")

# Data models
class TranslationRequest(BaseModel):
    text: str
    max_length: int = 512
    do_sample: bool = False
    temperature: float = 0.7
    num_beams: int = 5

class TranslationResponse(BaseModel):
    translation: str
    model_info: dict
    
class FileTranslationResponse(BaseModel):
    translation: str
    original_text: str
    model_info: dict

# Store model and tokenizer in global variables to avoid reloading
tokenizer = None
model = None
processor = None

def extract_text_from_pdf(pdf_bytes):
    """Extract text from PDF bytes"""
    text = ""
    try:
        pdf_file = io.BytesIO(pdf_bytes)
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None

def load_model_and_tokenizer():
    """Load the model and tokenizer (once)"""
    global tokenizer, model, processor
    
    if tokenizer is None:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    
    if model is None:
        print("Loading base model...")
        # Simplified device handling
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        # Explicitly move model to device
        base_model = base_model.to(DEVICE)
    
        print("Applying LoRA weights...")
        model = PeftModel.from_pretrained(base_model, LORA_MODEL)
        model = model.to(DEVICE)  # Ensure model is on the right device
        
        if DEVICE == "cuda" and torch.cuda.is_available():
            try:
                model = model.half()  # Convert to FP16 for faster inference on GPU
            except Exception as e:
                print(f"Could not convert model to half precision: {e}")
        
        model.eval()
    
    if processor is None:
        # Initialize IndicProcessor for preprocessing and postprocessing
        processor = IndicProcessor(inference=True)
    
    return tokenizer, model, processor

def split_sentences(input_text, lang):
    """Split text into sentences based on language - Windows compatible version"""
    if lang == "eng_Latn":
        # Use NLTK tokenizer for English
        try:
            sents_nltk = sent_tokenize(input_text)
            input_sentences = [sent.replace("\xad", "") for sent in sents_nltk]
        except Exception as e:
            # If NLTK fails, fall back to simple period-based splitting
            print(f"Sentence tokenization error: {e}. Using simple splitting.")
            input_sentences = [s.strip() for s in input_text.split('.') if s.strip()]
            if not input_sentences:
                input_sentences = [input_text]
    else:
        # For non-English languages, use the default NLTK tokenizer
        try:
            input_sentences = sent_tokenize(input_text)
        except Exception:
            # Simple fallback
            input_sentences = [s.strip() for s in input_text.split('.') if s.strip()]
            if not input_sentences:
                input_sentences = [input_text]
    
    return input_sentences

def batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip):
    """Translate a batch of sentences"""
    translations = []
    
    try:
        # Get the device that the model is on
        model_device = next(model.parameters()).device
        
        for i in range(0, len(input_sentences), BATCH_SIZE):
            batch = input_sentences[i : i + BATCH_SIZE]
            print(f"Translating sentence {i+1}/{len(input_sentences)}...")

            # Preprocess the batch and extract entity mappings
            batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)

            # Tokenize the batch and generate input encodings
            inputs = tokenizer(
                batch,
                truncation=True,
                padding="longest",
                return_tensors="pt",
            )
            
            # Move inputs to the same device as the model
            inputs = {k: v.to(model_device) for k, v in inputs.items()}

            # Generate translations using the model
            with torch.no_grad():
                generated_tokens = model.generate(
                    **inputs,
                    use_cache=True,
                    min_length=0,
                    max_length=1024,
                    num_beams=5,
                    length_penalty=1.0,
                    no_repeat_ngram_size=3,
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

    except Exception as e:
        print(f"Translation error: {str(e)}")
        if not translations:
            translations = ["Error occurred during translation."]
    
    return translations

def translate_paragraph(input_text, src_lang, tgt_lang, model, tokenizer, ip):
    """Translate a full paragraph by splitting it into sentences and rejoining"""
    # For very long inputs, process in chunks to avoid truncation
    if len(input_text) > 1000:  # If text is very long
        print("Long text detected. Processing in chunks for better translation quality.")
        
        # Split into manageable chunks roughly by sentences
        chunks = []
        input_sentences = split_sentences(input_text, src_lang)
        current_chunk = []
        current_length = 0
        
        for sentence in input_sentences:
            if current_length + len(sentence) < 1000 or not current_chunk:
                current_chunk.append(sentence)
                current_length += len(sentence)
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = len(sentence)
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        # Translate each chunk and join
        print(f"Processing {len(chunks)} chunks...")
        translated_chunks = []
        for i, chunk in enumerate(chunks):
            print(f"Translating chunk {i+1}/{len(chunks)}...")
            sentences = split_sentences(chunk, src_lang)
            translated_text = batch_translate(sentences, src_lang, tgt_lang, model, tokenizer, ip)
            translated_chunks.append(" ".join(translated_text))
            
        return " ".join(translated_chunks)
    else:
        # Normal processing for shorter texts
        input_sentences = split_sentences(input_text, src_lang)
        translated_text = batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip)
        return " ".join(translated_text)

def translate_legal_text(input_text, max_length=512, do_sample=False, temperature=0.7, num_beams=5):
    """Main function to translate legal English text to Hindi"""
    if not input_text.strip():
        return "कृपया अनुवाद के लिए कुछ टेक्स्ट दर्ज करें।"
    
    try:
        # Load the model and tokenizer
        tokenizer, model, ip = load_model_and_tokenizer()
        
        # Use the more robust paragraph translation approach
        hindi_translation = translate_paragraph(
            input_text, SRC_LANG, TGT_LANG, model, tokenizer, ip
        )
        
        # Check if translation might be incomplete (basic heuristic)
        if len(hindi_translation) < len(input_text) * 0.5:
            print("Initial translation appears incomplete. Trying alternative method...")
            
            try:
                # Get model device
                model_device = next(model.parameters()).device
                
                # Fallback to direct approach if translation seems too short
                formatted_input = f"{input_text}</s>{flores_codes[TGT_LANG]}"
                inputs = tokenizer(formatted_input, return_tensors="pt")
                inputs = {k: v.to(model_device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_length=max_length,
                        num_beams=num_beams,
                        do_sample=do_sample,
                        temperature=temperature if do_sample else 1.0,
                        length_penalty=1.0,
                        early_stopping=True
                    )
                
                fallback_translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Use the longer translation
                if len(fallback_translation) > len(hindi_translation):
                    print("Used alternative translation method (longer result).")
                    return fallback_translation
            except Exception as e:
                print(f"Alternative translation method failed: {str(e)}")
        
        return hindi_translation
    
    except Exception as e:
        error_msg = f"Error during translation: {str(e)}"
        print(error_msg)
        return f"अनुवाद में त्रुटि: {str(e)}\nकृपया छोटे इनपुट के साथ प्रयास करें।"

@app.get("/")
def read_root():
    return {
        "message": "Legal English to Hindi Translation API",
        "status": "online",
        "device": DEVICE,
        "platform": platform.system(),
        "pytorch_version": torch.__version__
    }

@app.post("/translate", response_model=TranslationResponse)
def translate(request: TranslationRequest):
    try:
        translation = translate_legal_text(
            request.text,
            max_length=request.max_length,
            do_sample=request.do_sample,
            temperature=request.temperature,
            num_beams=request.num_beams
        )
        
        model_info = {
            "device": DEVICE,
            "base_model": BASE_MODEL,
            "adaptation": LORA_MODEL,
            "src_lang": SRC_LANG,
            "tgt_lang": TGT_LANG
        }
        
        return {"translation": translation, "model_info": model_info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/translate/file", response_model=FileTranslationResponse)
async def translate_file(
    file: UploadFile = File(...),
    max_length: int = Form(1024),
    do_sample: bool = Form(False),
    temperature: float = Form(0.7),
    num_beams: int = Form(5)
):
    try:
        # Read file content
        file_content = await file.read()
        
        # Get file type from filename
        file_name = file.filename.lower()
        
        # Extract text based on file type
        if file_name.endswith('.pdf'):
            extracted_text = extract_text_from_pdf(file_content)
            if not extracted_text:
                raise HTTPException(status_code=400, detail="Could not extract text from PDF file")
        else:
            # For text files, try to decode with different encodings
            try:
                extracted_text = file_content.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    extracted_text = file_content.decode('latin-1')
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Could not decode text file: {str(e)}")
        
        # Translate the extracted text
        translation = translate_legal_text(
            extracted_text,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            num_beams=num_beams
        )
        
        model_info = {
            "device": DEVICE,
            "base_model": BASE_MODEL,
            "adaptation": LORA_MODEL,
            "src_lang": SRC_LANG,
            "tgt_lang": TGT_LANG,
            "file_name": file.filename
        }
        
        return {
            "translation": translation,
            "original_text": extracted_text,
            "model_info": model_info
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "healthy", "device": DEVICE}

def start():
    """Start the API server"""
    # Load model on startup to make first requests faster
    load_model_and_tokenizer()
    
    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    start()