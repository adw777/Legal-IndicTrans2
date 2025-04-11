import streamlit as st
import torch
import sys
import time
# import os
# sys.path.append(os.path.join(os.path.dirname(__file__), 'IndicTransToolkit'))

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
from IndicTransToolkit.processor import IndicProcessor
from nltk import sent_tokenize
from mosestokenizer import MosesSentenceSplitter
from indicnlp.tokenize.sentence_tokenize import sentence_split, DELIM_PAT_NO_DANDA

# Set page configuration
st.set_page_config(
    page_title="Legal English to Hindi Translator",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

@st.cache_resource
def load_model_and_tokenizer():
    """Load the model and tokenizer (cached by Streamlit)"""
    with st.spinner("Loading tokenizer..."):
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    
    with st.spinner("Loading base model..."):
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto" if DEVICE == "cuda" else None,
        )
    
    with st.spinner("Applying LoRA weights..."):
        model = PeftModel.from_pretrained(base_model, LORA_MODEL)
        
        if DEVICE == "cuda":
            model = model.half()  # Convert to FP16 for faster inference on GPU
        
        model.eval()
    
    # Initialize IndicProcessor for preprocessing and postprocessing
    ip = IndicProcessor(inference=True)
    
    return tokenizer, model, ip

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

def batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip):
    """Translate a batch of sentences"""
    translations = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(0, len(input_sentences), BATCH_SIZE):
        batch = input_sentences[i : i + BATCH_SIZE]
        # Update progress
        progress = float(i) / float(max(1, len(input_sentences) - BATCH_SIZE))
        progress_bar.progress(progress)
        status_text.text(f"Translating sentence {i+1}/{len(input_sentences)}...")

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

    # Complete the progress bar
    progress_bar.progress(1.0)
    status_text.text("Translation complete!")
    time.sleep(0.5)
    status_text.empty()
    progress_bar.empty()
    
    return translations

def translate_paragraph(input_text, src_lang, tgt_lang, model, tokenizer, ip):
    """Translate a full paragraph by splitting it into sentences and rejoining"""
    # For very long inputs, process in chunks to avoid truncation
    if len(input_text) > 1000:  # If text is very long
        st.info("Long text detected. Processing in chunks for better translation quality.")
        
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
        st.info(f"Processing {len(chunks)} chunks...")
        translated_chunks = []
        for i, chunk in enumerate(chunks):
            st.text(f"Translating chunk {i+1}/{len(chunks)}...")
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
    
    # Load the model and tokenizer
    tokenizer, model, ip = load_model_and_tokenizer()
    
    try:
        # Use the more robust paragraph translation approach
        hindi_translation = translate_paragraph(
            input_text, SRC_LANG, TGT_LANG, model, tokenizer, ip
        )
        
        # Check if translation might be incomplete (basic heuristic)
        if len(hindi_translation) < len(input_text) * 0.5:
            st.warning("Initial translation appears incomplete. Trying alternative method...")
            
            # Fallback to direct approach if translation seems too short
            formatted_input = f"{input_text}</s>{flores_codes[TGT_LANG]}"
            inputs = tokenizer(formatted_input, return_tensors="pt").to(model.device)
            
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
                st.success("Used alternative translation method (longer result).")
                return fallback_translation
        
        return hindi_translation
    
    except Exception as e:
        st.error(f"Error during translation: {str(e)}")
        return f"अनुवाद में त्रुटि: {str(e)}\nकृपया छोटे इनपुट के साथ प्रयास करें।"

def display_file_translation(file):
    """Handle file uploads and translation"""
    try:
        # Read the uploaded file
        stringio = file.getvalue().decode("utf-8")
        
        with st.spinner("Translating uploaded file..."):
            translation = translate_legal_text(stringio)
        
        st.download_button(
            label="Download Translation",
            data=translation.encode('utf-8'),
            file_name=f"translated_{file.name}",
            mime="text/plain"
        )
        
        return translation
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def main():
    # Header section
    st.title("🏛️ Legal English to Hindi Translator")
    st.markdown("""
    Translate legal English text to Hindi using a specialized model fine-tuned for legal domain translation.
    
    **Model details:** [LegalLoRA-IndicTrans2-en_indic](https://huggingface.co/axondendriteplus/LegalLoRA-IndicTrans2-en_indic) - 
    A PEFT/LoRA adaptation of [AI4Bharat's IndicTrans2](https://huggingface.co/ai4bharat/indictrans2-en-indic-1B) for legal domain.
    """)
    
    # Device information
    st.sidebar.title("System Info")
    st.sidebar.info(f"Using device: {DEVICE}")
    
    # Advanced options in sidebar
    st.sidebar.title("Advanced Options")
    max_length = st.sidebar.slider("Maximum Length", 100, 1024, 512, 32, 
                                 help="Maximum length of generated translation")
    do_sample = st.sidebar.checkbox("Use Sampling", False, 
                                  help="Enable for more creative translations")
    temperature = st.sidebar.slider("Temperature", 0.1, 1.5, 0.7, 0.1, 
                                  help="Higher = more creative, only used if sampling is enabled")
    num_beams = st.sidebar.slider("Number of Beams", 1, 10, 5, 1, 
                                help="Higher = more diverse candidates considered")
    
    # About section in sidebar
    st.sidebar.title("About")
    st.sidebar.markdown("""
    This app translates English legal text to Hindi using a specialized model.
    
    **Features:**
    - Optimized for legal terminology
    - Handles complex legal sentences
    - Processes long documents in chunks
    
    **Limitations:**
    - Technical terms might not translate perfectly
    - Maximum input length is limited
    """)
    
    # Input tabs
    tab1, tab2 = st.tabs(["Text Input", "File Upload"])
    
    with tab1:
        # Example selector
        examples = {
            "Select an example": "",
            "Court Order": "The court held that the order was arbitrary and unconstitutional.",
            "Constitutional Petition": "The petitioner seeks relief under Article 32 of the Constitution.",
            "Judgment Status": "The judgment is reserved until further notice.",
            "Jurisdiction": "The High Court has jurisdiction over this matter as per Section 5.",
            "Prima Facie": "The plaintiff failed to establish a prima facie case.",
            "Arbitration Case": "Arbitration and Conciliation Act, 1996 – Ex-parte arbitral awards – Enforcement by employee, when denial of the authenticity of the arbitration agreement by employer – Service dispute by the employee against the State Government.",
        }
        example_choice = st.selectbox("Try with an example:", options=list(examples.keys()))
        
        # Text input area
        if example_choice != "Select an example":
            default_text = examples[example_choice]
        else:
            default_text = ""
            
        input_text = st.text_area("Enter English legal text:", value=default_text, height=200)
        
        col1, col2 = st.columns([1, 5])
        with col1:
            translate_button = st.button("Translate", type="primary")
        with col2:
            st.write("")  # Placeholder for layout
            
        # Translation result
        if translate_button and input_text:
            with st.spinner("Translating..."):
                translation = translate_legal_text(
                    input_text, 
                    max_length=max_length,
                    do_sample=do_sample,
                    temperature=temperature,
                    num_beams=num_beams
                )
            
            st.subheader("Hindi Translation:")
            st.text_area("", value=translation, height=300)
            
            # Provide download button for translation
            st.download_button(
                label="Download Translation",
                data=translation.encode('utf-8'),
                file_name="translation.txt",
                mime="text/plain"
            )
    
    with tab2:
        st.write("Upload an English legal document to translate:")
        uploaded_file = st.file_uploader("Choose a file", type=['txt', 'docx', 'pdf'])
        
        if uploaded_file is not None:
            # Display file details
            file_details = {"Filename": uploaded_file.name, "Filetype": uploaded_file.type, "Filesize": f"{uploaded_file.size / 1024:.2f} KB"}
            st.write(file_details)
            
            # For PDF and DOCX files
            if uploaded_file.type == "application/pdf":
                st.warning("PDF support requires additional libraries. Processing as plain text.")
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                st.warning("DOCX support requires additional libraries. Processing as plain text.")
            
            # Handle translation
            translation = display_file_translation(uploaded_file)
            
            if translation:
                st.subheader("Hindi Translation:")
                st.text_area("", value=translation, height=300)

if __name__ == "__main__":
    main()