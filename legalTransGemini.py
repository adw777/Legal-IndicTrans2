import os
import argparse
from google import genai
from google.genai import types
import os 
from dotenv import load_dotenv

load_dotenv()

def count_tokens(text):
    """Approximate token count - actual tokenizers vary but roughly 4 chars per token is a reasonable estimate"""
    return len(text) // 4

def chunk_text(text, max_tokens=900000):  # Using 900K to leave room for prompts
    """Split text into chunks respecting the token limit"""
    tokens_estimated = count_tokens(text)
    
    if tokens_estimated <= max_tokens:
        return [text]
    
    # Find a reasonable chunk size in characters
    chars_per_chunk = (len(text) * max_tokens) // tokens_estimated
    
    chunks = []
    current_pos = 0
    text_length = len(text)
    
    while current_pos < text_length:
        # Try to find a natural break point
        end_pos = min(current_pos + chars_per_chunk, text_length)
        
        # If we're not at the end, try to break at a paragraph or sentence
        if end_pos < text_length:
            # Look for paragraph break
            paragraph_break = text.rfind('\n\n', current_pos, end_pos)
            if paragraph_break != -1 and paragraph_break > current_pos + chars_per_chunk // 2:
                end_pos = paragraph_break + 2
            else:
                # Look for sentence break
                sentence_break = text.rfind('. ', current_pos, end_pos)
                if sentence_break != -1 and sentence_break > current_pos + chars_per_chunk // 3:
                    end_pos = sentence_break + 2
        
        chunks.append(text[current_pos:end_pos])
        current_pos = end_pos
    
    return chunks

def translate_english_to_hindi_legal(text, api_key):
    """Translate English text to Hindi in legal context using Gemini"""
    client = genai.Client(api_key=api_key)
    
    system_instruction = """
    You are a professional legal translator specializing in English to Hindi translation for legal documents.
    Translate the provided English text into formal Hindi appropriate for legal contexts.
    Follow these rules strictly:
    1. Output ONLY the Hindi translation, nothing else
    2. Maintain legal terminology accuracy
    3. Preserve the meaning and intent of the original text
    4. Use formal Hindi language appropriate for legal documents
    5. Maintain paragraph structure from the original text
    6. Do not add any comments, explanations, or additional text
    """
    
    # Handle text as chunks if necessary
    chunks = chunk_text(text)
    translation = []
    
    for i, chunk in enumerate(chunks):
        print(f"Translating chunk {i+1}/{len(chunks)}...")
        
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction),
                contents=chunk
            )
            
            translation.append(response.text)
        except Exception as e:
            print(f"Error translating chunk {i+1}: {e}")
            # Add error marker in output
            translation.append(f"[TRANSLATION ERROR FOR CHUNK {i+1}]")
    
    return "\n".join(translation)

def main():
    parser = argparse.ArgumentParser(description='Translate English legal text to Hindi')
    parser.add_argument('--input', '-i', help='Input file path (if not provided, will use stdin)')
    parser.add_argument('--output', '-o', help='Output file path (if not provided, will print to stdout)')
    parser.add_argument('--api-key', help='Gemini API key (if not provided, will use GEMINI_API_KEY env variable)')
    
    args = parser.parse_args()
    
    # Get API key
    # api_key = args.api_key or os.environ.get('GEMINI_API_KEY')
    # api_key = os.getenv("GEMINI_API_KEY")

    api_key = "AIzaSyDizZsf_WytJf8qsA7F_ihTHZRLwb1Mz40"

    if not api_key:
        print("Error: Gemini API key not provided. Use --api-key or set GEMINI_API_KEY environment variable.")
        return
    
    # Get input text
    if args.input:
        try:
            with open(args.input, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            print(f"Error reading input file: {e}")
            return
    else:
        print("Enter/paste English text (Ctrl+D or Ctrl+Z on a new line to finish):")
        text_lines = []
        try:
            while True:
                try:
                    line = input()
                    text_lines.append(line)
                except EOFError:
                    break
        except KeyboardInterrupt:
            print("\nInput interrupted.")
            return
        
        text = '\n'.join(text_lines)
    
    # Translate
    hindi_translation = translate_english_to_hindi_legal(text, api_key)
    
    # Output
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(hindi_translation)
            print(f"Translation saved to {args.output}")
        except Exception as e:
            print(f"Error writing to output file: {e}")
            print("Translation output:")
            print(hindi_translation)
    else:
        print("\nHindi Translation:")
        print("=" * 40)
        print(hindi_translation)

if __name__ == "__main__":
    main()