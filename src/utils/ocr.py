import io
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from src.utils.logger import setup_logger
import base64

logger = setup_logger(__name__)

def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

def extract_text_from_file(file_content: bytes, filename: str) -> str:
    """
    Extracts text from a file (TXT, PDF, or Image).
    For Images and PDFs, it uses GPT-4o Vision capabilities for best results.
    """
    logger.info(f"Extracting text from {filename}...")
    
    file_ext = filename.split('.')[-1].lower()
    
    if file_ext == 'txt':
        return file_content.decode('utf-8')
        
    elif file_ext in ['jpg', 'jpeg', 'png']:
        return extract_text_from_image(file_content)
        
    elif file_ext == 'pdf':
        try:
            # First attempt: Try extracting text directly (for digital PDFs)
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(file_content))
            text_parts = []
            for page in reader.pages:
                text_parts.append(page.extract_text() or "")
            
            digital_text = "\n".join(text_parts).strip()
            
            if len(digital_text) > 50:
                logger.info(f"Successfully extracted {len(digital_text)} chars from digital PDF.")
                return digital_text
            
            logger.info("Digital PDF extraction yielded little text. Attempting OCR...")
            
            # Second attempt: Convert to images and use VLM (for scanned documents)
            from pdf2image import convert_from_bytes
            images = convert_from_bytes(file_content)
            
            full_text = []
            for i, img in enumerate(images):
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG')
                img_bytes = img_byte_arr.getvalue()
                
                logger.info(f"OCRing page {i+1}...")
                text = extract_text_from_image(img_bytes)
                full_text.append(f"--- Page {i+1} ---\n{text}")
                
            combined_text = "\n".join(full_text)
            logger.info(f"OCR extracted {len(combined_text)} chars.")
            
            if not combined_text.strip():
                return "Error: OCR succeeded but returned no text."
            return combined_text
            
        except Exception as e:
            logger.error(f"Error parsing PDF: {e}")
            return f"Error extracting text from PDF: {str(e)}"
            
    else:
        return "Unsupported file format."

def extract_text_from_image(image_bytes: bytes) -> str:
    """
    Uses a vision model via OpenRouter to extract text from an image.
    Tries several models including free options and confirmed working fallbacks.
    """
    # Using slugs without :free often works better as OpenRouter routes to free ones if available
    models = [
        "meta-llama/llama-3.2-11b-vision-instruct",
        "google/gemini-2.0-flash-001",
        "qwen/qwen-2-vl-7b-instruct",
        "google/gemini-flash-1.5",
        "openai/gpt-4o-mini" # Last resort confirmed working for this user
    ]
    
    base64_image = encode_image(image_bytes)
    
    for model_name in models:
        try:
            logger.info(f"Starting Image OCR via {model_name}...")
            llm = ChatOpenAI(model=model_name, temperature=0, max_retries=1)
            
            message = HumanMessage(
                content=[
                    {"type": "text", "text": "Transcribe the text from this image exactly as it appears. Focus on technical skills, experience, projects, and contact info. Do not include any other commentary."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            )
            
            response = llm.invoke([message])
            extracted_text = response.content.strip()
            
            if extracted_text and len(extracted_text) > 10:
                logger.info(f"OCR complete using {model_name}. Extracted {len(extracted_text)} characters.")
                return extracted_text
                
        except Exception as e:
            logger.warning(f"Failed to use model {model_name}: {e}")
            continue
            
    logger.error("All OCR models failed to extract text from image.")
    return ""  # Return empty string on failure instead of error message
