from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import numpy as np
from nltk.tokenize import sent_tokenize
import logging

import nltk
nltk.download('punkt')

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

model_name = 'intfloat/e5-small-v2'
model = SentenceTransformer(model_name)
# Initialize the Hugging Face tokenizer using the same model name
tokenizer = AutoTokenizer.from_pretrained(model_name)

@app.route('/health', methods=['GET'])
def health():
   return jsonify({"status": "ok"}), 200

@app.route('/embed', methods=['POST'])
def embed_text():
    data = request.json
    text = data.get('text', None)
    prefix = data.get('prefix', None)

    if text is None:
        return jsonify({"error": "No text provided"}), 400
    if prefix is None:
        return jsonify({"error": "No prefix provided"}), 400

    try:
        # Tokenize the text into sentences
        sentences = sent_tokenize(text)

        # Combine the sentences back into chunks of approximately 500 tokens each
        chunks = []
        current_chunk = ""
        current_length = 0
        num_tokens = 0
        invalid_tokens = 0

        for sentence in sentences:
            # Tokenize the current sentence
            sentence_ids = tokenizer.encode(sentence, add_special_tokens=False)
            sentence_length = len(sentence_ids)
            num_tokens = num_tokens + sentence_length

            if sentence_length > 500:
                logging.info(f"Skipping too long of a sentence: {sentence}")
                invalid_tokens = invalid_tokens + sentence_length
                continue

            # Check if adding this sentence exceeds the chunk size
            if current_length + sentence_length > 500:
                chunks.append(current_chunk.strip())
                current_chunk = ""
                current_length = 0

            current_chunk += sentence + " "
            current_length += sentence_length

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        # Process chunks as before
        # ...

        # Loop over the token IDs and break them into CHUNK_SIZE chunks
        chunk_embeddings = []
        for chunk in chunks:
            # Decode chunk back to text
            chunk_text = prefix + chunk
            # logging.info(f"Chunk text: {chunk_text}")

            # Generate embedding for the chunk
            chunk_embedding = model.encode([chunk_text], normalize_embeddings=True)[0]
            chunk_embeddings.append(chunk_embedding)

        # Average the embeddings to get a single document embedding
        if len(chunk_embeddings) == 0:
            logging.warning("No valid chunks found for embedding.")
            embedding = None
        else:
            embedding = np.mean(chunk_embeddings, axis=0).tolist()
        logging.info(f"Generated embedding for {num_tokens} tokens."
                     f" Found {invalid_tokens} invalid tokens.")
        return jsonify({"embedding": embedding, "num_tokens": num_tokens,
                        "invalid_tokens": invalid_tokens})

    except Exception as e:
        # Log the exception for debugging
        logging.error(f"An error occurred: {e}")
        return jsonify({"error": f"An error occurred during encoding: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
