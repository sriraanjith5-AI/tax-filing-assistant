import hashlib

def generate_chunk_id(content: str) -> str:
    normalized_content = content.strip()

    return hashlib.sha256(
        normalized_content.encode("utf-8")
    ).hexdigest()


chunk_id = generate_chunk_id("Hello World")
print(f"Chunk ID for 'Hello World': {chunk_id}")