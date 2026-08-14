import hashlib

def generate_chunk_id(source: str, page: int, chunk_number: int,content: str) -> str:
        normalized_content=content.strip()
        identity = (
                f"{source}|"
                f"{page}|"
                f"{chunk_number}|"
                f"{normalized_content}"
                  )
        return hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()


#chunk_id = generate_chunk_id("Hello World")
#print(f"Chunk ID for 'Hello World': {chunk_id}")

test_source = "source-A"
test_page = 1
test_chunk_number = 1
test_content = "ABC"

original_id = generate_chunk_id(
    test_source,
    test_page,
    test_chunk_number,
    test_content
)

# Change source
different_source_id = generate_chunk_id(
    "source-B",
    test_page,
    test_chunk_number,
    test_content
)
print(f"Original ID: {original_id}")
print(f"Different Source ID: {different_source_id}")
assert original_id != different_source_id

# Change page
different_page_id = generate_chunk_id(
    test_source,
    2,
    test_chunk_number,
    test_content
)

assert original_id != different_page_id
print(f"Original ID: {original_id}")
print(f"Different Page ID: {different_page_id}")

# Change chunk number
different_chunk_number_id = generate_chunk_id(
    test_source,
    test_page,
    2,
    test_content
)

assert original_id != different_chunk_number_id
print(f"Original ID: {original_id}")
print(f"Different Chunk Number ID: {different_chunk_number_id}")
# Change content
different_content_id = generate_chunk_id(
    test_source,
    test_page,
    test_chunk_number,
    "XYZ"
)

assert original_id != different_content_id
print(f"Original ID: {original_id}")
print(f"Different Content ID: {different_content_id}")