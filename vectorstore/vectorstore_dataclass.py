from dataclasses import dataclass

@dataclass
class VectorStoreResponse:
    total_received_chunks: int
    total_stored_chunks: int
    total_skipped_chunks: int
    total_failed_chunks: int