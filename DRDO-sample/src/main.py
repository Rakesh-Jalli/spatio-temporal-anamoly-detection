"""
Main entry point for the Spatio-Temporal Anomaly Detection System.
"""

from detection.yolo_obb import YOLOOBBDetection
from tracking.bytetrack import ByteTrackTracker
from semantic.clip_embeddings import CLIPEmbeddings
from database.chromadb_store import ChromaDBStore
from anomaly.scoring import AnomalyScorer

def main():
    print("Spatio-Temporal Anomaly Detection System Initialized.")

    # Initialize components
    detector = YOLOOBBDetection()
    tracker = ByteTrackTracker()
    embedder = CLIPEmbeddings()
    db = ChromaDBStore()
    scorer = AnomalyScorer()

    # Placeholder for processing images
    print("Components loaded successfully.")

if __name__ == "__main__":
    main()