import os
from google.cloud import storage
from dotenv import load_dotenv    
load_dotenv()

def test_gcp():
    try:
        project_id = os.getenv("GCP_PROJECT_ID")
        bucket_name = os.getenv("GCP_STORAGE_BUCKET")
        print(project_id)
        print(bucket_name)
        client = storage.Client(project=project_id)
        bucket = client.bucket(bucket_name)
        
        print(f"Project ID: {project_id}")
        print(f"Bucket Name: {bucket_name}")
        print(f"Bucket exists: {bucket.exists()}")
        
        # Test upload
        blob = bucket.blob("test-file.txt")
        blob.upload_from_string("Hello GCP!")
        
        print("✅ GCP Storage is working!")
        
        # Clean up test file
        blob.delete()
        
    except Exception as e:
        print(f"❌ GCP Error: {e}")
        print("Make sure your environment variables are set correctly")

if __name__ == "__main__":
    test_gcp()