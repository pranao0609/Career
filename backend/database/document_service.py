import logging
import uuid
import os
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from fastapi import UploadFile, HTTPException
import aiofiles
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# Google Cloud Storage imports (you'll need to install: pip install google-cloud-storage)
from google.cloud import storage
from google.cloud.exceptions import GoogleCloudError

# Firebase imports
from .firestore import firestore_db
from models.document import (
    DocumentUploadInfo, 
    DocumentFile, 
    FileMetadata, 
    SocialLinks,
    DocumentType, 
    DocumentStatus,
    FileUploadValidation,
    DocumentUploadResponse
)

logger = logging.getLogger(__name__)

class DocumentService:
    def __init__(self):
        self.collection_name = "user_documents"
        self.validation_rules = FileUploadValidation()
        
        # GCP Storage configuration
        self.gcp_project_id = os.getenv("GCP_PROJECT_ID", "your-project-id")
        self.gcp_bucket_name = os.getenv("GCP_STORAGE_BUCKET", "your-bucket-name")
        
        # Initialize GCP Storage client
        try:
            self.storage_client = storage.Client(project=self.gcp_project_id)
            self.bucket = self.storage_client.bucket(self.gcp_bucket_name)
            logger.info(f"✅ GCP Storage initialized - Bucket: {self.gcp_bucket_name}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize GCP Storage: {e}")
            self.storage_client = None
            self.bucket = None
    
    def _get_db(self):
        """Get Firestore database instance"""
        return firestore_db.get_db()
    
    def _generate_upload_session_id(self) -> str:
        """Generate unique session ID for upload tracking"""
        return str(uuid.uuid4())
    
    def _get_file_extension(self, filename: str) -> str:
        """Extract file extension from filename"""
        return Path(filename).suffix.lower()
    
    def _validate_file(self, file: UploadFile, document_type: DocumentType) -> Tuple[bool, List[str]]:
        """Validate uploaded file"""
        errors = []
        
        # Check file size (approximate, as we can't get exact size without reading)
        # This is a basic check - in production you might want more sophisticated validation
        
        # Check file type
        if not self.validation_rules.validate_file_type(file.content_type, document_type):
            errors.append(f"Invalid file type: {file.content_type} for {document_type.value}")
        
        # Check file extension
        extension = self._get_file_extension(file.filename)
        if document_type == DocumentType.RESUME and extension not in ['.pdf', '.doc', '.docx']:
            errors.append(f"Invalid file extension for resume: {extension}")
        elif document_type == DocumentType.CERTIFICATE and extension not in ['.pdf', '.jpg', '.jpeg', '.png']:
            errors.append(f"Invalid file extension for certificate: {extension}")
        
        return len(errors) == 0, errors
    
    async def _upload_to_gcp(self, file: UploadFile, file_path: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """Upload file to Google Cloud Storage"""
        if not self.storage_client or not self.bucket:
            logger.error("GCP Storage not properly initialized")
            return False, None, "GCP Storage not available"
        
        try:
            # Read file content
            file_content = await file.read()
            file.file.seek(0)  # Reset file pointer for potential re-use
            
            # Create blob in GCP bucket
            blob = self.bucket.blob(file_path)
            
            # Upload file
            blob.upload_from_string(
                file_content,
                content_type=file.content_type
            )
            
            # Make blob publicly accessible (optional - configure based on your needs)
            # blob.make_public()
            
            # Get public URL (if made public)
            public_url = None  # blob.public_url if made public
            
            logger.info(f"✅ File uploaded to GCP: {file_path}")
            return True, public_url, None
            
        except GoogleCloudError as e:
            logger.error(f"❌ GCP upload failed: {str(e)}")
            return False, None, f"GCP upload error: {str(e)}"
        except Exception as e:
            logger.error(f"❌ Unexpected upload error: {str(e)}")
            return False, None, f"Upload error: {str(e)}"
    
    def _generate_gcp_file_path(self, user_email: str, document_type: DocumentType, filename: str) -> str:
        """Generate GCP storage file path"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        safe_email = user_email.replace("@", "_").replace(".", "_")
        return f"documents/{safe_email}/{document_type.value}/{timestamp}_{filename}"
    
    async def upload_documents(
        self, 
        user_email: str,
        domain: str,
        portfolio_url: Optional[str],
        linkedin_url: Optional[str],
        github_url: Optional[str],
        personal_portfolio_url: Optional[str],
        resume_file: Optional[UploadFile],
        certificate_files: List[UploadFile]
    ) -> DocumentUploadResponse:
        """Upload user documents and save metadata to Firebase"""
        
        upload_session_id = self._generate_upload_session_id()
        logger.info(f"🚀 Starting document upload for {user_email} - Session: {upload_session_id}")
        
        try:
            uploaded_files = []
            total_size = 0
            gcp_urls = []
            
            # Validate and upload resume
            resume_document = None
            if resume_file:
                is_valid, errors = self._validate_file(resume_file, DocumentType.RESUME)
                if not is_valid:
                    raise HTTPException(status_code=400, detail=f"Resume validation failed: {', '.join(errors)}")
                
                # Generate GCP file path
                gcp_file_path = self._generate_gcp_file_path(user_email, DocumentType.RESUME, resume_file.filename)
                
                # Upload to GCP
                upload_success, public_url, error_msg = await self._upload_to_gcp(resume_file, gcp_file_path)
                if not upload_success:
                    raise HTTPException(status_code=500, detail=f"Resume upload failed: {error_msg}")
                
                # Get file size
                file_content = await resume_file.read()
                file_size = len(file_content)
                resume_file.file.seek(0)
                total_size += file_size
                
                # Create file metadata
                file_metadata = FileMetadata(
                    filename=resume_file.filename,
                    file_size=file_size,
                    content_type=resume_file.content_type,
                    file_extension=self._get_file_extension(resume_file.filename),
                    gcp_bucket_name=self.gcp_bucket_name,
                    gcp_file_path=gcp_file_path,
                    gcp_public_url=public_url,
                    is_valid=True,
                    validation_errors=[]
                )
                
                resume_document = DocumentFile(
                    document_type=DocumentType.RESUME,
                    file_metadata=file_metadata,
                    status=DocumentStatus.UPLOADED,
                    document_title="Resume/CV"
                )
                
                uploaded_files.append({
                    "type": "resume",
                    "filename": resume_file.filename,
                    "size": file_size,
                    "gcp_path": gcp_file_path
                })
                
                if public_url:
                    gcp_urls.append(public_url)
                
                logger.info(f"✅ Resume uploaded: {resume_file.filename}")
            
            # Validate and upload certificates
            certificate_documents = []
            if certificate_files:
                for i, cert_file in enumerate(certificate_files):
                    if cert_file.filename:  # Skip empty file uploads
                        is_valid, errors = self._validate_file(cert_file, DocumentType.CERTIFICATE)
                        if not is_valid:
                            logger.warning(f"Certificate {cert_file.filename} validation failed: {errors}")
                            continue
                        
                        # Generate GCP file path
                        gcp_file_path = self._generate_gcp_file_path(user_email, DocumentType.CERTIFICATE, cert_file.filename)
                        
                        # Upload to GCP
                        upload_success, public_url, error_msg = await self._upload_to_gcp(cert_file, gcp_file_path)
                        if not upload_success:
                            logger.error(f"Certificate upload failed: {error_msg}")
                            continue
                        
                        # Get file size
                        file_content = await cert_file.read()
                        file_size = len(file_content)
                        cert_file.file.seek(0)
                        total_size += file_size
                        
                        # Create file metadata
                        file_metadata = FileMetadata(
                            filename=cert_file.filename,
                            file_size=file_size,
                            content_type=cert_file.content_type,
                            file_extension=self._get_file_extension(cert_file.filename),
                            gcp_bucket_name=self.gcp_bucket_name,
                            gcp_file_path=gcp_file_path,
                            gcp_public_url=public_url,
                            is_valid=True,
                            validation_errors=[]
                        )
                        
                        certificate_document = DocumentFile(
                            document_type=DocumentType.CERTIFICATE,
                            file_metadata=file_metadata,
                            status=DocumentStatus.UPLOADED,
                            document_title=f"Certificate {i+1}"
                        )
                        
                        certificate_documents.append(certificate_document)
                        
                        uploaded_files.append({
                            "type": "certificate",
                            "filename": cert_file.filename,
                            "size": file_size,
                            "gcp_path": gcp_file_path
                        })
                        
                        if public_url:
                            gcp_urls.append(public_url)
                        
                        logger.info(f"✅ Certificate uploaded: {cert_file.filename}")
            
            # Create social links object
            social_links = SocialLinks(
                linkedin_url=linkedin_url or None,
                github_url=github_url or None,
                personal_portfolio_url=personal_portfolio_url or None
            )
            
            # Create document upload info
            document_info = DocumentUploadInfo(
                user_email=user_email,
                domain=domain,
                portfolio_url=portfolio_url,
                social_links=social_links,
                resume=resume_document,
                certificates=certificate_documents,
                upload_session_id=upload_session_id,
                total_files_count=len(uploaded_files),
                total_files_size=total_size,
                overall_status=DocumentStatus.COMPLETED,
                completion_percentage=100.0,
                completed_at=datetime.utcnow()
            )
            
            # Save to Firebase
            await self._save_to_firebase(document_info)
            
            return DocumentUploadResponse(
                success=True,
                message=f"Successfully uploaded {len(uploaded_files)} file(s)",
                user_email=user_email,
                upload_session_id=upload_session_id,
                uploaded_files=uploaded_files,
                total_files=len(uploaded_files),
                total_size=total_size,
                gcp_urls=gcp_urls,
                created_at=document_info.created_at,
                status=DocumentStatus.COMPLETED
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Document upload failed for {user_email}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Document upload failed: {str(e)}")
    
    async def _save_to_firebase(self, document_info: DocumentUploadInfo):
        """Save document info to Firebase Firestore"""
        try:
            db = self._get_db()
            doc_ref = db.collection(self.collection_name).document(document_info.user_email)
            
            # Convert to dict for Firebase storage
            document_data = document_info.model_dump(by_alias=True)
            
            # Convert datetime objects to timestamp
            document_data["created_at"] = document_info.created_at
            document_data["updated_at"] = document_info.updated_at
            if document_info.completed_at:
                document_data["completed_at"] = document_info.completed_at
            
            # Save to Firestore
            doc_ref.set(document_data)
            
            logger.info(f"✅ Document info saved to Firebase for {document_info.user_email}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save document info to Firebase: {str(e)}")
            raise Exception(f"Firebase save failed: {str(e)}")
    
    async def get_user_documents(self, user_email: str) -> Optional[DocumentUploadInfo]:
        """Retrieve user document information from Firebase"""
        try:
            db = self._get_db()
            doc_ref = db.collection(self.collection_name).document(user_email)
            doc = doc_ref.get()
            
            if not doc.exists:
                logger.info(f"No documents found for user: {user_email}")
                return None
            
            data = doc.to_dict()
            
            # Convert back to Pydantic model
            document_info = DocumentUploadInfo.model_validate(data)
            
            logger.info(f"✅ Retrieved documents for {user_email}")
            return document_info
            
        except Exception as e:
            logger.error(f"❌ Failed to retrieve documents for {user_email}: {str(e)}")
            raise Exception(f"Failed to retrieve documents: {str(e)}")
    
    async def delete_user_documents(self, user_email: str) -> bool:
        """Delete user documents from both GCP and Firebase"""
        try:
            # Get document info first
            document_info = await self.get_user_documents(user_email)
            if not document_info:
                logger.warning(f"No documents found to delete for {user_email}")
                return False
            
            # Delete from GCP Storage
            if self.storage_client and self.bucket:
                files_to_delete = []
                
                # Collect all file paths
                if document_info.resume:
                    files_to_delete.append(document_info.resume.file_metadata.gcp_file_path)
                
                for cert in document_info.certificates:
                    files_to_delete.append(cert.file_metadata.gcp_file_path)
                
                # Delete files from GCP
                for file_path in files_to_delete:
                    try:
                        blob = self.bucket.blob(file_path)
                        blob.delete()
                        logger.info(f"✅ Deleted from GCP: {file_path}")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to delete from GCP {file_path}: {e}")
            
            # Delete from Firebase
            db = self._get_db()
            doc_ref = db.collection(self.collection_name).document(user_email)
            doc_ref.delete()
            
            logger.info(f"✅ Deleted all documents for {user_email}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete documents for {user_email}: {str(e)}")
            raise Exception(f"Document deletion failed: {str(e)}")
    
    async def update_document_status(self, user_email: str, status: DocumentStatus):
        """Update document processing status"""
        try:
            db = self._get_db()
            doc_ref = db.collection(self.collection_name).document(user_email)
            
            doc_ref.update({
                "overall_status": status.value,
                "updated_at": datetime.utcnow()
            })
            
            logger.info(f"✅ Updated status to {status.value} for {user_email}")
            
        except Exception as e:
            logger.error(f"❌ Failed to update status for {user_email}: {str(e)}")
            raise Exception(f"Status update failed: {str(e)}")

# Global document service instance
document_service = DocumentService()