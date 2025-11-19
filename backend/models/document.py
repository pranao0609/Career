from pydantic import BaseModel, EmailStr, Field, HttpUrl
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class DocumentType(str, Enum):
    RESUME = "resume"
    CERTIFICATE = "certificate"
    PORTFOLIO = "portfolio"
    OTHER = "other"

class DocumentStatus(str, Enum):
    PENDING = "pending"
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class FileMetadata(BaseModel):
    """Metadata for uploaded files"""
    filename: str = Field(..., description="Original filename")
    file_size: int = Field(..., description="File size in bytes")
    content_type: str = Field(..., description="MIME type of the file")
    file_extension: str = Field(..., description="File extension")
    upload_timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # GCP Storage info
    gcp_bucket_name: str = Field(..., description="GCP Storage bucket name")
    gcp_file_path: str = Field(..., description="Path in GCP bucket")
    gcp_public_url: Optional[str] = Field(None, description="Public URL if applicable")
    
    # File validation
    is_valid: bool = Field(default=True, description="File validation status")
    validation_errors: List[str] = Field(default_factory=list, description="Validation error messages")

class DocumentFile(BaseModel):
    """Individual document file information"""
    document_type: DocumentType = Field(..., description="Type of document")
    file_metadata: FileMetadata = Field(..., description="File metadata")
    status: DocumentStatus = Field(default=DocumentStatus.PENDING, description="Processing status")
    
    # Additional fields for specific document types
    document_title: Optional[str] = Field(None, description="User-provided title for the document")
    document_description: Optional[str] = Field(None, description="User-provided description")
    tags: List[str] = Field(default_factory=list, description="User-defined tags")

class SocialLinks(BaseModel):
    """Social media and professional links"""
    linkedin_url: Optional[str] = Field(None, description="LinkedIn profile URL")
    github_url: Optional[str] = Field(None, description="GitHub profile URL")
    personal_portfolio_url: Optional[str] = Field(None, description="Personal portfolio website")
    
    # Additional professional links
    behance_url: Optional[str] = Field(None, description="Behance portfolio URL")
    dribbble_url: Optional[str] = Field(None, description="Dribbble portfolio URL")
    medium_url: Optional[str] = Field(None, description="Medium blog URL")

class DocumentUploadInfo(BaseModel):
    """Complete document upload information for a user"""
    user_email: EmailStr = Field(..., description="User's email address")
    
    # Professional information
    domain: str = Field(..., min_length=1, max_length=100, description="Professional domain/expertise area")
    portfolio_url: Optional[str] = Field(None, description="Primary portfolio website URL")
    
    # Social links
    social_links: SocialLinks = Field(default_factory=SocialLinks, description="Social media links")
    
    # Uploaded documents
    resume: Optional[DocumentFile] = Field(None, description="Resume/CV file")
    certificates: List[DocumentFile] = Field(default_factory=list, description="Certificate files")
    
    # Metadata
    upload_session_id: str = Field(..., description="Unique session identifier for this upload")
    total_files_count: int = Field(default=0, description="Total number of files uploaded")
    total_files_size: int = Field(default=0, description="Total size of all files in bytes")
    
    # Status tracking
    overall_status: DocumentStatus = Field(default=DocumentStatus.PENDING, description="Overall upload status")
    completion_percentage: float = Field(default=0.0, ge=0.0, le=100.0, description="Upload completion percentage")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(None, description="When upload was completed")

class DocumentUploadRequest(BaseModel):
    """Request model for document upload"""
    user_email: EmailStr
    domain: str = Field(..., min_length=1, max_length=100)
    portfolio_url: Optional[str] = None
    linkedin_url: Optional[str] = None
    github_url: Optional[str] = None
    personal_portfolio_url: Optional[str] = None

class DocumentUploadResponse(BaseModel):
    """Response model for document upload"""
    success: bool
    message: str
    user_email: str
    upload_session_id: str
    uploaded_files: List[Dict[str, Any]]
    total_files: int
    total_size: int
    gcp_urls: List[str] = Field(default_factory=list, description="GCP Storage URLs")
    created_at: datetime
    status: DocumentStatus

class DocumentRetrievalResponse(BaseModel):
    """Response model for retrieving user documents"""
    success: bool
    user_email: str
    document_info: Optional[DocumentUploadInfo]
    found: bool

# Utility models for validation
class FileUploadValidation(BaseModel):
    """File upload validation rules"""
    max_file_size_mb: int = 10  # 10MB per file
    allowed_resume_types: List[str] = ["application/pdf", "application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
    allowed_certificate_types: List[str] = ["application/pdf", "image/jpeg", "image/png", "image/jpg"]
    max_certificates_count: int = 10
    
    def validate_file_size(self, file_size: int) -> bool:
        return file_size <= (self.max_file_size_mb * 1024 * 1024)
    
    def validate_file_type(self, content_type: str, document_type: DocumentType) -> bool:
        if document_type == DocumentType.RESUME:
            return content_type in self.allowed_resume_types
        elif document_type == DocumentType.CERTIFICATE:
            return content_type in self.allowed_certificate_types
        return False

# Error models
class DocumentUploadError(BaseModel):
    """Error model for document upload failures"""
    error_code: str
    error_message: str
    failed_files: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)