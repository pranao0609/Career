from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Path, Depends
from typing import List, Optional
import logging
from datetime import datetime

from ..document_service import document_service
from models.document import DocumentUploadResponse, DocumentRetrievalResponse, DocumentStatus

router = APIRouter(prefix="/api/documents")
logger = logging.getLogger(__name__)

@router.post("/upload/{user_email}", response_model=DocumentUploadResponse)
async def upload_documents(
    user_email: str = Path(..., description="User email as identifier"),
    domain: str = Form(..., description="Professional domain/expertise area"),
    portfolio_url: Optional[str] = Form(None, description="Portfolio website URL"),
    linkedin_url: Optional[str] = Form(None, description="LinkedIn profile URL"),
    github_url: Optional[str] = Form(None, description="GitHub profile URL"),
    personal_portfolio_url: Optional[str] = Form(None, description="Personal portfolio URL"),
    resume: Optional[UploadFile] = File(None, description="Resume/CV file"),
    certificates: List[UploadFile] = File(default=[], description="Certificate files")
):
    """
    Upload user documents (resume and certificates) along with profile information.
    Files are stored in GCP Storage and metadata is saved to Firebase.
    """
    try:
        logger.info(f"📁 Starting document upload for user: {user_email}")
        
        # Validate required fields
        if not domain or domain.strip() == "":
            raise HTTPException(status_code=400, detail="Domain field is required")
        
        # Validate that at least resume is provided
        if not resume or not resume.filename:
            raise HTTPException(status_code=400, detail="Resume file is required")
        
        # Filter out empty certificate files
        valid_certificates = [cert for cert in certificates if cert.filename and cert.filename.strip()]
        
        logger.info(f"📊 Upload details - Resume: {resume.filename if resume else 'None'}, Certificates: {len(valid_certificates)}")
        
        # Call document service to handle the upload
        result = await document_service.upload_documents(
            user_email=user_email,
            domain=domain.strip(),
            portfolio_url=portfolio_url.strip() if portfolio_url else None,
            linkedin_url=linkedin_url.strip() if linkedin_url else None,
            github_url=github_url.strip() if github_url else None,
            personal_portfolio_url=personal_portfolio_url.strip() if personal_portfolio_url else None,
            resume_file=resume,
            certificate_files=valid_certificates
        )
        
        logger.info(f"✅ Document upload completed for {user_email}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Document upload failed for {user_email}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Document upload failed: {str(e)}")

@router.get("/user/{user_email}", response_model=DocumentRetrievalResponse)
async def get_user_documents(
    user_email: str = Path(..., description="User email as identifier")
):
    """
    Retrieve all document information for a specific user.
    """
    try:
        logger.info(f"🔍 Retrieving documents for user: {user_email}")
        
        document_info = await document_service.get_user_documents(user_email)
        
        if document_info:
            logger.info(f"✅ Documents found for {user_email}")
            return DocumentRetrievalResponse(
                success=True,
                user_email=user_email,
                document_info=document_info,
                found=True
            )
        else:
            logger.info(f"📭 No documents found for {user_email}")
            return DocumentRetrievalResponse(
                success=True,
                user_email=user_email,
                document_info=None,
                found=False
            )
            
    except Exception as e:
        logger.error(f"❌ Failed to retrieve documents for {user_email}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve documents: {str(e)}")

@router.delete("/user/{user_email}")
async def delete_user_documents(
    user_email: str = Path(..., description="User email as identifier")
):
    """
    Delete all documents for a specific user from both GCP Storage and Firebase.
    """
    try:
        logger.info(f"🗑️ Deleting documents for user: {user_email}")
        
        success = await document_service.delete_user_documents(user_email)
        
        if success:
            return {
                "success": True,
                "message": f"All documents deleted successfully for {user_email}",
                "user_email": user_email,
                "deleted_at": datetime.utcnow().isoformat()
            }
        else:
            return {
                "success": False,
                "message": f"No documents found to delete for {user_email}",
                "user_email": user_email
            }
            
    except Exception as e:
        logger.error(f"❌ Failed to delete documents for {user_email}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Document deletion failed: {str(e)}")

@router.patch("/user/{user_email}/status")
async def update_document_status(
    user_email: str = Path(..., description="User email as identifier"),
    status: DocumentStatus = Form(..., description="New document status")
):
    """
    Update the processing status of documents for a user.
    """
    try:
        logger.info(f"🔄 Updating document status for {user_email} to {status.value}")
        
        await document_service.update_document_status(user_email, status)
        
        return {
            "success": True,
            "message": f"Document status updated to {status.value}",
            "user_email": user_email,
            "new_status": status.value,
            "updated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to update status for {user_email}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Status update failed: {str(e)}")

@router.get("/health")
async def document_service_health():
    """
    Check the health of the document service and its dependencies.
    """
    try:
        health_status = {
            "service": "document_service",
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "dependencies": {}
        }
        
        # Check GCP Storage connectivity
        if document_service.storage_client and document_service.bucket:
            try:
                # Try to check if bucket exists
                document_service.bucket.reload()
                health_status["dependencies"]["gcp_storage"] = {
                    "status": "connected",
                    "bucket": document_service.gcp_bucket_name
                }
            except Exception as e:
                health_status["dependencies"]["gcp_storage"] = {
                    "status": "error",
                    "error": str(e)
                }
                health_status["status"] = "degraded"
        else:
            health_status["dependencies"]["gcp_storage"] = {
                "status": "not_configured"
            }
            health_status["status"] = "degraded"
        
        # Check Firebase connectivity
        try:
            db = document_service._get_db()
            # Try a simple operation to test connectivity
            test_collection = db.collection("health_check").limit(1).get()
            health_status["dependencies"]["firebase"] = {
                "status": "connected"
            }
        except Exception as e:
            health_status["dependencies"]["firebase"] = {
                "status": "error",
                "error": str(e)
            }
            health_status["status"] = "degraded"
        
        return health_status
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {str(e)}")
        return {
            "service": "document_service",
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

# Optional: Webhook endpoint for processing completion notifications
@router.post("/webhook/processing-complete")
async def processing_complete_webhook(
    user_email: str = Form(...),
    session_id: str = Form(...),
    status: DocumentStatus = Form(...)
):
    """
    Webhook endpoint for external processing services to notify completion.
    """
    try:
        logger.info(f"📨 Processing complete webhook for {user_email}, session: {session_id}")
        
        await document_service.update_document_status(user_email, status)
        
        return {
            "success": True,
            "message": "Status updated successfully",
            "user_email": user_email,
            "session_id": session_id,
            "status": status.value
        }
        
    except Exception as e:
        logger.error(f"❌ Webhook processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Webhook processing failed: {str(e)}")

# Statistics endpoint (optional)
@router.get("/stats")
async def get_document_statistics():
    """
    Get basic statistics about document uploads.
    """
    try:
        # This is a basic implementation - you might want to add more sophisticated analytics
        return {
            "message": "Document statistics endpoint",
            "note": "Implementation depends on your analytics requirements",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Stats retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")
    

# Add these imports at the top of your documents_routes.py
from fastapi.responses import StreamingResponse
import io

@router.get("/download/{user_email}/{file_type}/{file_index}")
async def download_file(
    user_email: str = Path(..., description="User email"),
    file_type: str = Path(..., description="'resume' or 'certificate'"),
    file_index: int = Path(..., description="Certificate index (0 for resume, 0-n for certificates)")
):
    """Download file from GCP Storage"""
    try:
        # Get document info
        document_info = await document_service.get_user_documents(user_email)
        if not document_info:
            raise HTTPException(status_code=404, detail="User documents not found")
        
        target_file = None
        
        if file_type == "resume" and document_info.resume:
            target_file = document_info.resume
        elif file_type == "certificate" and file_index < len(document_info.certificates):
            target_file = document_info.certificates[file_index]
        
        if not target_file:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Download from GCP
        if document_service.storage_client and document_service.bucket:
            blob = document_service.bucket.blob(target_file.file_metadata.gcp_file_path)
            
            if not blob.exists():
                raise HTTPException(status_code=404, detail="File not found in storage")
            
            file_content = blob.download_as_bytes()
            
            return StreamingResponse(
                io.BytesIO(file_content),
                media_type=target_file.file_metadata.content_type,
                headers={
                    "Content-Disposition": f"attachment; filename=\"{target_file.file_metadata.filename}\""
                }
            )
        else:
            raise HTTPException(status_code=500, detail="Storage not available")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File download failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

@router.get("/view/{user_email}/{file_type}/{file_index}")
async def view_file(
    user_email: str = Path(..., description="User email"),
    file_type: str = Path(..., description="'resume' or 'certificate'"),
    file_index: int = Path(..., description="File index")
):
    """View file in browser (inline)"""
    try:
        # Get document info
        document_info = await document_service.get_user_documents(user_email)
        if not document_info:
            raise HTTPException(status_code=404, detail="User documents not found")
        
        target_file = None
        
        if file_type == "resume" and document_info.resume:
            target_file = document_info.resume
        elif file_type == "certificate" and file_index < len(document_info.certificates):
            target_file = document_info.certificates[file_index]
        
        if not target_file:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Download from GCP
        if document_service.storage_client and document_service.bucket:
            blob = document_service.bucket.blob(target_file.file_metadata.gcp_file_path)
            file_content = blob.download_as_bytes()
            
            return StreamingResponse(
                io.BytesIO(file_content),
                media_type=target_file.file_metadata.content_type,
                headers={
                    "Content-Disposition": f"inline; filename=\"{target_file.file_metadata.filename}\""
                }
            )
        else:
            raise HTTPException(status_code=500, detail="Storage not available")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File view failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"View failed: {str(e)}")

@router.delete("/file/{user_email}/{file_type}/{file_index}")
async def delete_specific_file(
    user_email: str = Path(..., description="User email"),
    file_type: str = Path(..., description="'resume' or 'certificate'"),
    file_index: int = Path(..., description="File index")
):
    """Delete a specific file"""
    try:
        # Get document info
        document_info = await document_service.get_user_documents(user_email)
        if not document_info:
            raise HTTPException(status_code=404, detail="User documents not found")
        
        target_file = None
        file_path_to_delete = None
        
        if file_type == "resume" and document_info.resume:
            target_file = document_info.resume
            file_path_to_delete = target_file.file_metadata.gcp_file_path
            # Remove resume from document info
            document_info.resume = None
        elif file_type == "certificate" and file_index < len(document_info.certificates):
            target_file = document_info.certificates[file_index]
            file_path_to_delete = target_file.file_metadata.gcp_file_path
            # Remove certificate from list
            document_info.certificates.pop(file_index)
        
        if not target_file:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Delete from GCP Storage
        if document_service.storage_client and document_service.bucket:
            blob = document_service.bucket.blob(file_path_to_delete)
            if blob.exists():
                blob.delete()
                logger.info(f"Deleted from GCP: {file_path_to_delete}")
        
        # Update document info in Firebase
        document_info.total_files_count -= 1
        document_info.total_files_size -= target_file.file_metadata.file_size
        document_info.updated_at = datetime.utcnow()
        
        await document_service._save_to_firebase(document_info)
        
        return {
            "success": True,
            "message": f"File deleted successfully",
            "deleted_file": target_file.file_metadata.filename,
            "user_email": user_email,
            "remaining_files": document_info.total_files_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File deletion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")