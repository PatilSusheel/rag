import os
import uuid
import traceback
import tempfile
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import login

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import boto3
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.text import partition_text
from unstructured.partition.pptx import partition_pptx
from unstructured.partition.csv import partition_csv
from unstructured.partition.docx import partition_docx
import redis
import hashlib
import jwt
from passlib.context import CryptContext
import logging
from transformers import pipeline

# SQLAlchemy imports
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, Boolean, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import func
import psycopg2
from sqlalchemy.exc import SQLAlchemyError

# Load environment variables
load_dotenv()
login(token=os.getenv("HF_TOKEN"))
pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-1B-Instruct")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database Configuration - Fixed to use correct environment variable
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    logger.error("DATABASE_URL environment variable is not set")
    raise ValueError("DATABASE_URL environment variable is required")

logger.info(f"Connecting to database: {DATABASE_URL.split('@')[0]}@***")

try:
    engine = create_engine(
        DATABASE_URL,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        pool_recycle=300,
        echo=False  # Set to True for SQL debugging
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()
    logger.info("Database engine created successfully")
except Exception as e:
    logger.error(f"Failed to create database engine: {e}")
    raise

# Database Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    __table_args__ = (
        Index('idx_users_username_lower', func.lower(username)),
        Index('idx_users_email_lower', func.lower(email)),
    )

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    user_id = Column(UUID(as_uuid=False), nullable=False)
    file_type = Column(String(10), nullable=False)
    file_size = Column(Integer, nullable=False)
    s3_url = Column(String(500), nullable=True)
    upload_time = Column(DateTime, default=datetime.utcnow)
    content_length = Column(Integer, nullable=False)
    element_count = Column(Integer, default=0)
    original_filename = Column(String(255), nullable=True)
    
    __table_args__ = (
        Index('idx_documents_user_id', user_id),
        Index('idx_documents_upload_time', upload_time.desc()),
        Index('idx_documents_file_type', file_type),
    )

# Create tables with error handling
try:
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")
except Exception as e:
    logger.error(f"Failed to create database tables: {e}")
    raise

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()

# Pydantic Models
class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    password: str = Field(..., min_length=6)

class UserLogin(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    document_ids: Optional[List[str]] = None

class DocumentMetadata(BaseModel):
    title: str
    file_type: str
    file_size: int
    upload_time: str
    user_id: str
    content_preview: str

# Initialize FastAPI
app = FastAPI(
    title="Document Query RAG System",
    description="A full-stack RAG application for document querying",
    version="1.0.0"
)

# CORS Configuration - Fixed to include more common ports
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://localhost:3001", 
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Security Configuration - Using environment variables
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not JWT_SECRET_KEY:
    logger.error("JWT_SECRET_KEY environment variable is not set")
    raise ValueError("JWT_SECRET_KEY environment variable is required")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

# File Configuration
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB default
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")

# Create upload directory
try:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    logger.info(f"Upload directory created/verified: {UPLOAD_DIR}")
except Exception as e:
    logger.error(f"Failed to create upload directory: {e}")
    raise

# Initialize Redis - Using environment variables
redis_client = None
try:
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    redis_db = int(os.getenv("REDIS_DB", "0"))
    
    redis_client = redis.Redis(
        host=redis_host, 
        port=redis_port, 
        db=redis_db, 
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5
    )
    redis_client.ping()
    logger.info(f"Redis connected successfully at {redis_host}:{redis_port}")
except Exception as e:
    logger.warning(f"Redis not available: {e}, using database storage only")
    redis_client = None

# Initialize AWS S3 - Using environment variables
s3_client = None
aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_region = os.getenv('AWS_DEFAULT_REGION')
S3_BUCKET_NAME = os.getenv('AWS_S3_BUCKET')

if all([aws_access_key, aws_secret_key, aws_region, S3_BUCKET_NAME]):
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region
        )
        # Test S3 connection
        s3_client.head_bucket(Bucket=S3_BUCKET_NAME)
        logger.info(f"AWS S3 client initialized successfully for bucket: {S3_BUCKET_NAME}")
    except Exception as e:
        logger.warning(f"Failed to initialize S3 client: {e}")
        s3_client = None
else:
    logger.warning("S3 configuration incomplete, file storage will be local only")

# Helper Functions
def hash_password(password: str) -> str:
    """Hash password using bcrypt"""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Dict:
    """Verify and decode JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except (jwt.PyJWTError, jwt.InvalidTokenError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security), 
    db: Session = Depends(get_db)
) -> User:
    """Get current authenticated user"""
    token = credentials.credentials
    payload = verify_token(token)
    user_id = payload.get("user_id")
    
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload"
        )
    
    user = db.query(User).filter(User.id == user_id, User.is_active == True).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )
    
    return user

def validate_file_size(file_content: bytes) -> None:
    """Validate file size against maximum allowed"""
    if len(file_content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size allowed: {MAX_FILE_SIZE / 1024 / 1024:.1f}MB"
        )

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent security issues"""
    if not filename:
        return "unnamed_file"
    
    # Remove path separators and keep only safe characters
    safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    sanitized = "".join(c for c in filename if c in safe_chars or c.isspace())
    sanitized = sanitized.strip()
    
    # Ensure filename is not empty and not too long
    if not sanitized:
        sanitized = "unnamed_file"
    if len(sanitized) > 100:
        sanitized = sanitized[:100]
    
    return sanitized

def parse_document(file_path: str, file_type: str) -> List:
    """Parse document based on file type"""
    try:
        file_type_lower = file_type.lower()
        
        if file_type_lower == 'pdf':
            elements = partition_pdf(file_path)
        elif file_type_lower == 'txt':
            elements = partition_text(file_path)
        elif file_type_lower in ['ppt', 'pptx']:
            elements = partition_pptx(file_path)
        elif file_type_lower == 'csv':
            elements = partition_csv(file_path)
        elif file_type_lower in ['doc', 'docx']:
            elements = partition_docx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        if not elements:
            raise ValueError(f"No content could be extracted from the {file_type} file")
        
        return elements
        
    except Exception as e:
        logger.error(f"Document parsing failed for {file_type}: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to parse {file_type} document: {str(e)}"
        )

def generate_response_with_llama(messages: list) -> str:
    """Simplified version of LLM generation"""
    try:
        prompt = messages[0]['content']
        
        # Simple generation without special formatting
        response = pipe(
            prompt,
            max_new_tokens=300,      # Reduced from 300
            temperature=0.3,         # Lower temperature for faster, more focused responses
            do_sample=False,         # Disable sampling for speed
            pad_token_id=pipe.tokenizer.eos_token_id,
            eos_token_id=pipe.tokenizer.eos_token_id,
            return_full_text=False,
            # Add these for speed optimization
            use_cache=True,
            clean_up_tokenization_spaces=True
        )
        
        generated_text = response[0]['generated_text']
        
        # Remove the original prompt from the response
        if prompt in generated_text:
            generated_text = generated_text.replace(prompt, "").strip()
        
        return generated_text if generated_text else "I apologize, but I couldn't generate a response based on the provided documents."

    except Exception as e:
        logger.error(f"Error in simple LLM generation: {e}")
        return "I apologize, but I encountered an error while processing your request."


def advanced_text_search(
    query: str, 
    db: Session, 
    user_id: str, 
    document_ids: Optional[List[str]] = None
) -> List[Dict]:
    """Advanced search with scoring algorithm"""
    if not query or not query.strip():
        return []
    
    query_lower = query.lower().strip()
    query_terms = [term.strip() for term in query_lower.split() if len(term.strip()) > 2]
    
    if not query_terms:
        return []
    
    # Build database query
    db_query = db.query(Document).filter(Document.user_id == user_id)
    
    if document_ids:
        db_query = db_query.filter(Document.id.in_(document_ids))
    
    documents = db_query.all()
    
    if not documents:
        return []
    
    results = []
    
    for doc in documents:
        if not doc.content:
            continue
            
        content_lower = doc.content.lower()
        title_lower = doc.title.lower()
        
        # Advanced scoring algorithm
        score = 0
        
        # Exact phrase matching (highest score)
        if query_lower in content_lower:
            score += 15
        if query_lower in title_lower:
            score += 20
        
        # Individual term matching with position weighting
        for term in query_terms:
            content_matches = content_lower.count(term)
            title_matches = title_lower.count(term)
            
            score += content_matches * 3
            score += title_matches * 8
            
            # Bonus for terms appearing early in content
            term_position = content_lower.find(term)
            if term_position != -1 and term_position < 500:
                score += 2
        
        # Boost score for document recency
        if doc.upload_time:
            days_old = (datetime.utcnow() - doc.upload_time).days
            if days_old < 7:
                score *= 1.3  # 30% boost for very recent documents
            elif days_old < 30:
                score *= 1.1  # 10% boost for recent documents
        
        # Boost shorter documents (often more focused)
        if doc.content_length < 5000:
            score *= 1.1
        
        if score > 0:
            results.append({
                'score': score,
                'document': doc,
                'id': doc.id
            })
    
    # Sort by score (highest first)
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:10]  # Return top 10 results

def extract_relevant_snippet(content: str, query: str, max_length: int = 800) -> str:
    """Extract most relevant snippet from content"""
    if not content or not query:
        return content[:max_length] if content else ""
    
    content = content.strip()
    if len(content) <= max_length:
        return content
    
    query_terms = [term.lower().strip() for term in query.split() if len(term.strip()) > 2]
    
    if not query_terms:
        return content[:max_length]
    
    content_lower = content.lower()
    best_position = 0
    best_score = 0
    
    # Find the position with the most query term matches
    window_size = max_length // 2
    step_size = max(50, window_size // 10)
    
    for i in range(0, max(1, len(content) - window_size), step_size):
        window_end = min(len(content), i + window_size)
        window = content_lower[i:window_end]
        
        score = sum(window.count(term) for term in query_terms)
        
        if score > best_score:
            best_score = score
            best_position = i
    
    # Extract snippet around best position
    start = max(0, best_position - window_size // 4)
    end = min(len(content), start + max_length)
    
    # Adjust to word boundaries
    if start > 0:
        # Find the next space to start cleanly
        space_pos = content.find(' ', start)
        if space_pos != -1 and space_pos < start + 50:
            start = space_pos + 1
    
    if end < len(content):
        # Find the previous space to end cleanly
        space_pos = content.rfind(' ', end - 50, end)
        if space_pos != -1:
            end = space_pos
    
    snippet = content[start:end].strip()
    
    # Add ellipsis if needed
    if start > 0:
        snippet = "..." + snippet
    if end < len(content):
        snippet = snippet + "..."
    
    return snippet

# API Endpoints

@app.post("/auth/register")
async def register_user(user: UserCreate, db: Session = Depends(get_db)):
    """User registration endpoint with enhanced validation"""
    try:
        # Validate input
        if not user.username or not user.email or not user.password:
            raise HTTPException(status_code=400, detail="All fields are required")
        
        # Check if user already exists
        existing_user = db.query(User).filter(
            (func.lower(User.username) == user.username.lower()) |
            (func.lower(User.email) == user.email.lower())
        ).first()
        
        if existing_user:
            if existing_user.username.lower() == user.username.lower():
                raise HTTPException(status_code=400, detail="Username already registered")
            else:
                raise HTTPException(status_code=400, detail="Email already registered")
        
        # Create new user
        hashed_password = hash_password(user.password)
        
        db_user = User(
            username=user.username.strip(),
            email=user.email.strip().lower(),
            password_hash=hashed_password
        )
        
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        
        # Create access token
        access_token = create_access_token({
            "user_id": db_user.id, 
            "username": db_user.username
        })
        
        logger.info(f"New user registered: {user.username}")
        
        return {
            "message": "User registered successfully",
            "access_token": access_token,
            "token_type": "bearer",
            "user_id": db_user.id,
            "username": db_user.username
        }
        
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")

@app.post("/auth/login")
async def login_user(user: UserLogin, db: Session = Depends(get_db)):
    """User login endpoint with enhanced validation"""
    try:
        if not user.username or not user.password:
            raise HTTPException(status_code=400, detail="Username and password are required")
        
        # Find user (case-insensitive username)
        db_user = db.query(User).filter(
            func.lower(User.username) == user.username.lower(),
            User.is_active == True
        ).first()
        
        if not db_user or not verify_password(user.password, db_user.password_hash):
            raise HTTPException(
                status_code=401,
                detail="Invalid username or password"
            )
        
        # Create access token
        access_token = create_access_token({
            "user_id": db_user.id,
            "username": db_user.username
        })
        
        logger.info(f"User logged in: {user.username}")
        
        return {
            "message": "Login successful",
            "access_token": access_token,
            "token_type": "bearer",
            "user_id": db_user.id,
            "username": db_user.username
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Enhanced document upload with comprehensive error handling"""
    temp_file_path = None
    
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Validate file type
        allowed_extensions = {'.pdf', '.txt', '.pptx', '.ppt', '.csv', '.docx', '.doc'}
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{file_extension}'. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Read and validate file content
        file_content = await file.read()
        
        if not file_content:
            raise HTTPException(status_code=400, detail="Empty file provided")
        
        validate_file_size(file_content)
        
        file_size = len(file_content)
        safe_filename = sanitize_filename(file.filename)
        
        # Upload to S3 if configured
        s3_url = None
        if s3_client and S3_BUCKET_NAME:
            try:
                s3_key = f"documents/{current_user.id}/{uuid.uuid4()}/{safe_filename}"
                s3_client.put_object(
                    Bucket=S3_BUCKET_NAME, 
                    Key=s3_key, 
                    Body=file_content,
                    ContentType=file.content_type or 'application/octet-stream'
                )
                s3_url = f"s3://{S3_BUCKET_NAME}/{s3_key}"
                logger.info(f"File uploaded to S3: {s3_key}")
            except Exception as e:
                logger.warning(f"S3 upload failed: {e}")
        
        # Create temporary file for parsing
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        # Parse document
        elements = parse_document(temp_file_path, file_extension[1:])  # Remove dot
        document_text = "\n".join([str(element) for element in elements if str(element).strip()])
        
        if not document_text.strip():
            raise HTTPException(status_code=400, detail="No text content found in document")
        
        # Store document in database
        db_document = Document(
            title=safe_filename,
            content=document_text,
            user_id=current_user.id,
            file_type=file_extension[1:],  # Remove dot
            file_size=file_size,
            s3_url=s3_url,
            content_length=len(document_text),
            element_count=len(elements),
            original_filename=file.filename
        )
        
        db.add(db_document)
        db.commit()
        db.refresh(db_document)
        
        # Store in Redis if available
        if redis_client:
            try:
                redis_client.hset(f"document:{db_document.id}", mapping={
                    "title": safe_filename,
                    "user_id": current_user.id,
                    "upload_time": db_document.upload_time.isoformat(),
                    "file_type": file_extension[1:]
                })
                redis_client.expire(f"document:{db_document.id}", 86400 * 30)  # 30 days TTL
            except Exception as e:
                logger.warning(f"Redis storage failed: {e}")
        
        logger.info(f"Document uploaded successfully: {safe_filename} by {current_user.username}")
        
        return {
            "message": "Document uploaded and processed successfully",
            "document_id": db_document.id,
            "title": safe_filename,
            "file_type": file_extension[1:],
            "file_size": file_size,
            "content_length": len(document_text),
            "elements_extracted": len(elements)
        }
        
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        error_trace = traceback.format_exc()
        logger.error(f"Upload error: {error_trace}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
        
    finally:
        # Cleanup temp file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file: {e}")

@app.post("/query")
async def query_documents(
    request: QueryRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Enhanced document querying with LLM integration (TRUE RAG)"""
    try:
        query = request.query.strip()
        
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        if len(query) > 1000:
            raise HTTPException(status_code=400, detail="Query too long (max 1000 characters)")
        
        # Check if user has documents
        user_docs_query = db.query(Document).filter(Document.user_id == current_user.id)
        
        if request.document_ids:
            user_docs_query = user_docs_query.filter(Document.id.in_(request.document_ids))
        
        user_doc_count = user_docs_query.count()
        
        if user_doc_count == 0:
            message = "No documents available for querying."
            if request.document_ids:
                message += " The specified document IDs were not found or don't belong to you."
            else:
                message += " Please upload documents first."
                
            return JSONResponse(content={
                "response": message,
                "total_matches": 0,
                "documents_searched": 0,
                "query": query
            })
        
        # Perform search (RETRIEVAL step)
        search_results = advanced_text_search(query, db, current_user.id, request.document_ids)
        
        if search_results:
            # Get top results and combine context
            top_results = search_results[:3]  # Top 3 matches for context
            
            # Prepare context from retrieved documents
            context_snippets = []
            source_docs = []
            
            for result in top_results:
                doc = result['document']
                snippet = extract_relevant_snippet(doc.content, query, 500)
                context_snippets.append(f"Document: {doc.title}\nContent: {snippet}")
                
                source_docs.append({
                    "id": result['id'],
                    "title": doc.title,
                    "score": round(result['score'], 2),
                    "file_type": doc.file_type,
                    "upload_time": doc.upload_time.isoformat() if doc.upload_time else None
                })
            
            # Combine context
            combined_context = "\n\n---\n\n".join(context_snippets)
            
            # Create prompt for LLM (GENERATION step)
            prompt = f"""Based on the following documents, please answer the user's question comprehensively and accurately.

Context from documents:
{combined_context}

User Question: {query}

Please provide a helpful and accurate answer based on the information in the documents. If the documents don't contain enough information to fully answer the question, please indicate what information is available and what might be missing."""

            # Generate response using LLM
            try:
                llm_response = generate_response_with_llama([{"content": prompt}])
                
                # Clean up the response (remove the prompt echo if present)
                if prompt in llm_response:
                    llm_response = llm_response.replace(prompt, "").strip()
                
                logger.info(f"RAG Query processed: '{query}' by {current_user.username} - {len(search_results)} matches")
                
                return JSONResponse(content={
                    "response": llm_response,
                    "total_matches": len(search_results),
                    "documents_searched": user_doc_count,
                    "source_documents": source_docs,
                    "query": query,
                    "is_ai_generated": True
                })
                
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                # Fallback to snippet-based response
                fallback_response = f"I found relevant information in your documents, but couldn't generate a comprehensive answer. Here's what I found:\n\n{combined_context}"
                
                return JSONResponse(content={
                    "response": fallback_response,
                    "total_matches": len(search_results),
                    "documents_searched": user_doc_count,
                    "source_documents": source_docs,
                    "query": query,
                    "is_ai_generated": False,
                    "note": "LLM processing failed, showing raw document content"
                })
                
        else:
            # No matches found - generate helpful response
            no_match_prompt = f"""The user asked: "{query}"

No relevant documents were found in their uploaded files. Please provide a helpful response explaining that no relevant information was found and suggest what they might do next (such as uploading more relevant documents, trying different search terms, etc.)."""

            try:
                llm_response = generate_response_with_llama([{"content": no_match_prompt}])
                
                return JSONResponse(content={
                    "response": llm_response,
                    "total_matches": 0,
                    "documents_searched": user_doc_count,
                    "query": query,
                    "is_ai_generated": True
                })
            except Exception as e:
                logger.error(f"LLM generation failed for no-match case: {e}")
                return JSONResponse(content={
                    "response": f"No relevant content found for query: '{query}'. Try using different keywords or upload more documents.",
                    "total_matches": 0,
                    "documents_searched": user_doc_count,
                    "suggestions": [
                        "Try broader search terms",
                        "Use synonyms or related terms", 
                        "Check if your documents contain relevant content",
                        "Upload more relevant documents"
                    ],
                    "query": query,
                    "is_ai_generated": False
                })
            
    except HTTPException:
        raise
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Query error: {error_trace}")
        raise HTTPException(status_code=500, detail="Query processing failed")

@app.get("/documents")
async def list_user_documents(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """List all documents for the current user"""
    try:
        documents = db.query(Document).filter(
            Document.user_id == current_user.id
        ).order_by(Document.upload_time.desc()).all()
        
        user_docs = []
        for doc in documents:
            content_preview = (doc.content[:200] + "...") if len(doc.content) > 200 else doc.content
            
            user_docs.append({
                "id": doc.id,
                "title": doc.title,
                "file_type": doc.file_type,
                "file_size": doc.file_size,
                "upload_time": doc.upload_time.isoformat() if doc.upload_time else None,
                "content_length": doc.content_length,
                "content_preview": content_preview
            })
        
        return {
            "documents": user_docs,
            "total_count": len(user_docs)
        }
    except Exception as e:
        logger.error(f"Error fetching documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve documents")

@app.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str, 
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a specific document"""
    try:
        doc = db.query(Document).filter(
            Document.id == doc_id,
            Document.user_id == current_user.id
        ).first()
        
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete from S3 if exists
        if doc.s3_url and s3_client and S3_BUCKET_NAME:
            try:
                s3_key = doc.s3_url.replace(f"s3://{S3_BUCKET_NAME}/", "")
                s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
                logger.info(f"Deleted from S3: {s3_key}")
            except Exception as e:
                logger.warning(f"S3 deletion failed: {e}")
        
        # Delete from Redis if available
        if redis_client:
            try:
                redis_client.delete(f"document:{doc_id}")
            except Exception as e:
                logger.warning(f"Redis deletion failed: {e}")
        
        # Delete from database
        doc_title = doc.title
        db.delete(doc)
        db.commit()
        
        logger.info(f"Document deleted: {doc_title} by {current_user.username}")
        
        return {
            "message": f"Document '{doc_title}' deleted successfully"
        }
    except Exception as e:
        logger.error(f"Delete document error: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete document")

@app.get("/health")
async def health_check(db: Session = Depends(get_db)):
    """Enhanced health check"""
    redis_status = False
    if redis_client:
        try:
            redis_client.ping()
            redis_status = True
        except Exception:
            pass
    
    s3_status = s3_client is not None and S3_BUCKET_NAME is not None
    
    # Database statistics
    try:
        total_users = db.query(User).count()
        total_documents = db.query(Document).count()
        db_status = True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        total_users = 0
        total_documents = 0
        db_status = False
    
    return {
        "status": "healthy" if db_status else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "services": {
            "database": db_status,
            "redis": redis_status,
            "s3": s3_status,
        },
        "statistics": {
            "total_users": total_users,
            "total_documents": total_documents,
            "storage_type": f"database ({'with redis' if redis_status else 'only'})",
            "max_file_size_mb": MAX_FILE_SIZE / 1024 / 1024
        },
        "configuration": {
            "database_type": "PostgreSQL",
            "database_url": DATABASE_URL.split("://")[0] + "://***",  # Hide credentials
            "upload_dir": UPLOAD_DIR,
            "token_expire_minutes": ACCESS_TOKEN_EXPIRE_MINUTES
        }
    }

@app.get("/")
async def root():
    return {
        "message": "Document Query RAG System API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "auth": ["/auth/register", "/auth/login"],
            "documents": ["/upload", "/documents", "/query"],
            "management": ["/health"]
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting server on {host}:{port}")
    logger.info(f"Database: PostgreSQL")
    logger.info(f"Database URL: {DATABASE_URL.split('://')[0]}://***")
    
    # Configure uvicorn for better timeout handling
    uvicorn.run(
        app, 
        host=host, 
        port=port,
        timeout_keep_alive=300,  # 5 minutes keep alive
        timeout_graceful_shutdown=30,
        limit_concurrency=10,    # Limit concurrent connections
        limit_max_requests=100,  # Limit max requests per worker
    )