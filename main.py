#!/usr/bin/env python3
"""
DADDY Memory Server - Self-hosted Mem0 OSS
Provides persistent memory for ElevenLabs conversational AI
"""

import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
from mem0 import Memory
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/memory_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Pydantic models
class MemoryRequest(BaseModel):
    user_id: str
    content: str
    metadata: Optional[Dict[str, Any]] = None

class MemorySearchRequest(BaseModel):
    user_id: str
    query: Optional[str] = None
    limit: int = 5

class MemoryResponse(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any]
    score: Optional[float] = None

# Global memory instance
memory_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup memory client"""
    global memory_client
    
    # Initialize Mem0 with configuration
    config = {
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": "daddy_memories",
                "path": "./chroma_db"
            }
        },
        "llm": {
            "provider": "openai",
            "config": {
                "model": "gpt-3.5-turbo",
                "temperature": 0.1
            }
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": "text-embedding-ada-002"
            }
        }
    }
    
    try:
        memory_client = Memory(config=config)
        logger.info("✅ Memory client initialized successfully")
        yield
    except Exception as e:
        logger.error(f"❌ Failed to initialize memory client: {e}")
        raise
    finally:
        logger.info("🔄 Shutting down memory client")

# Create FastAPI app
app = FastAPI(
    title="DADDY Memory Server",
    description="Self-hosted memory layer for DADDY conversational AI",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_memory_client():
    """Dependency to get memory client"""
    if memory_client is None:
        raise HTTPException(status_code=500, detail="Memory client not initialized")
    return memory_client

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "status": "healthy",
        "service": "daddy-memory-server",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "daddy-memory-server",
        "version": "1.0.0"
    }

@app.post("/memories", response_model=Dict[str, Any])
async def store_memory(
    request: MemoryRequest,
    memory: Memory = Depends(get_memory_client)
):
    """Store a new memory for a user"""
    try:
        # Create messages format for Mem0
        messages = [{"role": "user", "content": request.content}]
        
        # Add metadata with DADDY-specific context
        metadata = {
            "app": "daddy-mobile",
            "timestamp": "now",
            **(request.metadata or {})
        }
        
        # Store memory
        result = memory.add(
            messages=messages,
            user_id=request.user_id,
            metadata=metadata
        )
        
        logger.info(f"💾 Stored memory for user {request.user_id}")
        
        return {
            "success": True,
            "memory_id": result.get("id"),
            "message": "Memory stored successfully"
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to store memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memories/search", response_model=List[MemoryResponse])
async def search_memories(
    request: MemorySearchRequest,
    memory: Memory = Depends(get_memory_client)
):
    """Search and retrieve relevant memories for a user"""
    try:
        # Search memories
        if request.query:
            # Search with query
            results = memory.search(
                query=request.query,
                user_id=request.user_id,
                limit=request.limit
            )
        else:
            # Get all memories for user
            results = memory.get_all(
                user_id=request.user_id,
                limit=request.limit
            )
        
        # Format response
        memories = []
        for result in results:
            memories.append(MemoryResponse(
                id=result.get("id", ""),
                content=result.get("memory", ""),
                metadata=result.get("metadata", {}),
                score=result.get("score")
            ))
        
        logger.info(f"🔍 Retrieved {len(memories)} memories for user {request.user_id}")
        
        return memories
        
    except Exception as e:
        logger.error(f"❌ Failed to search memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memories/{user_id}", response_model=List[MemoryResponse])
async def get_user_memories(
    user_id: str,
    limit: int = 10,
    memory: Memory = Depends(get_memory_client)
):
    """Get all memories for a specific user"""
    try:
        results = memory.get_all(user_id=user_id, limit=limit)
        
        memories = []
        for result in results:
            memories.append(MemoryResponse(
                id=result.get("id", ""),
                content=result.get("memory", ""),
                metadata=result.get("metadata", {})
            ))
        
        logger.info(f"📋 Retrieved {len(memories)} memories for user {user_id}")
        
        return memories
        
    except Exception as e:
        logger.error(f"❌ Failed to get user memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/memories/{memory_id}")
async def delete_memory(
    memory_id: str,
    memory: Memory = Depends(get_memory_client)
):
    """Delete a specific memory"""
    try:
        memory.delete(memory_id=memory_id)
        
        logger.info(f"🗑️ Deleted memory {memory_id}")
        
        return {"success": True, "message": "Memory deleted successfully"}
        
    except Exception as e:
        logger.error(f"❌ Failed to delete memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/memories/user/{user_id}")
async def delete_user_memories(
    user_id: str,
    memory: Memory = Depends(get_memory_client)
):
    """Delete all memories for a user"""
    try:
        # Get all memories for user first
        memories = memory.get_all(user_id=user_id)
        
        # Delete each memory
        deleted_count = 0
        for mem in memories:
            memory.delete(memory_id=mem.get("id"))
            deleted_count += 1
        
        logger.info(f"🧹 Deleted {deleted_count} memories for user {user_id}")
        
        return {
            "success": True,
            "deleted_count": deleted_count,
            "message": f"Deleted {deleted_count} memories for user"
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to delete user memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ElevenLabs webhook endpoint for memory integration
@app.post("/webhook/elevenlabs/memory")
async def elevenlabs_memory_webhook(
    payload: Dict[str, Any],
    memory: Memory = Depends(get_memory_client)
):
    """Handle memory operations from ElevenLabs webhooks"""
    try:
        tool_name = payload.get("tool_name")
        parameters = payload.get("parameters", {})
        
        if tool_name == "store_daddy_memory":
            # Store memory from conversation
            user_id = parameters.get("user_id")
            content = parameters.get("content")
            memory_type = parameters.get("memory_type", "conversation")
            
            if not user_id or not content:
                raise HTTPException(status_code=400, detail="Missing user_id or content")
            
            messages = [{"role": "user", "content": content}]
            result = memory.add(
                messages=messages,
                user_id=user_id,
                metadata={
                    "type": memory_type,
                    "source": "elevenlabs_webhook",
                    "timestamp": "now"
                }
            )
            
            return {
                "success": True,
                "data": {"memory_id": result.get("id")},
                "message": f"Stored {memory_type} memory for user"
            }
            
        elif tool_name == "retrieve_daddy_memories":
            # Retrieve memories for conversation context
            user_id = parameters.get("user_id")
            query = parameters.get("query")
            limit = parameters.get("limit", 5)
            
            if not user_id:
                raise HTTPException(status_code=400, detail="Missing user_id")
            
            if query:
                results = memory.search(query=query, user_id=user_id, limit=limit)
            else:
                results = memory.get_all(user_id=user_id, limit=limit)
            
            # Format for ElevenLabs response
            memories_text = "\n".join([
                f"- {result.get('memory', '')}" for result in results
            ])
            
            return {
                "success": True,
                "data": {
                    "memories": memories_text,
                    "count": len(results)
                },
                "message": f"Retrieved {len(results)} memories"
            }
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown tool: {tool_name}")
            
    except Exception as e:
        logger.error(f"❌ Webhook error: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Webhook processing failed"
        }

if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Ensure required environment variables
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"❌ Missing required environment variables: {missing_vars}")
        exit(1)
    
    # Get port from environment
    port = int(os.getenv("PORT", 8000))
    
    # Run server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
