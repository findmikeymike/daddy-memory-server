# DADDY Memory Server - Self-Hosted Mem0 OSS

Enterprise-grade memory layer for DADDY conversational AI using open-source Mem0.

## 🚀 Quick Start

```bash
# Clone and setup
cd memory-server
cp .env.example .env
# Edit .env with your OpenAI API key

# Start with Docker Compose
docker-compose up -d

# Verify it's running
curl http://localhost:8000/health
```

## 📊 Cost Analysis

**Self-hosted vs Mem0.ai SaaS:**
- **SaaS**: $249/month for Pro plan (1000 users)
- **Self-hosted**: ~$50/month Railway + OpenAI API costs
- **Savings**: ~$200/month (80% cost reduction)

## 🔧 API Endpoints

### Store Memory
```bash
POST /memories
{
  "user_id": "user123",
  "content": "User prefers morning workouts",
  "metadata": {"type": "preference"}
}
```

### Search Memories
```bash
POST /memories/search
{
  "user_id": "user123",
  "query": "workout preferences",
  "limit": 5
}
```

### ElevenLabs Webhook Integration
```bash
POST /webhook/elevenlabs/memory
{
  "tool_name": "store_daddy_memory",
  "parameters": {
    "user_id": "user123",
    "content": "User achieved fitness goal",
    "memory_type": "achievement"
  }
}
```

## 🔗 ElevenLabs Integration

1. **Deploy to Railway**
2. **Add webhook tool in ElevenLabs Dashboard:**
   - URL: `https://your-memory-server.railway.app/webhook/elevenlabs/memory`
   - Method: POST
   - Tools: `store_daddy_memory`, `retrieve_daddy_memories`

3. **DADDY automatically gets persistent memory across conversations**

## 🏗️ Architecture

- **Mem0 OSS**: Core memory processing
- **ChromaDB**: Vector embeddings storage
- **PostgreSQL**: Metadata and relationships
- **Redis**: Caching and sessions
- **FastAPI**: REST API server

## 🔒 Production Ready

- Health checks
- Logging
- Error handling
- CORS configuration
- Docker containerization
- Horizontal scaling ready
