from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import pathlib
import os
import io
import json
from typing import Optional

import numpy as np
try:
    import torch
except Exception:
    torch = None
try:
    import faiss
except Exception:
    faiss = None
try:
    from pymongo import MongoClient
except Exception:
    MongoClient = None

from backend.model_utils import preprocess_image, load_model, load_gallery_map, reduce_embedding_dimension

app = FastAPI()

# Allow frontend (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globals for lazy-loaded resources
_MODEL = None
_INDEX = None
_GALLERY = None
_EMB_DIM = None
_MONGO_CLIENT = None
_MONGO_DB = None
_IMAGES_COLLECTION = None


def _ensure_resources():
    global _MODEL, _INDEX, _GALLERY, _EMB_DIM, _MONGO_CLIENT, _MONGO_DB, _IMAGES_COLLECTION
    if _MODEL is not None and _INDEX is not None:
        return

    MODEL_PATH = os.environ.get('MODEL_PATH', 'models/triplet_model.pth')
    INDEX_PATH = os.environ.get('INDEX_PATH', 'models/fashion_train_index.faiss')
    GALLERY_PATH = os.environ.get('GALLERY_PATH', 'models/gallery.json')
    MONGO_URI = os.environ.get('MONGO_URI')

    # Load model
    if _MODEL is None:
        if not torch:
            raise RuntimeError('torch not available on server')
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f'Model file not found: {MODEL_PATH}')
        _MODEL = load_model(MODEL_PATH, device='cpu')

    # Load FAISS index
    if _INDEX is None:
        if not faiss:
            raise RuntimeError('faiss not available on server')
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError(f'FAISS index file not found: {INDEX_PATH}')
        _INDEX = faiss.read_index(INDEX_PATH)
        try:
            _EMB_DIM = _INDEX.d
        except Exception:
            _EMB_DIM = None

    # Load gallery mapping if present (local files only)
    _GALLERY = load_gallery_map(GALLERY_PATH)
    
    # Connect to MongoDB if configured
    if MONGO_URI and _MONGO_CLIENT is None:
        try:
            if MongoClient is None:
                raise RuntimeError('pymongo not installed')
            _MONGO_CLIENT = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            _MONGO_CLIENT.admin.command('ping')
            _MONGO_DB = _MONGO_CLIENT['fashion_db']
            _IMAGES_COLLECTION = _MONGO_DB['images']
            print("[INFO] Connected to MongoDB")
        except Exception as e:
            print(f"[WARNING] MongoDB connection failed: {e}. Will use local files only.")
            _MONGO_CLIENT = None


@app.post('/api/search')
async def search_similar(image: UploadFile = File(...), top_k: int = 6):
    try:
        _ensure_resources()
    except FileNotFoundError as e:
        return JSONResponse(status_code=500, content={'detail': str(e)})
    except RuntimeError as e:
        return JSONResponse(status_code=500, content={'detail': str(e)})

    body = await image.read()
    if not body:
        raise HTTPException(status_code=400, detail='No image uploaded')

    try:
        tensor = preprocess_image(body, size=224)
    except Exception as e:
        return JSONResponse(status_code=500, content={'detail': f'Preprocessing error: {str(e)}'})

    try:
        with torch.no_grad():
            out = _MODEL(tensor)
        if isinstance(out, torch.Tensor):
            vec = out.squeeze().cpu().numpy().astype('float32')
        else:
            vec = np.array(out).squeeze().astype('float32')
    except Exception as e:
        return JSONResponse(status_code=500, content={'detail': f'Model inference error: {str(e)}'})

    if vec.ndim == 1:
        q = np.expand_dims(vec, axis=0).astype('float32')
    else:
        q = vec.astype('float32')

    # Debug: Print dimensions
    print(f'[DEBUG] Query vector shape: {q.shape}')
    print(f'[DEBUG] Query vector dimension (cols): {q.shape[1] if q.ndim > 1 else 1}')
    print(f'[DEBUG] FAISS index dimension: {_EMB_DIM}')
    print(f'[DEBUG] First 10 embedding values: {q[0, :10]}')
    print(f'[DEBUG] Vector L2 norm: {np.linalg.norm(q[0])}')  # Should be ~1.0 after normalization
    
    # L2 normalize (model should already do this, but ensure it for FAISS IndexFlatIP)
    try:
        if faiss:
            faiss.normalize_L2(q)
            print(f'[DEBUG] After L2 normalization: {q.shape}')
            print(f'[DEBUG] Vector L2 norm after normalization: {np.linalg.norm(q[0])}')  # Should be ~1.0
    except Exception as e:
        print(f'[WARNING] Could not L2 normalize: {e}')

    try:
        k = max(1, int(top_k))
        distances, indices = _INDEX.search(q, k)
        dists = distances[0].tolist()
        idxs = indices[0].tolist()
        print(f'[DEBUG] FAISS search returned {len(idxs)} results')
        print(f'[DEBUG] Top 3 distances (cosine similarities): {dists[:3]}')
    except Exception as e:
        import traceback
        error_msg = f'FAISS search error: {str(e)}\n{traceback.format_exc()}'
        print(error_msg)
        return JSONResponse(status_code=500, content={'detail': error_msg})

    results = []
    for idx, dist in zip(idxs, dists):
        # Always use the API endpoint to serve images
        # This way, the frontend can fetch them regardless of the file path structure
        url = f'/api/image/{idx}'
        
        try:
            sim = float(dist)
        except Exception:
            sim = 0.0
        results.append({'image_url': url, 'similarity': round(sim, 4), 'index': int(idx), 'distance': float(dist)})

    return JSONResponse(content={'results': results})


# Simple admin endpoints (token-protected via X-Admin-Token header)
ADMIN_TOKEN = os.environ.get('ADMIN_TOKEN')

def _check_admin(token: Optional[str]):
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail='Admin functionality not configured on this server.')
    if not token or token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail='Invalid admin token')


@app.post('/api/admin/rebuild-index')
async def admin_rebuild_index(x_admin_token: Optional[str] = Header(None)):
    _check_admin(x_admin_token)
    # In production, trigger index rebuild job / pipeline here
    return JSONResponse(content={'status': 'ok', 'message': 'Rebuild started (demo).'})


@app.post('/api/admin/upload-gallery')
async def admin_upload_gallery(file: UploadFile = File(...), x_admin_token: Optional[str] = Header(None)):
    _check_admin(x_admin_token)
    # Demo: acknowledge receipt
    return JSONResponse(content={'status': 'ok', 'filename': file.filename, 'message': 'Received (demo).'})


@app.post('/api/admin/run-test')
async def admin_run_test(x_admin_token: Optional[str] = Header(None)):
    _check_admin(x_admin_token)
    # Return a small dummy test
    dummy_results = [
        {'image_url': f'https://picsum.photos/seed/{i}/200/200', 'similarity': round(0.8 - i * 0.05, 3)}
        for i in range(3)
    ]
    return JSONResponse(content={'status': 'ok', 'results': dummy_results})


@app.get('/api/image/{idx:int}')
async def serve_gallery_image(idx: int):
    """Serve an image from the gallery by index.
    First tries MongoDB, then local files, then returns placeholder.
    """
    _ensure_resources()
    
    # Try MongoDB first
    if _IMAGES_COLLECTION:
        try:
            doc = _IMAGES_COLLECTION.find_one({'image_id': idx})
            if doc and 'image_data' in doc:
                print(f'[DEBUG] Serving image {idx} from MongoDB')
                return StreamingResponse(
                    io.BytesIO(doc['image_data']),
                    media_type='image/jpeg'
                )
        except Exception as e:
            print(f"[WARNING] MongoDB fetch failed: {e}")
    
    # Try local files
    if _GALLERY:
        image_path = _GALLERY.get(str(idx))
        if image_path:
            # Convert relative path to absolute
            if not os.path.isabs(image_path):
                project_root = pathlib.Path(__file__).resolve().parent.parent
                image_path = os.path.join(str(project_root), image_path)
            
            image_path = os.path.normpath(image_path)
            
            if os.path.exists(image_path):
                print(f'[DEBUG] Serving image {idx} from local file: {image_path}')
                return FileResponse(image_path, media_type='image/jpeg')
    
    # Fallback: placeholder
    print(f'[WARNING] Image {idx} not found in MongoDB or local files. Using placeholder.')
    return JSONResponse(
        status_code=200,
        content={'detail': f'Image {idx} not available', 'placeholder': True}
    )


# Serve the frontend static files from the `frontend` directory.
frontend_path = pathlib.Path(__file__).resolve().parent.parent / 'frontend'
if frontend_path.exists():
    app.mount('/', StaticFiles(directory=str(frontend_path), html=True), name='frontend')
