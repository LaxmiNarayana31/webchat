from pathlib import Path
from fastapi import APIRouter
from fastapi.responses import FileResponse, Response
from backend.app.core.logging import logger

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent.parent / "frontend"

router = APIRouter(tags=["Frontend"])
frontend_router = router

def get_base_dir() -> Path:
    dist_dir = FRONTEND_DIR / "dist"
    return dist_dir if dist_dir.exists() else FRONTEND_DIR

@router.get("/", include_in_schema=False)
async def serve_index():
    try:
        index_path = get_base_dir() / "index.html"
        if index_path.exists():
            return FileResponse(index_path, media_type="text/html")
        return Response(content="Frontend index.html not found.", status_code=404)
    except Exception as e:
        logger.error(f"Error serving frontend index: {e}", exc_info=True)
        return Response(content="Internal server error loading frontend.", status_code=500)

@router.get("/index.html", include_in_schema=False)
async def serve_index_html():
    try:
        index_path = get_base_dir() / "index.html"
        if index_path.exists():
            return FileResponse(index_path, media_type="text/html")
        return Response(content="Frontend index.html not found.", status_code=404)
    except Exception as e:
        logger.error(f"Error serving index.html: {e}", exc_info=True)
        return Response(content="Internal server error loading frontend.", status_code=500)

@router.get("/assets/{filename}", include_in_schema=False)
async def serve_assets(filename: str):
    try:
        asset_path = FRONTEND_DIR / "dist" / "assets" / filename
        if asset_path.exists():
            if filename.endswith(".js"):
                return FileResponse(asset_path, media_type="application/javascript")
            if filename.endswith(".css"):
                return FileResponse(asset_path, media_type="text/css")
            return FileResponse(asset_path)
        return Response(content=f"Asset '{filename}' not found.", status_code=404)
    except Exception as e:
        logger.error(f"Error serving asset '{filename}': {e}", exc_info=True)
        return Response(content="Error retrieving asset.", status_code=500)

@router.get("/favicon.ico", include_in_schema=False)
@router.get("/favicon.svg", include_in_schema=False)
async def serve_favicon():
    try:
        # Check for favicon.svg first, then favicon.ico
        for name, media in [("favicon.svg", "image/svg+xml"), ("favicon.ico", "image/x-icon")]:
            fav_path = get_base_dir() / name
            if fav_path.exists():
                return FileResponse(fav_path, media_type=media)
            public_path = FRONTEND_DIR / "public" / name
            if public_path.exists():
                return FileResponse(public_path, media_type=media)
        return Response(status_code=204)
    except Exception as e:
        logger.error(f"Error serving favicon: {e}", exc_info=True)
        return Response(status_code=204)

@router.get("/{full_path:path}", include_in_schema=False)
async def serve_spa_fallback(full_path: str):
    if full_path.startswith("api/") or full_path.startswith("assets/"):
        return Response(content="Not Found", status_code=404)
    try:
        # Check if the requested file exists directly in dist or public
        candidate = get_base_dir() / full_path
        if candidate.is_file():
            media_type = "image/svg+xml" if full_path.endswith(".svg") else None
            return FileResponse(candidate, media_type=media_type)
        candidate_public = FRONTEND_DIR / "public" / full_path
        if candidate_public.is_file():
            media_type = "image/svg+xml" if full_path.endswith(".svg") else None
            return FileResponse(candidate_public, media_type=media_type)

        index_path = get_base_dir() / "index.html"
        if index_path.exists():
            return FileResponse(index_path, media_type="text/html")
        return Response(content="Frontend index.html not found.", status_code=404)
    except Exception as e:
        logger.error(f"Error serving SPA fallback: {e}", exc_info=True)
        return Response(content="Internal server error loading frontend.", status_code=500)

