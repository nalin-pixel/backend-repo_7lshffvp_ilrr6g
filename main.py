import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, EmailStr

from database import db, create_document, get_documents
from schemas import ProfileCache, Message

app = FastAPI(title="Futuristic Portfolio API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
CACHE_TTL_MINUTES = int(os.getenv("CACHE_TTL_MINUTES", "60"))


def gh_headers():
    headers = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return headers


@app.get("/")
def read_root():
    return {"message": "Portfolio backend running", "time": datetime.utcnow().isoformat()}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set",
        "database_name": os.getenv("DATABASE_NAME") or "❌ Not Set",
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["connection_status"] = "Connected"
            try:
                response["collections"] = db.list_collection_names()
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️ Connected but Error: {str(e)[:80]}"
        else:
            response["database"] = "⚠️ Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:80]}"
    return response


# ---------- GitHub Fetch & Cache ----------

class AggregateResponse(BaseModel):
    profile: Dict[str, Any]
    repos: List[Dict[str, Any]]
    languages: Dict[str, int]
    readme: Optional[str] = None
    stats: Dict[str, Any]


def build_username_from_url(link: str) -> str:
    # Accept full URL or username
    if not link:
        raise HTTPException(status_code=400, detail="GitHub username or profile URL is required")
    link = link.strip()
    if "/" in link:
        # e.g., https://github.com/username
        parts = link.rstrip("/").split("/")
        return parts[-1]
    return link


def fetch_profile(username: str) -> Dict[str, Any]:
    r = requests.get(f"https://api.github.com/users/{username}", headers=gh_headers(), timeout=15)
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=f"Failed to fetch profile: {r.text}")
    return r.json()


def fetch_repos(username: str) -> List[Dict[str, Any]]:
    repos = []
    page = 1
    while True:
        r = requests.get(
            f"https://api.github.com/users/{username}/repos",
            params={"per_page": 100, "page": page, "sort": "updated"},
            headers=gh_headers(),
            timeout=20,
        )
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=f"Failed to fetch repos: {r.text}")
        batch = r.json()
        repos.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return repos


def fetch_readme(username: str) -> Optional[str]:
    # Try to get profile README from special repo <username>/<username>
    r = requests.get(f"https://api.github.com/repos/{username}/{username}/readme", headers=gh_headers(), timeout=15)
    if r.status_code != 200:
        return None
    data = r.json()
    download_url = data.get("download_url")
    if not download_url:
        return None
    raw = requests.get(download_url, timeout=15)
    return raw.text if raw.status_code == 200 else None


def compute_languages(repos: List[Dict[str, Any]]) -> Dict[str, int]:
    totals: Dict[str, int] = {}
    for repo in repos:
        langs_url = repo.get("languages_url")
        if not langs_url:
            continue
        try:
            r = requests.get(langs_url, headers=gh_headers(), timeout=15)
            if r.status_code == 200:
                data = r.json()
                for lang, bytes_ in data.items():
                    totals[lang] = totals.get(lang, 0) + int(bytes_)
        except Exception:
            continue
    return dict(sorted(totals.items(), key=lambda x: x[1], reverse=True))


def summarize_stats(profile: Dict[str, Any], repos: List[Dict[str, Any]]) -> Dict[str, Any]:
    first_year = None
    for repo in repos:
        created = repo.get("created_at")
        if created:
            year = int(created[:4])
            first_year = year if first_year is None or year < first_year else first_year
    years_coding = max(1, datetime.utcnow().year - (first_year or datetime.utcnow().year))
    stars = sum([repo.get("stargazers_count", 0) for repo in repos])
    forks = sum([repo.get("forks_count", 0) for repo in repos])
    popular = sorted(repos, key=lambda r: (r.get("stargazers_count", 0), r.get("forks_count", 0)), reverse=True)[:6]
    return {
        "years_coding": years_coding,
        "repo_count": len(repos),
        "stars": stars,
        "forks": forks,
        "top_repos": [
            {
                "name": r.get("name"),
                "html_url": r.get("html_url"),
                "description": r.get("description"),
                "language": r.get("language"),
                "stargazers_count": r.get("stargazers_count"),
                "forks_count": r.get("forks_count"),
                "homepage": r.get("homepage"),
                "topics": r.get("topics", []),
            }
            for r in popular
        ],
    }


def cache_key(username: str) -> str:
    return f"gh::{username.lower()}"


@app.get("/api/github/aggregate")
def github_aggregate(link: str):
    username = build_username_from_url(link)

    # Check cache
    if db is not None:
        existing = db["profilecache"].find_one({"username": username})
        if existing:
            fetched_at = existing.get("fetched_at")
            try:
                if isinstance(fetched_at, str):
                    fetched_at = datetime.fromisoformat(fetched_at)
            except Exception:
                fetched_at = datetime.utcnow() - timedelta(minutes=CACHE_TTL_MINUTES + 1)
            if fetched_at and fetched_at > datetime.utcnow() - timedelta(minutes=CACHE_TTL_MINUTES):
                # Return cached
                data = existing.get("data", {})
                return data

    profile = fetch_profile(username)
    repos = fetch_repos(username)
    languages = compute_languages(repos)
    readme = fetch_readme(username)
    stats = summarize_stats(profile, repos)

    aggregate = {
        "profile": profile,
        "repos": stats["top_repos"],
        "languages": languages,
        "readme": readme,
        "stats": {k: v for k, v in stats.items() if k != "top_repos"},
    }

    # Cache
    if db is not None:
        try:
            db["profilecache"].update_one(
                {"username": username},
                {"$set": {"username": username, "data": aggregate, "fetched_at": datetime.utcnow()}},
                upsert=True,
            )
        except Exception:
            pass

    return aggregate


@app.get("/api/github/profile")
def github_profile(link: str):
    username = build_username_from_url(link)
    return fetch_profile(username)


@app.get("/api/github/repos")
def github_repos(link: str):
    username = build_username_from_url(link)
    return fetch_repos(username)


# ---------- Contact Endpoint ----------

class ContactIn(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    email: EmailStr
    message: str = Field(..., min_length=10, max_length=5000)
    website: Optional[str] = None


@app.post("/api/contact")
def contact(payload: ContactIn):
    doc = Message(**payload.model_dump())
    try:
        inserted_id = create_document("message", doc)
        return {"ok": True, "id": inserted_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
