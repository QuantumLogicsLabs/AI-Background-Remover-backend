# AI Background Remover — Backend (v2)

> **Team:** Web Team (API)  
> **Repo:** `AI-Background-Remover-backend`  
> **Parent repo:** `AI-Background-Remover` (this is a submodule)  
> **Tech:** Python 3.11 · FastAPI 0.115 · Uvicorn · MongoDB (Motor) · JWT auth

---

## What This Repo Is

The FastAPI backend for the AI Background Remover application. It receives image uploads from the React frontend, orchestrates AI processing, manages user authentication and quotas, stores job metadata in MongoDB, and serves the results back.

It acts as the **bridge between the UI and the AI pipeline** and implements all business logic, auth, and storage layers.  
It does not contain model code — it calls the AI package (the `AI/` submodule) via `services/bg_removal.py`.

---

## v2 Feature Set

| Feature                             | Endpoint(s)                                        | Status |
| ----------------------------------- | -------------------------------------------------- | ------ |
| User registration & login           | `POST /api/auth/register`, `POST /api/auth/login`  | ✅     |
| JWT auth + httpOnly refresh cookies | `POST /api/auth/refresh`, `POST /api/auth/logout`  | ✅     |
| Background removal                  | `POST /api/remove-background`                      | ✅     |
| Image enhancement                   | `POST /api/enhance`                                | ✅     |
| Background replacement              | `POST /api/replace-bg`                             | ✅     |
| Smart crop                          | `POST /api/smart-crop`                             | ✅     |
| Recolor & eraser                    | `POST /api/recolor`                                | ✅     |
| Inpainting                          | `POST /api/inpaint`                                | ✅     |
| Vectorize (PNG → SVG)               | `POST /api/vectorize`                              | ✅     |
| Batch processing                    | `POST /api/batch`                                  | ✅     |
| History & download                  | `GET /api/history`, `GET /api/download/{filename}` | ✅     |
| AI chatbot (vision-aware)           | `POST /api/chat`                                   | ✅     |
| AI image analysis                   | `POST /api/ai/analyze`                             | ✅     |
| Per-user daily quota                | Middleware on all AI routes                        | ✅     |
| Usage stats & analytics             | `GET /api/stats`, `GET /api/analytics`             | ✅     |
| Action history & undo               | `GET /api/action-history`                          | ✅     |
| Collaboration routes                | `POST /api/collab/*`                               | ✅     |
| Cloud storage (S3-compatible)       | `STORAGE_BACKEND=s3` env var                       | ✅     |
| Automatic file cleanup              | Background task every 60 mins                      | ✅     |
| Job queue                           | Async task queue for batch jobs                    | ✅     |

---

## Folder Structure

```
backend/
│
├── app.py                      ← FastAPI app, CORS, router registration,
│                                 MongoDB lifespan hooks (startup/shutdown),
│                                 job queue + cleanup tasks
│
├── routes/                     ← one file per feature area (20+ routes)
│   ├── auth.py                 ← registration, login, logout, refresh, me
│   ├── remove_bg.py            ← POST /api/remove-background
│   ├── enhance.py              ← POST /api/enhance
│   ├── replace_bg.py           ← POST /api/replace-bg
│   ├── smart_crop.py           ← POST /api/smart-crop
│   ├── recolor.py              ← POST /api/recolor
│   ├── inpaint.py              ← POST /api/inpaint
│   ├── vectorize.py            ← POST /api/vectorize
│   ├── batch.py                ← POST /api/batch
│   ├── download.py             ← GET /api/download/{filename}
│   ├── history.py              ← GET /api/history (per-user)
│   ├── history_all.py          ← GET /api/history/all (admin)
│   ├── images.py               ← DELETE /api/image/{id}
│   ├── chat.py                 ← POST /api/chat (AI chatbot)
│   ├── image.py                ← POST /api/ai/analyze (AI image analysis)
│   ├── stats.py                ← GET /api/stats
│   ├── analytics.py            ← GET /api/analytics
│   ├── action_history.py       ← GET /api/action-history
│   ├── collab.py               ← Collaboration routes
│   └── prompts.py              ← User-saved prompts
│
├── services/                   ← business logic, separated from routes
│   ├── bg_removal.py           ← async wrapper around AI inference
│   ├── enhancement.py          ← image enhancement (brightness, contrast, etc.)
│   ├── compositing.py          ← background replacement logic
│   ├── smart_crop.py           ← smart crop calculations
│   ├── recolor.py              ← pixel recolouring
│   ├── inpainting.py           ← object erasure / inpainting
│   ├── ai_service.py           ← Gemini / Groq chat + vision API wrapper
│   ├── auth.py                 ← JWT token generation + validation
│   ├── database.py             ← Motor (async MongoDB) connection helpers
│   ├── storage.py              ← local disk OR S3-compatible cloud storage
│   ├── quota.py                ← per-user daily quota enforcement
│   ├── cleanup.py              ← background task — auto-delete old files
│   ├── job_queue.py            ← async task queue for batch processing
│   ├── tracking.py             ← usage event & action history tracking
│   ├── batch.py                ← batch orchestration logic
│   └── email.py                ← email notifications (optional)
│
├── models/                     ← Pydantic schemas + MongoDB models
│   ├── user.py                 ← User, UserCreate, UserLogin, UserOut
│   ├── ai.py                   ← AI-related schemas
│   ├── analytics.py            ← Analytics event schemas
│   └── collaboration.py        ← Collaboration schemas
│
├── uploads/                    ← incoming images are saved here temporarily
│   └── .gitkeep
│
├── output/                     ← processed transparent PNGs live here
│   └── .gitkeep
│
├── .env.example                ← all backend env vars documented
├── pytest.ini                  ← pytest configuration
└── README.md                   ← you are here
```

---

## API Reference (v2)

### Auth

#### `POST /api/auth/register`

Create an **unverified** account and email a 6-digit code. The user is **not** signed in yet.

Only Google accounts are accepted: the address must use a domain listed in `ALLOWED_EMAIL_DOMAINS` (default `gmail.com`, `googlemail.com`).

**Request body:**

```json
{
  "name": "John Doe",
  "email": "john@gmail.com",
  "password": "securepassword123"
}
```

**Response:** `201 Created`

```json
{ "email": "john@gmail.com", "verification_required": true }
```

**Errors:** `409` account already exists · `422` invalid or non-Google email · `503` the email could not be sent

---

#### `POST /api/auth/verify-email`

Confirm the 6-digit code. On success the account is marked verified and the user is signed in (same response as login, plus the `refresh_token` cookie).

**Request body:**

```json
{ "email": "john@gmail.com", "code": "482913" }
```

Codes expire after **10 minutes**, can be used **once**, and are locked after **5 wrong attempts** (request a new code). Failures return `400`.

---

#### `POST /api/auth/resend-verification`

Send a new code (at most one per 60 seconds per address). Always returns `202` with the same message whether or not the account exists, so emails cannot be enumerated.

**Request body:** `{ "email": "john@gmail.com" }`

---

#### `POST /api/auth/email-change` and `POST /api/auth/email-change/confirm`

Change the account email in two steps (requires `Authorization: Bearer <token>`).

1. `POST /api/auth/email-change` with `{ "email": "new@gmail.com" }` emails a code to the **new** address (`202`).
2. `POST /api/auth/email-change/confirm` with `{ "code": "482913" }` switches the account to the new address and returns the updated user.

`PATCH /api/auth/profile` only updates `name`; it never changes the email.

---

#### `POST /api/auth/login`

Login, receive JWT access token + httpOnly refresh cookie.

**Request body:**

```json
{
  "email": "john@example.com",
  "password": "securepassword123"
}
```

**Response:** `200 OK`

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "user": {
    "user_id": "uuid-string",
    "name": "John Doe",
    "email": "john@example.com",
    "created_at": "2026-09-01T10:00:00Z"
  }
}
```

Also sets a secure httpOnly cookie named `refresh_token` (30 days TTL).

If the password is correct but the email is not verified yet, login returns `403` with
`{"detail": {"code": "email_not_verified", "message": "...", "email": "..."}}`
and emails a fresh code. Finish with `POST /api/auth/verify-email`.

---

#### `POST /api/auth/refresh`

Refresh access token using the httpOnly refresh cookie.

**Response:** `200 OK`

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

---

#### `POST /api/auth/logout`

Logout, clear refresh cookie.

**Response:** `200 OK`

```json
{ "message": "Logged out successfully." }
```

---

#### `GET /api/auth/me`

Get current user info.

**Response:** `200 OK`

```json
{
  "user_id": "uuid-string",
  "name": "John Doe",
  "email": "john@example.com",
  "created_at": "2026-09-01T10:00:00Z"
}
```

---

### Background Removal

#### `POST /api/remove-background`

Upload an image and get a transparent PNG back.

**Request:** `multipart/form-data`
| Field | Type | Required | Notes |
|---|---|---|---|
| `file` | File | Yes | JPEG, PNG, or WebP. Max 10 MB. |
| `quality` | string | No | `fast` (default), `standard`, `quality` |

**Response:** `200 OK`

```json
{
  "output_filename": "abc123_result.png",
  "download_url": "/api/download/abc123_result.png",
  "quality": "fast"
}
```

---

### Enhancement

#### `POST /api/enhance`

Enhance image with brightness, contrast, saturation, sharpness adjustments.

**Request body:**

```json
{
  "image_data": "base64-encoded-image-string",
  "brightness": 1.2,
  "contrast": 1.1,
  "saturation": 1.0,
  "sharpness": 1.1,
  "denoise": false,
  "auto_wb": true,
  "denoise_strength": 9
}
```

**Response:** `200 OK`

```json
{
  "enhanced_image": "base64-encoded-result"
}
```

---

### Background Replacement

#### `POST /api/replace-bg`

Replace background with solid colour, gradient, or uploaded image.

---

### Smart Crop

#### `POST /api/smart-crop`

AI-guided crop to a specific aspect ratio with subject-aware padding.

---

### History

#### `GET /api/history`

Returns the current user's last 50 processing jobs, newest first.

**Response:** `200 OK`

```json
[
  {
    "upload_id": "abc123",
    "user_id": "uuid-string",
    "original_name": "photo.jpg",
    "output_filename": "abc123_result.png",
    "download_url": "/api/download/abc123_result.png",
    "quality": "fast",
    "created_at": "2026-09-01T10:22:00Z"
  }
]
```

---

### Download

#### `GET /api/download/{filename}`

Download a processed image file.

**Response:** PNG file stream (`image/png`)  
**Errors:** `404` if file not found.

---

### Delete

#### `DELETE /api/image/{id}`

Deletes a processed image from storage and removes its MongoDB record.

**Path param:** `id` — the `upload_id` (UUID portion before `_result.png`)  
**Response:** `200 OK`

```json
{ "message": "Image abc123 deleted successfully." }
```

---

### AI Chatbot

#### `POST /api/chat`

Chat with AI assistant (vision-aware, action-aware).

**Request body:**

```json
{
  "message": "Make the background white",
  "image_data": "base64-encoded-image-string" // optional
}
```

**Response:** `200 OK`

```json
{
  "reply": "I'll apply a clean white background for you.",
  "thinking": "The user wants a professional studio look...",
  "action": {
    "type": "apply_bg",
    "bgType": "solid",
    "solidColor": "#ffffff"
  }
}
```

---

### AI Image Analysis

#### `POST /api/ai/analyze`

Get comprehensive AI analysis: subject, quality scores, colour palette, recommendations.

**Response:** `200 OK`

```json
{
  "subject": "A vibrant green parrot on a perch",
  "image_type": "Wildlife/Pet",
  "quality_score": 94,
  "edge_score": 95,
  "lighting_score": 90,
  "color_palette": [
    { "hex": "#2E7D32", "name": "Emerald Green", "percentage": 45 }
  ],
  "editing_recommendations": [
    "Preserve fine feather edge boundaries during alpha matting"
  ]
}
```

---

### Health Check

#### `GET /`

```json
{ "status": "ok", "message": "AI Background Remover API v2 is running." }
```

---

## How a Request Flows Through the Code

```
POST /api/remove-background (requires JWT)
        │
        ▼
routes/remove_bg.py
  1. Dependency: get_current_user (validates JWT)
  2. Dependency: check_and_increment_quota (checks daily limit)
  3. Validate content_type (JPEG/PNG/WebP only)
  4. Validate file size (≤ 10 MB)
  5. Call services/bg_removal.py → remove_background_bytes()
        │
        ▼
  services/bg_removal.py
    Runs AI inference in thread pool executor
    (non-blocking — FastAPI event loop stays free)
        │
        ▼
  AI pipeline (AI/ submodule)
    preprocess → rembg/onnx/torch → postprocess → PNG bytes
        │
        ▼
  routes/remove_bg.py (continued)
  6. Save result PNG to storage (local or S3)
  7. Write job metadata to MongoDB via services/database.py
  8. Track usage event via services/tracking.py
  9. Return { output_filename, download_url, quality }
```

---

## Running Locally

### Prerequisites

- Python 3.11+
- MongoDB running on `localhost:27017` (or use Docker Compose from parent repo)
- The AI submodule present at `../AI/`

### Setup

```bash
# From the parent repo root, activate the virtual environment
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux

# Install all dependencies (from parent root)
pip install -r requirements.txt

# Copy the backend-specific environment config
cp backend/.env.example backend/.env

# Edit backend/.env
#   - Set SECRET_KEY (generate: python -c "import secrets; print(secrets.token_hex(32))")
#   - Set GEMINI_API_KEY (or GROQ_API_KEY)
#   - Set MONGO_URI if your MongoDB is not on localhost
```

### Run

```bash
cd backend
uvicorn app:app --reload --port 8000
```

Interactive API docs: http://localhost:8000/docs  
Alternative docs (ReDoc): http://localhost:8000/redoc

---

## Environment Variables

All variables are loaded from `backend/.env` (next to `app.py`) via `python-dotenv`.
Copy `backend/.env.example` to `backend/.env` to get started.

See `backend/.env.example` for the full documented list (30+ variables).

**Key required variables:**

| Variable                      | Description                                                                         |
| ----------------------------- | ----------------------------------------------------------------------------------- |
| `SECRET_KEY`                  | JWT signing secret — use `python -c "import secrets; print(secrets.token_hex(32))"` |
| `MONGO_URI`                   | MongoDB connection string                                                           |
| `GEMINI_API_KEY`              | Google Gemini API key (for chat / analysis)                                         |
| `ALLOWED_ORIGINS`             | Comma-separated frontend URLs for CORS                                              |
| `SMTP_HOST` / `SMTP_PORT`     | Mail server (Gmail: `smtp.gmail.com` / `587`)                                       |
| `SMTP_USER` / `SMTP_PASSWORD` | Sender login. For Gmail use an **App Password**, not your normal password           |
| `SMTP_FROM`                   | "From" address (defaults to `SMTP_USER`)                                            |
| `ALLOWED_EMAIL_DOMAINS`       | Comma-separated domains allowed to sign up (default `gmail.com,googlemail.com`)     |

---

## Email Verification & SMTP Setup

New password accounts must verify their email with a 6-digit code before they can sign in.
Accounts created with "Continue with Google" are verified by Google; GitHub accounts must also use an allowed domain.

**Development (no SMTP):** leave the `SMTP_*` variables empty. Nothing is emailed; the code is
printed in the backend terminal instead:

```
[EMAIL] ── Verification Code (DEV MODE) ──
  To:    you@gmail.com
  Code:  482913
```

**Sending real emails with Gmail:**

1. Use a dedicated sender Gmail account and turn on **2-Step Verification**.
2. Create an App Password at <https://myaccount.google.com/apppasswords>.
3. Set in `backend/.env`:
   ```env
   SMTP_HOST=smtp.gmail.com
   SMTP_PORT=587
   SMTP_USER=your-sender@gmail.com
   SMTP_PASSWORD=your-16-char-app-password
   SMTP_FROM=your-sender@gmail.com
   ```
4. **Restart the backend** (`.env` is read at startup).

If sending fails you will see `[EMAIL] Failed to send verification code ... 535`, which almost always means a
normal password was used instead of an App Password.

**Existing accounts:** users created before verification existed have no `email_verified` field and are asked to
verify on their next login. To skip that for everyone already in the database, run once:

```js
db.users.updateMany(
  { email_verified: { $exists: false } },
  { $set: { email_verified: true } },
);
```

---

## Database (MongoDB)

**Database:** `ai_bg_remover` (configurable via `MONGO_DB_NAME` env var)

**Collections:**

| Collection       | Purpose                                                |
| ---------------- | ------------------------------------------------------ |
| `users`          | User accounts (email, password hash, created_at)       |
| `history`        | Processing history (per user, per operation)           |
| `quota`          | Daily usage counts (TTL auto-expires at midnight UTC)  |
| `analytics`      | Usage events, AI suggestions, applied/rejected actions |
| `prompts`        | User-saved prompt templates                            |
| `action_history` | Undo/redo stack per user session                       |

The connection is opened on FastAPI startup and closed on shutdown — both handled in `app.py`'s lifespan context manager.

---

## Adding a New Route

1. Create `routes/your_feature.py` with an `APIRouter`.
2. Write your endpoint functions. Use `Depends(get_current_user)` for auth.
3. Import and register the router in `app.py`:
   ```python
   from routes.your_feature import router as your_router
   app.include_router(your_router, prefix="/api")
   ```
4. If you need database operations, call `get_collection()` from `services/database.py`.

---

## Adding a New Service

1. Create `services/your_service.py`.
2. Keep it `async` — FastAPI runs on an async event loop.
3. For CPU-heavy work (like AI calls), use `loop.run_in_executor(None, ...)` to avoid blocking.

---

## Testing

## Testing

```bash
cd backend
pytest
```

Tests live in `backend/tests/`. MongoDB is replaced by an in-memory fake and **no emails are sent**:
the `sent_codes` fixture in `tests/conftest.py` captures verification codes so tests can read them
with `code_sent_to(sent_codes, "user@gmail.com")`.

Seeded users in tests must include `"email_verified": True`, otherwise they are treated as unverified.

---

## Contribution

See [CONTRIBUTING.md](../CONTRIBUTING.md) in the parent repo for branch naming, commit format, and PR rules.

Your branch always goes into this submodule repo (`AI-Background-Remover-backend`), not the parent.

---

## Collaboration & Sharing

- **Share conversations** — export a chat conversation as PDF or plain text.
  `POST/GET /api/chat-history`, `GET /api/export/{conversation_id}?format=pdf|text`
- **Collaborative analysis** — multiple users can comment on the same image.
  `POST/GET /api/collab/{image_id}/comments`
- **AI action history** — tracks which AI suggestions were shown/applied.
  `POST/GET /api/actions`
- **Prompt templates** — save, list, reuse, and delete effective AI prompts.
  `POST/GET /api/prompts`, `POST /api/prompts/{id}/use`, `DELETE /api/prompts/{id}`

## Analytics & Insights

- **Usage analytics** — tracks which AI features are used most.
  `POST/GET /api/analytics/usage`
- **Success metrics** — how often AI suggestions are applied vs. just suggested.
  `GET /api/analytics/success`
- **Cost tracking** — logs AI API token usage and estimated cost per feature.
  `POST/GET /api/analytics/cost`
- **Quality feedback** — 1–5 star rating on AI suggestions.
  `POST /api/analytics/feedback`, `GET /api/analytics/feedback/summary`

Usage and action tracking is wired automatically into the `remove_bg`, `enhance`,
`replace_bg`, `smart_crop`, `recolor`, and `chat` routes via `services/tracking.py`,
so no extra frontend calls are needed to collect this data.

---

## Persistent Chat History & Multi-Turn Memory

- **Real conversation memory** — chat messages are sent to the AI as actual
  multi-turn history (a Gemini chat session / OpenAI-style message list), not
  a pasted text prefix, so the assistant remembers prior turns properly.
  `POST /api/chat` now accepts an optional `conversation_id` and a JSON-encoded
  `history` array (`[{role, content}, ...]`) in the form body.
- **Persisted conversations** — every chat turn is saved per user in a new
  `conversations` collection, keyed by `user_id` + `conversation_id`, so a
  conversation survives page refreshes and reopening the widget.
- **Restore last conversation** — `GET /api/chat/history` returns the most
  recently updated conversation (or a specific one via `?conversation_id=`).
- **Browse all past conversations** — `GET /api/chat/conversations` lists all
  of a user's conversations (preview text, message count, last-updated time),
  powering a dedicated **History tab** in the chatbot widget so users can
  reopen any earlier conversation, not just the latest one.
- **Clear conversation** — `DELETE /api/chat/history` clears a conversation
  (or all of a user's conversations), wired to a "Clear conversation" button
  in the widget header.
