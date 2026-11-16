import uuid
import json
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, status, File, UploadFile, Form, Depends, Query
from typing import Optional, List
from models.ai import ChatResponse, ChatHistoryResponse, ChatMessage, ConversationSummary, ConversationListResponse
from services.ai_service import AIService
from services.image_service import ImageService
from services.chat_validation import validate_message, parse_history
from services.upload_validation import read_upload_limited
from services.auth import get_current_user
from services.tracking import track_usage, track_cost
from services.database import get_collection
from models.user import UserOut

router = APIRouter(prefix="/chat", tags=["chat"])
ai_service = AIService()
image_service = ImageService()

MAX_IMAGE_MB = 10


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token) when the provider doesn't return exact counts."""
    return max(1, len(text) // 4)


@router.post("", response_model=ChatResponse)
async def chat(
    message: str = Form(...),
    file: Optional[UploadFile] = File(None),
    conversation_id: Optional[str] = Form(None),
    history: Optional[str] = Form(None),  # JSON-encoded list of {role, content}
    current_user: UserOut = Depends(get_current_user)
):
    try:
        message = validate_message(message)
        image_bytes = None
        if file:
            raw_bytes = await read_upload_limited(file, MAX_IMAGE_MB)

            if not image_service.validate(raw_bytes):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid image format. Supported formats are: JPEG, PNG, WEBP."
                )

            image_bytes = image_service.preprocess(raw_bytes)

        # ── Parse incoming history (best-effort — bad JSON is treated as empty) ──
        parsed_history = parse_history(history)

        # ── Persist this turn (user + assistant messages) to the conversation ──
        try:
            collection = get_collection("conversations")
            now = datetime.now(timezone.utc)
            await collection.update_one(
                {"user_id": current_user.user_id, "conversation_id": conv_id},
                {
                    "$push": {
                        "messages": {
                            "$each": [
                                {"role": "user", "content": message, "created_at": now},
                                {"role": "assistant", "content": reply, "created_at": now},
                            ]
                        }
                    },
                    "$setOnInsert": {
                        "user_id": current_user.user_id,
                        "conversation_id": conv_id,
                        "created_at": now,
                    },
                    "$set": {"updated_at": now},
                },
                upsert=True,
            )
        except Exception:
            pass  # persistence is best-effort; chat should still work

        # ── Analytics & Insights: usage + cost tracking (best-effort) ───────
        await track_usage(
            user_id=current_user.user_id,
            feature="chat",
            metadata={"has_image": bool(image_bytes)},
        )
        input_tokens = _estimate_tokens(message)
        output_tokens = _estimate_tokens(reply)
        await track_cost(
            user_id=current_user.user_id,
            feature="chat",
            provider=ai_service.provider,
            model=ai_service.vision_model if image_bytes else ai_service.chat_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        return ChatResponse(reply=reply, thinking=thinking, conversation_id=conv_id, action=action)
    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/conversations", response_model=ConversationListResponse)
async def list_conversations(
    current_user: UserOut = Depends(get_current_user),
):
    """List all of the current user's past conversations, most recently updated first."""
    try:
        collection = get_collection("conversations")
        cursor = collection.find(
            {"user_id": current_user.user_id}
        ).sort("updated_at", -1).limit(50)
        docs = await cursor.to_list(length=50)

        summaries = []
        for doc in docs:
            messages = doc.get("messages", [])
            last_msg = messages[-1]["content"] if messages else ""
            preview = (last_msg[:80] + "…") if len(last_msg) > 80 else last_msg
            updated_at = doc.get("updated_at") or doc.get("created_at")
            summaries.append(ConversationSummary(
                conversation_id=doc["conversation_id"],
                preview=preview or "(empty conversation)",
                message_count=len(messages),
                updated_at=updated_at.isoformat() if updated_at else "",
            ))

        return ConversationListResponse(conversations=summaries)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not list conversations: {exc}")


@router.get("/history", response_model=ChatHistoryResponse)
async def get_chat_history(
    conversation_id: Optional[str] = Query(None, description="Specific conversation to fetch; omit for the most recent one"),
    current_user: UserOut = Depends(get_current_user),
):
    """
    Return the current user's chat history.
    If conversation_id is omitted, returns the most recently updated conversation.
    """
    try:
        collection = get_collection("conversations")
        query = {"user_id": current_user.user_id}
        if conversation_id:
            query["conversation_id"] = conversation_id

        doc = await collection.find_one(query, sort=[("updated_at", -1)])
        if doc is None:
            # No conversation yet — return an empty, fresh conversation id.
            return ChatHistoryResponse(conversation_id=str(uuid.uuid4()), messages=[])

        messages = [
            ChatMessage(role=m["role"], content=m["content"])
            for m in doc.get("messages", [])
        ]
        return ChatHistoryResponse(conversation_id=doc["conversation_id"], messages=messages)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not fetch chat history: {exc}")


@router.delete("/history")
async def delete_chat_history(
    conversation_id: Optional[str] = Query(None, description="Specific conversation to delete; omit to delete all of the user's conversations"),
    current_user: UserOut = Depends(get_current_user),
):
    """Clear the current user's chat history."""
    try:
        collection = get_collection("conversations")
        query = {"user_id": current_user.user_id}
        if conversation_id:
            query["conversation_id"] = conversation_id

        result = await collection.delete_many(query)
        return {"status": "cleared", "deleted_count": result.deleted_count}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not clear chat history: {exc}")
