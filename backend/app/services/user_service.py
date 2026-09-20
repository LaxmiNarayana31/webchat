from datetime import datetime, timezone
import time
from typing import Any, Dict, Optional, Tuple

from backend.app.core.logging import logger
from backend.app.models.models import GuestUsageEntity, UserEntity
from backend.config.database import USER_DAILY_REQUEST_LIMIT, get_db_session


class UserService:
    """Manages user identities, guest tracking, and tiered quota enforcement."""

    @staticmethod
    def get_current_date_str() -> str:
        """Returns current date in YYYY-MM-DD UTC format."""
        try:
            return datetime.now(timezone.utc).strftime("%Y-%m-%d")
        except Exception as e:
            logger.error(f"UserService: Error getting current date string: {e}", exc_info=True)
            return time.strftime("%Y-%m-%d")

    def identify_or_register_user(self, email: str, client_id: Optional[str] = None) -> Dict[str, Any]:
        """Retrieves or registers user account by email and resets daily counter if new day."""
        try:
            clean_email = email.strip().lower() if email else ""
            if not clean_email or "@" not in clean_email:
                raise ValueError("A valid email address is required.")

            today_str = self.get_current_date_str()
            now = time.time()

            with get_db_session() as session:
                user = session.query(UserEntity).filter(UserEntity.email == clean_email).first()
                if not user:
                    user = UserEntity(
                        email=clean_email,
                        created_at=now,
                        last_active_at=now,
                        daily_requests_count=0,
                        last_request_date=today_str,
                        total_requests_count=0,
                    )
                    session.add(user)
                    session.flush()
                    logger.info(f"Registered new user account: {clean_email}")

                # Claim any guest sessions created on this client device and link them to this email
                if client_id and client_id.strip():
                    from backend.app.models.models import ChatSessionEntity
                    guest_sessions = session.query(ChatSessionEntity).filter(
                        ChatSessionEntity.guest_client_id == client_id.strip(),
                        ChatSessionEntity.user_id.is_(None)
                    ).all()
                    for gs in guest_sessions:
                        gs.user_id = user.id
                    session.flush()
                    if guest_sessions:
                        logger.info(f"Claimed {len(guest_sessions)} guest sessions for user '{clean_email}'")
                else:
                    user.last_active_at = now  # type: ignore
                    # Check for daily reset
                    if user.last_request_date != today_str:
                        user.daily_requests_count = 0  # type: ignore
                        user.last_request_date = today_str  # type: ignore
                    session.flush()

                requests_used = user.daily_requests_count
                daily_limit = USER_DAILY_REQUEST_LIMIT
                remaining = max(0, daily_limit - requests_used)

                return {
                    "user_id": user.id,
                    "email": user.email,
                    "is_guest": False,
                    "daily_limit": daily_limit,
                    "requests_used_today": requests_used,
                    "requests_remaining": remaining,
                    "can_request": remaining > 0,
                }
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"UserService: Error identifying user '{email}': {e}", exc_info=True)
            raise

    def check_and_consume_quota(
        self,
        email: Optional[str] = None,
        client_id: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Enforces quota limit and consumes one request if within limits."""
        """Enforces dual-layer quota limit (Email or Client ID + IP) and consumes one request if within limits."""
        try:
            today_str = self.get_current_date_str()
            now = time.time()
            clean_ip = (ip_address or "127.0.0.1").strip()

            # Registered Email Account
            if email and email.strip():
                clean_email = email.strip().lower()
                with get_db_session() as session:
                    user = session.query(UserEntity).filter(UserEntity.email == clean_email).first()
                    if not user:
                        # Auto-register
                        user = UserEntity(
                            email=clean_email,
                            created_at=now,
                            last_active_at=now,
                            daily_requests_count=0,
                            last_request_date=today_str,
                            total_requests_count=0,
                        )
                        session.add(user)
                        session.flush()

                    user.last_active_at = now  # type: ignore
                    if user.last_request_date != today_str:
                        user.daily_requests_count = 0  # type: ignore
                        user.last_request_date = today_str  # type: ignore

                    if user.daily_requests_count >= USER_DAILY_REQUEST_LIMIT:
                        quota = {
                            "is_guest": False,
                            "email": clean_email,
                            "daily_limit": USER_DAILY_REQUEST_LIMIT,
                            "requests_used_today": user.daily_requests_count,
                            "requests_remaining": 0,
                            "can_request": False,
                        }
                        msg = (
                            f"Daily query limit ({USER_DAILY_REQUEST_LIMIT} requests/day) "
                            "reached for this account. Quota resets at midnight UTC."
                        )
                        return False, msg, quota

                    # Consume user request
                    user.daily_requests_count += 1  # type: ignore
                    user.total_requests_count += 1  # type: ignore
                    session.flush()

                    quota = {
                        "is_guest": False,
                        "email": clean_email,
                        "daily_limit": USER_DAILY_REQUEST_LIMIT,
                        "requests_used_today": user.daily_requests_count,
                        "requests_remaining": max(0, USER_DAILY_REQUEST_LIMIT - user.daily_requests_count),
                        "can_request": True,
                    }
                    return True, "", quota

            # Guest User (No Email)
            # Client Device Rate Limiting (No email prompt or email verification needed)
            # Guest User (Dual-Layer: Client ID + IP Address)
            guest_key = (client_id or "anonymous_guest").strip()
            with get_db_session() as session:
                guest = session.query(GuestUsageEntity).filter(GuestUsageEntity.client_id == guest_key).first()
                if not guest:
                    guest = GuestUsageEntity(
                        client_id=guest_key,
                        ip_address=clean_ip,
                        request_count=0,
                        created_at=now,
                        last_active_at=now,
                    )
                    session.add(guest)
                    session.flush()

                # Reset counter if 24 hours have elapsed
                if now - (guest.last_active_at or 0) > 86400:
                    guest.request_count = 0  # type: ignore

                guest.last_active_at = now  # type: ignore
                guest.ip_address = clean_ip  # type: ignore

                # 1. Check direct client device quota
                if guest.request_count >= USER_DAILY_REQUEST_LIMIT:
                    quota = {
                        "is_guest": True,
                        "guest_limit": USER_DAILY_REQUEST_LIMIT,
                        "daily_limit": USER_DAILY_REQUEST_LIMIT,
                        "requests_used": guest.request_count,
                        "requests_remaining": 0,
                        "can_request": False,
                        "requires_email": False,
                    }
                    msg = (
                        f"Daily query limit of {USER_DAILY_REQUEST_LIMIT} requests reached for this device. "
                        "Quota resets at midnight UTC."
                    )
                    return False, msg, quota

                # 2. Check IP-wide quota across multiple client_ids (anti-spoofing / anti-tampering)
                # Ignore public/common proxy loops on localhost
                if clean_ip not in ["127.0.0.1", "localhost", "::1"]:
                    day_ago = now - 86400
                    ip_usage = (
                        session.query(GuestUsageEntity)
                        .filter(
                            GuestUsageEntity.ip_address == clean_ip,
                            GuestUsageEntity.last_active_at >= day_ago,
                        )
                        .all()
                    )
                    total_ip_requests = sum(g.request_count for g in ip_usage)
                    if total_ip_requests >= USER_DAILY_REQUEST_LIMIT * 2:  # Generous threshold for shared NAT
                        quota = {
                            "is_guest": True,
                            "guest_limit": USER_DAILY_REQUEST_LIMIT,
                            "daily_limit": USER_DAILY_REQUEST_LIMIT,
                            "requests_used": guest.request_count,
                            "requests_remaining": 0,
                            "can_request": False,
                            "requires_email": False,
                        }
                        msg = (
                            f"Daily query limit reached for IP network '{clean_ip}'. "
                            "Quota resets at midnight UTC."
                        )
                        return False, msg, quota

                # Consume device request
                guest.request_count += 1  # type: ignore
                session.flush()

                quota = {
                    "is_guest": True,
                    "guest_limit": USER_DAILY_REQUEST_LIMIT,
                    "daily_limit": USER_DAILY_REQUEST_LIMIT,
                    "requests_used": guest.request_count,
                    "requests_remaining": max(0, USER_DAILY_REQUEST_LIMIT - guest.request_count),
                    "can_request": True,
                    "requires_email": False,
                }
                return True, "", quota
        except Exception as e:
            logger.error(f"UserService: Error checking quota for email='{email}', client='{client_id}': {e}", exc_info=True)
            logger.error(f"UserService: Error checking quota for email='{email}', client='{client_id}', ip='{ip_address}': {e}", exc_info=True)
            raise

    def get_quota_status(
        self,
        email: Optional[str] = None,
        client_id: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Returns current quota details without consuming a request."""
        try:
            today_str = self.get_current_date_str()
            now = time.time()

            guest_key = (client_id or "anonymous_guest").strip()
            with get_db_session() as session:
                guest = session.query(GuestUsageEntity).filter(GuestUsageEntity.client_id == guest_key).first()
                used = guest.request_count if guest else 0
                if guest and (now - (guest.last_active_at or 0) > 86400):
                    used = 0
                return {
                    "is_guest": True,
                    "guest_limit": USER_DAILY_REQUEST_LIMIT,
                    "daily_limit": USER_DAILY_REQUEST_LIMIT,
                    "requests_used": used,
                    "requests_remaining": max(0, USER_DAILY_REQUEST_LIMIT - used),
                    "remaining": max(0, USER_DAILY_REQUEST_LIMIT - used),
                    "can_request": used < USER_DAILY_REQUEST_LIMIT,
                    "requires_email": False,
                }
        except Exception as e:
            logger.error(f"UserService: Error retrieving quota status for email='{email}', client='{client_id}': {e}", exc_info=True)
            raise
            logger.error(f"UserService: Error retrieving quota status: {e}", exc_info=True)
            return {
                "is_guest": True,
                "guest_limit": USER_DAILY_REQUEST_LIMIT,
                "daily_limit": USER_DAILY_REQUEST_LIMIT,
                "requests_used": 0,
                "requests_remaining": USER_DAILY_REQUEST_LIMIT,
                "can_request": True,
                "requires_email": False,
            }


user_service = UserService()
