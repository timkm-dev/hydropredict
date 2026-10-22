"""Google authorization-code flow with PKCE and verified OpenID identity tokens."""

from __future__ import annotations

import base64
import hashlib
import secrets
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode, urlsplit

import requests
from google.auth.exceptions import GoogleAuthError
from google.auth.transport.requests import Request
from google.oauth2 import id_token


@dataclass(frozen=True)
class GoogleLogin:
    client_id: str
    client_secret: str
    redirect_uri: str

    def __post_init__(self) -> None:
        uri = urlsplit(self.redirect_uri)
        if (
            uri.hostname not in {"127.0.0.1", "localhost"}
            or uri.scheme != "http"
            or uri.path != "/api/auth/google/callback"
            or uri.query
            or uri.fragment
        ):
            raise ValueError("Google redirect URI must use the local dashboard callback.")

    def authorize(self, state: str, nonce: str, verifier: str) -> str:
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        return "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(
            {
                "client_id": self.client_id,
                "redirect_uri": self.redirect_uri,
                "response_type": "code",
                "scope": "openid email profile",
                "state": state,
                "nonce": nonce,
                "code_challenge": challenge.rstrip(b"=").decode(),
                "code_challenge_method": "S256",
                "prompt": "select_account",
            }
        )

    def exchange(self, code: str, verifier: str, nonce: str) -> dict[str, Any]:
        try:
            response = requests.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "code": code,
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "redirect_uri": self.redirect_uri,
                    "grant_type": "authorization_code",
                    "code_verifier": verifier,
                },
                timeout=15,
            )
            response.raise_for_status()
            token = response.json().get("id_token")
            if not isinstance(token, str):
                raise ValueError("Missing identity token.")
            # The library verifies Google's signature, issuer, audience and timestamps.
            claims: dict[str, Any] = id_token.verify_oauth2_token(  # type: ignore[no-untyped-call]
                token, Request(), self.client_id
            )
            returned_nonce = claims.get("nonce")
            if not isinstance(returned_nonce, str) or not secrets.compare_digest(
                returned_nonce, nonce
            ):
                raise ValueError("Invalid nonce.")
            return claims
        except (requests.RequestException, GoogleAuthError, ValueError, KeyError) as error:
            raise ValueError("Google sign-in could not be verified. Please try again.") from error
