"""Unified sensitive data detection and telemetry event redaction.

Provides attribute-first, boundary-safe token matching for identifying sensitive
fields (passwords, tokens, credentials, PII) without false-positive matching
on words like 'spinner', 'shipping_address', 'typing', or 'author'.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Union

# Individual single-word tokens matching strictly at token boundaries
SENSITIVE_WORDS: Set[str] = {
    "password",
    "passwd",
    "pwd",
    "pass",
    "passcode",
    "passphrase",
    "secret",
    "token",
    "jwt",
    "bearer",
    "session",
    "cookie",
    "otp",
    "2fa",
    "mfa",
    "pin",
    "ssn",
    "cvv",
    "cvc",
    "csc",
    "cardnumber",
    "iban",
    "routing",
}

# Compound tokens (which may include spaces, hyphens, underscores or run together)
SENSITIVE_COMPOUNDS: Set[str] = {
    "api_key",
    "api-key",
    "apikey",
    "api key",
    "private_key",
    "private-key",
    "privatekey",
    "private key",
    "secret_key",
    "secret-key",
    "secretkey",
    "secret key",
    "access_key",
    "access-key",
    "accesskey",
    "access key",
    "auth_key",
    "auth-key",
    "authkey",
    "auth key",
    "one_time",
    "one-time",
    "onetime",
    "one time",
    "one_time_code",
    "one-time-code",
    "onetimecode",
    "one time code",
    "cc_number",
    "cc-number",
    "ccnumber",
    "cc number",
    "card_number",
    "card-number",
    "cardnumber",
    "card number",
    "credit_card",
    "credit-card",
    "creditcard",
    "credit card",
}

SENSITIVE_AUTOCOMPLETE_PREFIXES = ("current-password", "new-password", "one-time-code", "cc-")

EXPLICIT_PARAMETER_ALLOWLIST: Set[str] = {
    "interval",
    "timeframe",
    "search",
    "query",
    "filter",
    "page",
    "limit",
    "offset",
    "sort",
    "order",
    "start_date",
    "end_date",
}


def _matches_sensitive_text(text: Optional[str]) -> bool:
    if not text or not isinstance(text, str):
        return False

    # Check for direct type=password in selector strings
    lower_raw = text.lower()
    if 'type="password"' in lower_raw or "type='password'" in lower_raw or "type=password" in lower_raw:
        return True

    # Normalize camelCase to snake_case (e.g. cardNumber -> card_number, userPin -> user_pin)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text).lower()

    # 1. Match compound patterns with non-alphanumeric boundaries
    for comp in SENSITIVE_COMPOUNDS:
        pattern = r"(^|[^a-z0-9])" + re.escape(comp) + r"($|[^a-z0-9])"
        if re.search(pattern, s):
            return True

    # 2. Match individual tokens
    tokens = re.findall(r"[a-z0-9]+", s)
    for tok in tokens:
        if tok in SENSITIVE_WORDS:
            return True

    return False


def is_sensitive_field(attrs: Union[Dict[str, Any], str, Any]) -> bool:
    """Detects whether a field, attribute dictionary, or selector represents sensitive input."""
    if not attrs:
        return False

    if isinstance(attrs, str):
        return _matches_sensitive_text(attrs)

    if not isinstance(attrs, dict):
        return False

    # 1. Attribute-based first: input type
    input_type = str(attrs.get("type") or "").strip().lower()
    if input_type == "password":
        return True

    # 2. Attribute-based: autocomplete
    autocomplete = str(attrs.get("autocomplete") or "").strip().lower()
    if autocomplete:
        if "password" in autocomplete or any(autocomplete.startswith(p) or p in autocomplete for p in SENSITIVE_AUTOCOMPLETE_PREFIXES):
            return True

    # 3. Attribute-based: inputmode numeric combined with OTP / PIN / Code indications
    inputmode = str(attrs.get("inputmode") or "").strip().lower()
    if inputmode in ("numeric", "tel"):
        for k in ("name", "id", "label", "placeholder", "autocomplete"):
            v = str(attrs.get(k) or "").lower()
            if any(term in v for term in ("otp", "pin", "code", "cvv", "cvc", "card")):
                return True

    # 4. Text attributes inspection
    text_keys = (
        "name",
        "id",
        "label",
        "placeholder",
        "aria_label",
        "aria-label",
        "aria_description",
        "aria-description",
        "data_role",
        "data-role",
        "target_selector",
        "selector",
        "parameter_ref",
        "key",
        "trigger_element",
        "target_anchor",
        "target_tag",
        "class",
        "className",
        "description",
        "suggested_selector",
    )
    for key in text_keys:
        val = attrs.get(key)
        if isinstance(val, str) and _matches_sensitive_text(val):
            return True

    return False


def redact_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """Redacts key, code, value, input_payload, and default_value in sensitive telemetry events.
    
    Replaces string and non-string containers (lists, dicts) wholesale with '[REDACTED]'.
    """
    if not isinstance(event, dict):
        return event

    redacted = dict(event)
    is_sensitive = is_sensitive_field(redacted)

    # Also check if target_selector or target_tag in event indicates sensitive field
    target_sel = str(redacted.get("target_selector") or "")
    target_tag = str(redacted.get("target_tag") or "")
    if not is_sensitive and (is_sensitive_field(target_sel) or is_sensitive_field(target_tag)):
        is_sensitive = True

    if is_sensitive:
        for field in ("key", "code", "value", "input_payload", "default_value"):
            if field in redacted:
                redacted[field] = "[REDACTED]"

        if redacted.get("action_type") in ("fill", "type") or redacted.get("event_type") == "keydown":
            redacted["key"] = "[REDACTED]"
            redacted["code"] = "[REDACTED]"

    return redacted
