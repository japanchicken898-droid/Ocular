"""Unified sensitive data detection and telemetry event redaction.

Provides attribute-first, boundary-safe token matching for identifying sensitive
fields (passwords, tokens, credentials, PII) without false-positive matching
on words like 'spinner', 'shipping_address', 'typing', or 'author'.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Set, Union

# Individual single-word tokens matching strictly at token boundaries
SENSITIVE_WORDS: Set[str] = {
    # Passwords
    "password",
    "passwd",
    "pwd",
    "pass",
    "passcode",
    "passphrase",
    # Credentials
    "cred",
    "credential",
    "credentials",
    # Tokens & Auth
    "token",
    "jwt",
    "bearer",
    "auth",
    "access_token",
    "refreshtoken",
    "refresh_token",
    "id_token",
    # Pins
    "pin",
    "pincode",
    "pin_code",
    # Secrets
    "secret",
    "secret_key",
    "client_secret",
    # Seed phrases
    "seed",
    "seed_phrase",
    "seedphrase",
    "mnemonic",
    "recovery_phrase",
    "recoveryphrase",
    "private_key",
    # Session IDs
    "session",
    "sessionid",
    "session_id",
    "jsessionid",
    "phpsessid",
    "aspsessionid",
    "sid",
    # Cookies
    "cookie",
    # Cards
    "card",
    "cardnumber",
    "card_number",
    "creditcard",
    "credit_card",
    "cc_number",
    "debitcard",
    "debit_card",
    "pan",
    # CVVs
    "cvv",
    "cvc",
    "csc",
    "cvv2",
    "cid",
    # SSN / Tax / Identification
    "ssn",
    "social_security",
    "socialsecurity",
    "tax_id",
    "taxid",
    "national_id",
    "sin",
    # Date of Birth
    "dob",
    "birthdate",
    "birth_date",
    "date_of_birth",
    # Banking
    "iban",
    "swift",
    "routing",
    "routing_number",
    "account_number",
    "bank_account",
    # OTP / MFA
    "otp",
    "2fa",
    "mfa",
    "totp",
    "one_time_code",
    "verification_code",
    "verificationcode",
    "security_code",
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
    "client_secret",
    "client-secret",
    "clientsecret",
    "client secret",
    "access_key",
    "access-key",
    "accesskey",
    "access key",
    "access_token",
    "access-token",
    "accesstoken",
    "access token",
    "refresh_token",
    "refresh-token",
    "refreshtoken",
    "refresh token",
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
    "verification_code",
    "verification-code",
    "verificationcode",
    "verification code",
    "security_code",
    "security-code",
    "securitycode",
    "security code",
    "seed_phrase",
    "seed-phrase",
    "seedphrase",
    "seed phrase",
    "recovery_phrase",
    "recovery-phrase",
    "recoveryphrase",
    "recovery phrase",
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
    "debit_card",
    "debit-card",
    "debitcard",
    "debit card",
    "tax_id",
    "tax-id",
    "taxid",
    "tax id",
    "national_id",
    "national-id",
    "nationalid",
    "national id",
    "social_security",
    "social-security",
    "socialsecurity",
    "social security",
    "date_of_birth",
    "date-of-birth",
    "birth_date",
    "birth-date",
    "birthdate",
    "bank_account",
    "bank-account",
    "bankaccount",
    "account_number",
    "account-number",
    "accountnumber",
    "routing_number",
    "routing-number",
    "routingnumber",
}

SAFE_FALSE_POSITIVES: Set[str] = {
    "spinner",
    "shipping",
    "address",
    "shipping_address",
    "shipping-address",
    "typing",
    "author",
    "bypass",
    "pinterest",
    "compass",
    "tokenizer",
}

SENSITIVE_SUFFIXES = (
    "password",
    "passwd",
    "passcode",
    "passphrase",
    "secret",
    "phrase",
    "token",
)

SENSITIVE_AUTOCOMPLETE_PREFIXES = (
    "current-password",
    "new-password",
    "one-time-code",
    "cc-",
)

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

    raw = text.strip()
    if not raw:
        return False

    # Check for direct type=password in selector strings
    lower_raw = raw.lower()
    if 'type="password"' in lower_raw or "type='password'" in lower_raw or "type=password" in lower_raw:
        return True

    # Check exact match in safe false positives
    if lower_raw in SAFE_FALSE_POSITIVES or lower_raw.replace("-", "_") in SAFE_FALSE_POSITIVES:
        return False

    # Normalize camelCase to space-separated words (e.g. verificationCode -> verification Code)
    s_expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", raw)
    s = s_expanded.lower()

    # 1. Match compound patterns with non-alphanumeric boundaries
    for comp in SENSITIVE_COMPOUNDS:
        pattern = r"(^|[^a-z0-9])" + re.escape(comp) + r"($|[^a-z0-9])"
        if re.search(pattern, s):
            return True

    # 2. Match individual tokens and stems
    tokens = re.findall(r"[a-z0-9]+", s)
    # Check if whole tokens match safe false positives entirely
    if len(tokens) == 1 and tokens[0] in SAFE_FALSE_POSITIVES:
        return False
    if len(tokens) == 2 and f"{tokens[0]}_{tokens[1]}" in SAFE_FALSE_POSITIVES:
        return False

    for tok in tokens:
        if tok in SAFE_FALSE_POSITIVES:
            continue

        if tok in SENSITIVE_WORDS:
            return True

        # Strip trailing digits (e.g. password1 -> password, pin2 -> pin, cvv2 -> cvv)
        stem = re.sub(r"\d+$", "", tok)
        if stem and stem != tok:
            if stem in SENSITIVE_WORDS:
                return True

        # Suffix matching for compound words run together (e.g. userpassword, clientsecret, seedphrase)
        if len(tok) >= 6:
            for sfx in SENSITIVE_SUFFIXES:
                if tok.endswith(sfx) and tok not in SAFE_FALSE_POSITIVES:
                    return True

        # Pin suffix matching (e.g. userpin, mypin)
        if len(tok) > 3 and tok.endswith("pin") and not tok.endswith("spin") and not tok.endswith("pine"):
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
        if "password" in autocomplete or any(
            autocomplete.startswith(p) or p in autocomplete
            for p in SENSITIVE_AUTOCOMPLETE_PREFIXES
        ):
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

    # 5. Default-deny for anonymous/unknown free-text inputs:
    # If the element represents an interactive input/textarea without any non-sensitive identifier
    tag = str(attrs.get("target_tag") or attrs.get("tag") or "").lower()
    role = str(attrs.get("role") or "").lower()
    if tag in ("textarea",) or role in ("textbox",):
        # If no identifiable name/id/label was provided or all were empty
        has_id = any(attrs.get(k) for k in ("name", "id", "label", "parameter_ref", "key"))
        if not has_id:
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


def scrub_exception(exc: Union[Exception, str], secrets: Optional[Iterable[str]] = None) -> str:
    """Removes sensitive values, Playwright fill/type arguments, credentials, and tokens from exceptions and logs."""
    msg = str(exc)

    # 1. Scrub passed secrets
    if secrets:
        for s in secrets:
            if s and isinstance(s, str) and len(s) > 1:
                msg = msg.replace(s, "[REDACTED]")

    # 2. Scrub Playwright locator.fill / page.fill / type arguments
    # Matches .fill("secret"), .type("secret"), locator.fill('secret')
    msg = re.sub(
        r'(\b(?:fill|type)\s*\([^,)]+,\s*)(["\'])(.*?)\2(\s*\))',
        r'\1"[REDACTED]"\4',
        msg,
    )
    # Matches locator.fill("secret") where single arg is passed
    msg = re.sub(
        r'(\.\s*(?:fill|type)\s*\(\s*)(["\'])(.*?)\2(\s*\))',
        r'\1"[REDACTED]"\4',
        msg,
    )
    # Also matches call log entries like: - locator.fill("secret") or - fill "secret"
    msg = re.sub(
        r'(-\s*(?:locator\.)?(?:fill|type)\s*\(?)(["\'])(.*?)\2(\)?)',
        r'\1"[REDACTED]"\4',
        msg,
    )

    # 3. Scrub basic auth credentials in URLs: http://user:pass@host -> http://[REDACTED]:[REDACTED]@host
    msg = re.sub(r'(https?://)([^:\s/@]+):([^@\s/]+)@', r'\1[REDACTED]:[REDACTED]@', msg)

    # 4. Scrub token / secret query parameters: ?token=xyz, &secret=abc, etc.
    msg = re.sub(
        r'([?&](?:token|key|secret|password|auth|access_token|api_key)=)[^&\s]+',
        r'\1[REDACTED]',
        msg,
        flags=re.IGNORECASE,
    )

    return msg


def get_client_js_sensitive_check() -> str:
    """Returns JavaScript code for client-side checkSensitiveString ensuring parity with Python."""
    words_json = "[" + ", ".join(f'"{w}"' for w in sorted(SENSITIVE_WORDS)) + "]"
    compounds_json = "[" + ", ".join(f'"{c}"' for c in sorted(SENSITIVE_COMPOUNDS)) + "]"
    safe_json = "[" + ", ".join(f'"{s}"' for s in sorted(SAFE_FALSE_POSITIVES)) + "]"

    return f"""
        const SENSITIVE_WORDS = new Set({words_json});
        const SENSITIVE_COMPOUNDS = {compounds_json};
        const SAFE_FALSE_POSITIVES = new Set({safe_json});
        const SENSITIVE_SUFFIXES = ["password", "passwd", "passcode", "passphrase", "secret", "phrase", "token"];

        function checkSensitiveString(str) {{
            if (!str || typeof str !== 'string') return false;
            const raw = str.trim();
            if (!raw) return false;
            const lowerRaw = raw.toLowerCase();
            if (lowerRaw.indexOf('type="password"') !== -1 || lowerRaw.indexOf("type='password'") !== -1 || lowerRaw.indexOf("type=password") !== -1) return true;
            if (SAFE_FALSE_POSITIVES.has(lowerRaw) || SAFE_FALSE_POSITIVES.has(lowerRaw.replace(/-/g, '_'))) return false;

            const expanded = raw.replace(/([a-z0-9])([A-Z])/g, '$1 $2').toLowerCase();

            for (let i = 0; i < SENSITIVE_COMPOUNDS.length; i++) {{
                const comp = SENSITIVE_COMPOUNDS[i];
                const re = new RegExp('(^|[^a-z0-9])' + comp.replace(/[-_ ]/g, '[-_ ]') + '($|[^a-z0-9])');
                if (re.test(expanded)) return true;
            }}

            const tokens = expanded.match(/[a-z0-9]+/g);
            if (!tokens) return false;
            if (tokens.length === 1 && SAFE_FALSE_POSITIVES.has(tokens[0])) return false;
            if (tokens.length === 2 && SAFE_FALSE_POSITIVES.has(tokens[0] + '_' + tokens[1])) return false;

            for (let i = 0; i < tokens.length; i++) {{
                const tok = tokens[i];
                if (SAFE_FALSE_POSITIVES.has(tok)) continue;
                if (SENSITIVE_WORDS.has(tok)) return true;

                const stem = tok.replace(/\\d+$/, '');
                if (stem && stem !== tok && SENSITIVE_WORDS.has(stem)) return true;

                if (tok.length >= 6) {{
                    for (let j = 0; j < SENSITIVE_SUFFIXES.length; j++) {{
                        if (tok.endsWith(SENSITIVE_SUFFIXES[j]) && !SAFE_FALSE_POSITIVES.has(tok)) return true;
                    }}
                }}

                if (tok.length > 3 && tok.endsWith('pin') && !tok.endsWith('spin') && !tok.endsWith('pine')) return true;
            }}

            return false;
        }}
    """
