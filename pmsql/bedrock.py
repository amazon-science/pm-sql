"""
Simple LLM client for experiments.

Handles Bedrock authentication and provides a clean interface for calling
Claude models. All config is hardcoded here — edit as needed.

Usage:
    from prompt_helpers import call_llm, call_llm_with_system

    # Simple call
    response = call_llm("What is 2+2?")

    # With system prompt
    response = call_llm_with_system(
        system="You are a SQL expert.",
        user="Write a query to count rows.",
    )
"""

import json
import logging
import os
import random
import time
from typing import Optional

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError, ReadTimeoutError, ConnectTimeoutError

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration — edit these as needed
# ============================================================================

REGION = "us-east-1"

# Errors that should NEVER be retried (client-side mistakes)
NON_RETRYABLE_ERROR_CODES = {
    "ValidationException",
    "AccessDeniedException",
}

# Errors that require re-authentication (get a new client)
REAUTH_ERROR_CODES = {
    "ExpiredTokenException",
    "ExpiredToken",
    "UnrecognizedClientException",
}

# Network-level exceptions that are always retryable
RETRYABLE_EXCEPTIONS = (
    ConnectionError,
    ReadTimeoutError,
    ConnectTimeoutError,
)


def _calculate_backoff(attempt: int, base: float = 1.0, max_delay: float = 30.0) -> float:
    """Calculate delay with exponential backoff and full jitter.

    Formula: random(0, min(max_delay, base * 2^attempt))
    AWS-recommended "full jitter" strategy to prevent thundering-herd retries.
    """
    delay = min(max_delay, base * (2 ** attempt))
    return random.uniform(0, delay)

# Cross-account role ARNs for Bedrock access. These are environment-specific,
# so put your own Bedrock role ARN(s) here, for example:
#
#     ROLE_ARNS = [
#         "arn:aws:iam::<YOUR_ACCOUNT_ID>:role/<YOUR_BEDROCK_ROLE>",
#     ]
#
# Alternatively, set them at runtime via the BEDROCK_ROLE_ARNS environment
# variable (comma-separated). If none are provided, the code falls back to the
# default boto3 credential chain (single account, no cross-account role).
ROLE_ARNS = [a.strip() for a in os.environ.get("BEDROCK_ROLE_ARNS", "").split(",") if a.strip()]

# Models
SONNET_4_6 = "us.anthropic.claude-sonnet-4-6"
OPUS_4_7 = "us.anthropic.claude-opus-4-7"

# Default model for experiments
DEFAULT_MODEL = SONNET_4_6

# ============================================================================
# Client management
# ============================================================================

_client = None


def _assume_role(role_arn: str):
    """Assume a cross-account role and return a bedrock-runtime client."""
    sts = boto3.client("sts", region_name=REGION)
    creds = sts.assume_role(
        RoleArn=role_arn,
        RoleSessionName="Text2SQL4PM-Experiment",
        DurationSeconds=3600,
    )["Credentials"]

    return boto3.client(
        "bedrock-runtime",
        region_name=REGION,
        aws_access_key_id=creds["AccessKeyId"],
        aws_secret_access_key=creds["SecretAccessKey"],
        aws_session_token=creds["SessionToken"],
        config=BotoConfig(
            connect_timeout=10,
            read_timeout=120,
            retries={"max_attempts": 3},
        ),
    )


def get_client():
    """Get a bedrock-runtime client. Tries cross-account roles, falls back to default credentials."""
    global _client
    if _client is not None:
        return _client

    # Try cross-account role assumption (randomize order to spread load)
    shuffled_roles = ROLE_ARNS.copy()
    random.shuffle(shuffled_roles)
    for role_arn in shuffled_roles:
        try:
            _client = _assume_role(role_arn)
            logger.info(f"Bedrock client ready via {role_arn}")
            return _client
        except Exception as e:
            logger.debug(f"Role {role_arn} failed: {e}")
            continue

    # Fallback: use default credentials (works if you have direct Bedrock access)
    try:
        _client = boto3.client(
            "bedrock-runtime",
            region_name=REGION,
            config=BotoConfig(
                connect_timeout=10,
                read_timeout=120,
                retries={"max_attempts": 3},
            ),
        )
        logger.info("Bedrock client ready via default credentials")
        return _client
    except Exception as e:
        raise RuntimeError(f"Cannot create Bedrock client: {e}")


def reset_client():
    """Force re-creation of the client (e.g., after credential refresh)."""
    global _client
    _client = None


# ============================================================================
# LLM call functions
# ============================================================================


def call_llm(
    prompt: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
    temperature: float = 0,
    system: Optional[str] = None,
    max_retries: int = 5,
) -> str:
    """
    Call an LLM and return the text response.

    Retry strategy:
    - Exponential backoff with full jitter
    - Non-retryable errors (ValidationException, AccessDenied) fail immediately
    - Retryable errors (Throttling, ServiceUnavailable) retry up to max_retries
    - Network errors (ConnectionError, ReadTimeout) retry up to max_retries

    Args:
        prompt: The user message.
        model: Model ID (default: Claude Sonnet 4.6).
        max_tokens: Max response tokens.
        temperature: Sampling temperature (0 = deterministic).
        system: Optional system prompt.
        max_retries: Maximum retry attempts (default: 5).

    Returns:
        The model's text response.
    """
    client = get_client()

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ],
    }
    if system:
        body["system"] = system

    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            response = client.invoke_model(modelId=model, body=json.dumps(body))
            result = json.loads(response["body"].read())

            text = result["content"][0]["text"]
            usage = result.get("usage", {})
            logger.debug(
                f"LLM call: model={model}, in={usage.get('input_tokens','?')}, "
                f"out={usage.get('output_tokens','?')}"
            )
            return text

        except ClientError as error:
            error_code = error.response.get("Error", {}).get("Code", "")
            last_exception = error

            # Fail immediately on non-retryable errors
            if error_code in NON_RETRYABLE_ERROR_CODES:
                logger.error(f"Non-retryable error ({error_code}): {error}")
                raise

            # Re-auth errors: reset client and retry with fresh credentials
            if error_code in REAUTH_ERROR_CODES:
                if attempt < max_retries:
                    logger.warning(
                        f"Auth expired ({error_code}), attempt {attempt + 1}/{max_retries + 1}, "
                        f"re-authenticating..."
                    )
                    reset_client()
                    client = get_client()
                    delay = _calculate_backoff(attempt, base=0.5, max_delay=5.0)
                    time.sleep(delay)
                    continue
                raise

            # Retry on other retryable errors (throttling, service unavailable)
            if attempt < max_retries:
                delay = _calculate_backoff(attempt)
                logger.warning(
                    f"Retryable error ({error_code}), attempt {attempt + 1}/{max_retries + 1}, "
                    f"retrying in {delay:.1f}s: {error}"
                )
                time.sleep(delay)
                continue

            logger.error(f"Max retries ({max_retries}) exhausted for error ({error_code}): {error}")
            raise

        except RETRYABLE_EXCEPTIONS as error:
            last_exception = error
            if attempt < max_retries:
                delay = _calculate_backoff(attempt)
                logger.warning(
                    f"Network error ({type(error).__name__}), attempt {attempt + 1}/{max_retries + 1}, "
                    f"retrying in {delay:.1f}s: {error}"
                )
                time.sleep(delay)
                continue

            logger.error(
                f"Max retries ({max_retries}) exhausted for network error "
                f"({type(error).__name__}): {error}"
            )
            raise

    raise last_exception


def call_llm_with_system(
    system: str,
    user: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
    temperature: float = 0,
) -> str:
    """Convenience wrapper: call LLM with a system prompt and user message."""
    return call_llm(prompt=user, model=model, max_tokens=max_tokens,
                    temperature=temperature, system=system)


def call_llm_raw(
    prompt: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
    temperature: float = 0,
    system: Optional[str] = None,
    max_retries: int = 5,
) -> dict:
    """
    Call an LLM and return the full response dict (includes usage, stop_reason, etc).
    Uses the same retry logic as call_llm.
    """
    client = get_client()

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ],
    }
    if system:
        body["system"] = system

    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            response = client.invoke_model(modelId=model, body=json.dumps(body))
            return json.loads(response["body"].read())

        except ClientError as error:
            error_code = error.response.get("Error", {}).get("Code", "")
            last_exception = error

            if error_code in NON_RETRYABLE_ERROR_CODES:
                raise

            if error_code in REAUTH_ERROR_CODES:
                if attempt < max_retries:
                    logger.warning(
                        f"Auth expired ({error_code}), attempt {attempt + 1}/{max_retries + 1}, "
                        f"re-authenticating..."
                    )
                    reset_client()
                    client = get_client()
                    delay = _calculate_backoff(attempt, base=0.5, max_delay=5.0)
                    time.sleep(delay)
                    continue
                raise

            if attempt < max_retries:
                delay = _calculate_backoff(attempt)
                logger.warning(
                    f"Retryable error ({error_code}), attempt {attempt + 1}/{max_retries + 1}, "
                    f"retrying in {delay:.1f}s"
                )
                time.sleep(delay)
                continue
            raise

        except RETRYABLE_EXCEPTIONS as error:
            last_exception = error
            if attempt < max_retries:
                delay = _calculate_backoff(attempt)
                logger.warning(
                    f"Network error ({type(error).__name__}), attempt {attempt + 1}/{max_retries + 1}, "
                    f"retrying in {delay:.1f}s"
                )
                time.sleep(delay)
                continue
            raise

    raise last_exception
