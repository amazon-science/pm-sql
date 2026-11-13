"""
Simple agent loop using invoke_model directly (avoids Strands converse_stream prefill bug).

This implements a basic tool-calling loop:
1. Send messages to Claude via invoke_model
2. If Claude wants to call a tool, execute it and feed result back
3. Repeat until Claude returns a final text response
4. Extract SQL from <final_sql> tags
"""

import json
import re
import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "event_log.sqlite"


def validate_sql_fn(sql: str) -> str:
    """Execute SQL to check validity. Returns 'VALID: N rows' or 'ERROR: ...'."""
    conn = sqlite3.connect(str(DB_PATH))
    start = time.time()

    def check_timeout():
        return 1 if time.time() - start > 5 else 0

    conn.set_progress_handler(check_timeout, 500)
    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        return f"VALID: {len(rows)} rows returned."
    except sqlite3.OperationalError as e:
        if "interrupt" in str(e).lower():
            return "ERROR: Query timed out (exceeded 5 seconds)."
        return f"ERROR: {e}"
    except Exception as e:
        return f"ERROR: {e}"
    finally:
        conn.set_progress_handler(None, 0)
        conn.close()


# Tool definition for Claude's tool_use format
TOOL_DEFINITION = {
    "name": "validate_sql",
    "description": "Validate a SQL query by executing it against the event_log database. Returns 'VALID: N rows returned.' or 'ERROR: description'.",
    "input_schema": {
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": "A SQLite SQL query to validate."
            }
        },
        "required": ["sql"]
    }
}


def run_agent(client, model_id, system_prompt, question, max_tool_calls=3):
    """Run a simple agent loop. Returns the final text response."""
    messages = [
        {"role": "user", "content": [{"type": "text", "text": question}]}
    ]

    tool_calls_made = 0

    for _ in range(max_tool_calls + 1):  # +1 for the final response
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "temperature": 0,
            "system": system_prompt,
            "messages": messages,
            "tools": [TOOL_DEFINITION],
        }

        response = client.invoke_model(modelId=model_id, body=json.dumps(body))
        result = json.loads(response["body"].read())

        stop_reason = result.get("stop_reason", "end_turn")
        content = result.get("content", [])

        # Add assistant response to messages
        messages.append({"role": "assistant", "content": content})

        # If model wants to use a tool
        if stop_reason == "tool_use":
            tool_results = []
            for block in content:
                if block.get("type") == "tool_use":
                    tool_name = block["name"]
                    tool_input = block["input"]
                    tool_use_id = block["id"]

                    # Execute the tool
                    if tool_name == "validate_sql":
                        tool_output = validate_sql_fn(tool_input.get("sql", ""))
                    else:
                        tool_output = f"ERROR: Unknown tool {tool_name}"

                    tool_calls_made += 1
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": [{"type": "text", "text": tool_output}]
                    })

            # Add tool results as user message
            messages.append({"role": "user", "content": tool_results})

            # Stop if we've hit max tool calls
            if tool_calls_made >= max_tool_calls:
                # One more call without tools to get final answer
                body_final = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 2048,
                    "temperature": 0,
                    "system": system_prompt,
                    "messages": messages,
                }
                response = client.invoke_model(modelId=model_id, body=json.dumps(body_final))
                result = json.loads(response["body"].read())
                content = result.get("content", [])
                return _extract_text(content), tool_calls_made

        else:
            # Model returned final response (end_turn)
            return _extract_text(content), tool_calls_made

    # Shouldn't reach here, but just in case
    return _extract_text(content), tool_calls_made


def _extract_text(content):
    """Extract text from content blocks."""
    texts = []
    for block in content:
        if block.get("type") == "text":
            texts.append(block["text"])
    return " ".join(texts)


def extract_final_sql(agent_response):
    """Extract SQL from <final_sql> tags, with fallback."""
    match = re.search(r'<final_sql>(.*?)</final_sql>', agent_response, re.DOTALL)
    if match:
        sql = match.group(1).strip()
        sql = ' '.join(sql.split())  # collapse whitespace
        sql = sql.split(';')[0].strip()
        return sql

    # Fallback: find SELECT or WITH
    text = agent_response.strip()
    for keyword in ["SELECT", "WITH"]:
        idx = text.upper().find(keyword)
        if idx >= 0:
            sql = text[idx:]
            if "```" in sql:
                sql = sql.split("```")[0]
            sql = sql.split(';')[0].strip()
            sql = ' '.join(sql.split())
            return sql

    return "SELECT 1"  # Last resort
