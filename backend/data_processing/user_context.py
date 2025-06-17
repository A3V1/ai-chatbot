from .mysql_connector import get_mysql_connection
import json

def ensure_user_context_row(phone_number: str):
    conn = get_mysql_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT context_id FROM user_context WHERE phone_number = %s", (phone_number,))
    exists = cursor.fetchone()
    if not exists:
        cursor.execute("INSERT INTO user_context (phone_number) VALUES (%s)", (phone_number,))
        conn.commit()
    cursor.close()
    conn.close()

def get_user_context(phone_number: str) -> dict:
    conn = get_mysql_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM user_context WHERE phone_number = %s", (phone_number,))
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    # Parse chat_history JSON if present
    if result and result.get('chat_history'):
        try:
            result['chat_history'] = json.loads(result['chat_history'])
        except Exception:
            result['chat_history'] = []
    return result or {}

def update_user_context(phone_number: str, updates: dict):
    ensure_user_context_row(phone_number)
    conn = get_mysql_connection()
    cursor = conn.cursor()
    for key, value in updates.items():
        if key == 'chat_history' and isinstance(value, list):
            value = json.dumps(value)
        cursor.execute(f"""
            UPDATE user_context SET {key} = %s WHERE phone_number = %s
        """, (value, phone_number))
    conn.commit()
    cursor.close()
    conn.close()

def save_user_data(phone_number: str, context_summary: str, chat_history: list = None):
    # Save context_summary and chat_history in user_context table
    updates = {"context_summary": context_summary}
    if chat_history is not None:
        updates["chat_history"] = chat_history
    update_user_context(phone_number, updates)

def load_user_data(phone_number: str):
    data = get_user_context(phone_number)
    context = data.copy() if data else {}
    chat_history = context.get('chat_history', [])
    return context, chat_history

def get_user_info(phone_number: str) -> dict:
    conn = get_mysql_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM user_info WHERE phone_number = %s", (phone_number,))
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    return result or {}
