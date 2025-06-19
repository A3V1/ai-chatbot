"""
user_context.py - User Context Management Module
------------------------------------------------
Handles all database operations related to user context and profile for the chatbot:
- Persists and retrieves chat history, conversation state, and context summary
- Manages user profile info and selected plan
- Provides utility functions for conversation state and chat message management
- Ensures robust error handling and logging
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from contextlib import contextmanager

from .mysql_connector import get_mysql_connection

# Configure logging for debugging and error tracking
logger = logging.getLogger(__name__)

# =============================
# Exception Classes
# =============================
class UserContextError(Exception):
    """Custom exception for user context operations (DB, serialization, etc.)"""
    pass

class DatabaseConnectionError(UserContextError):
    """Exception for database connection issues"""
    pass

# =============================
# UserContextManager: Core DB Logic
# =============================
class UserContextManager:
    """
    Manages all user context operations:
    - Ensures user context row exists
    - Loads and updates user context (state, summary, chat history)
    - Handles serialization/deserialization of chat history
    - Manages selected plan and user info
    - Provides robust error handling
    """
    @staticmethod
    @contextmanager
    def get_db_connection(dictionary: bool = False):
        """
        Context manager for DB connections. Yields (conn, cursor) and ensures cleanup.
        Args:
            dictionary: If True, returns results as dicts
        Yields:
            conn, cursor
        Raises:
            DatabaseConnectionError if connection fails
        """
        conn = None
        cursor = None
        try:
            conn = get_mysql_connection()
            cursor = conn.cursor(dictionary=dictionary)
            yield conn, cursor
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            if conn:
                conn.rollback()
            raise DatabaseConnectionError(f"Failed to connect to database: {e}")
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
    
    @staticmethod
    def _serialize_chat_history(chat_history: List[Dict[str, Any]]) -> str:
        """
        Serialize chat history (list of messages) to JSON string for DB storage.
        Raises UserContextError if serialization fails.
        """
        try:
            return json.dumps(chat_history, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to serialize chat history: {e}")
            raise UserContextError(f"Failed to serialize chat history: {e}")
    
    @staticmethod
    def _deserialize_chat_history(chat_history_json: str) -> List[Dict[str, Any]]:
        """
        Deserialize chat history JSON string from DB to list of messages.
        Returns empty list if invalid or empty.
        """
        if not chat_history_json:
            return []
        
        try:
            result = json.loads(chat_history_json)
            return result if isinstance(result, list) else []
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to deserialize chat history: {e}")
            return []
    
    @classmethod
    def ensure_user_context_row(cls, phone_number: str) -> bool:
        """
        Ensure a user_context row exists for the given phone number. Creates if missing.
        Returns True if exists or created.
        """
        if not phone_number or not phone_number.strip():
            raise ValueError("Phone number cannot be empty")
        
        try:
            with cls.get_db_connection() as (conn, cursor):
                # Check if row exists
                cursor.execute(
                    "SELECT context_id FROM user_context WHERE phone_number = %s",
                    (phone_number,)
                )
                exists = cursor.fetchone()
                
                if not exists:
                    # Create new row
                    cursor.execute(
                        "INSERT INTO user_context (phone_number) VALUES (%s)",
                        (phone_number,)
                    )
                    conn.commit()
                    logger.info(f"Created user context row for phone: {phone_number}")
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to ensure user context row for {phone_number}: {e}")
            raise UserContextError(f"Failed to ensure user context row: {e}")
    
    @classmethod
    def get_user_context(cls, phone_number: str) -> Dict[str, Any]:
        """
        Retrieve the full user context (state, summary, chat history, etc.) from DB.
        Returns dict of all columns for the user.
        """
        if not phone_number or not phone_number.strip():
            raise ValueError("Phone number cannot be empty")
        
        try:
            with cls.get_db_connection(dictionary=True) as (conn, cursor):
                cursor.execute(
                    "SELECT * FROM user_context WHERE phone_number = %s",
                    (phone_number,)
                )
                result = cursor.fetchone()
                
                if not result:
                    logger.debug(f"No context found for phone: {phone_number}")
                    return {}
                
                # Parse chat_history JSON if present
                if result.get('chat_history'):
                    result['chat_history'] = cls._deserialize_chat_history(
                        result['chat_history']
                    )
                
                logger.debug(f"Retrieved context for phone: {phone_number}")
                return result
                
        except Exception as e:
            logger.error(f"Failed to get user context for {phone_number}: {e}")
            raise UserContextError(f"Failed to retrieve user context: {e}")
    
    @classmethod
    def update_user_context(cls, phone_number: str, updates: Dict[str, Any]) -> bool:
        """
        Update fields in user_context for the user. Handles chat_history serialization.
        Returns True if update successful.
        """
        if not phone_number or not phone_number.strip():
            raise ValueError("Phone number cannot be empty")
        
        if not updates:
            logger.warning("No updates provided")
            return True
        
        try:
            # Ensure row exists
            cls.ensure_user_context_row(phone_number)
            
            with cls.get_db_connection() as (conn, cursor):
                for key, value in updates.items():
                    # Validate column name to prevent SQL injection
                    if not key.replace('_', '').isalnum():
                        raise ValueError(f"Invalid column name: {key}")
                    
                    # Handle special serialization for chat_history
                    if key == 'chat_history' and isinstance(value, list):
                        value = cls._serialize_chat_history(value)
                    
                    # Use parameterized query to prevent SQL injection
                    query = f"UPDATE user_context SET {key} = %s WHERE phone_number = %s"
                    cursor.execute(query, (value, phone_number))
                
                conn.commit()
                logger.info(f"Updated context for phone: {phone_number}, fields: {list(updates.keys())}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to update user context for {phone_number}: {e}")
            raise UserContextError(f"Failed to update user context: {e}")
    
    @classmethod
    def save_user_data(
        cls,
        phone_number: str,
        context_summary: str,
        chat_history: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        Save context summary and chat history for the user.
        Returns True if save successful.
        """
        if not phone_number or not phone_number.strip():
            raise ValueError("Phone number cannot be empty")
        
        try:
            updates = {"context_summary": context_summary}
            if chat_history is not None:
                updates["chat_history"] = chat_history
            
            return cls.update_user_context(phone_number, updates)
            
        except Exception as e:
            logger.error(f"Failed to save user data for {phone_number}: {e}")
            raise UserContextError(f"Failed to save user data: {e}")
    
    @classmethod
    def load_user_data(cls, phone_number: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Load user context and chat history for the user.
        Returns (context_dict, chat_history_list).
        """
        if not phone_number or not phone_number.strip():
            raise ValueError("Phone number cannot be empty")
        
        try:
            data = cls.get_user_context(phone_number)
            context = data.copy() if data else {}
            chat_history = context.get('chat_history', [])
            
            # Ensure chat_history is always a list
            if not isinstance(chat_history, list):
                chat_history = []
                logger.warning(f"Invalid chat history format for {phone_number}, defaulting to empty list")
            
            logger.debug(f"Loaded user data for phone: {phone_number}")
            return context, chat_history
            
        except Exception as e:
            logger.error(f"Failed to load user data for {phone_number}: {e}")
            raise UserContextError(f"Failed to load user data: {e}")
    
    @classmethod
    def get_user_info(cls, phone_number: str) -> Dict[str, Any]:
        """
        Retrieve user profile info from user_info table.
        Returns dict of all columns for the user.
        """
        if not phone_number or not phone_number.strip():
            raise ValueError("Phone number cannot be empty")
        
        try:
            with cls.get_db_connection(dictionary=True) as (conn, cursor):
                cursor.execute(
                    "SELECT * FROM user_info WHERE phone_number = %s",
                    (phone_number,)
                )
                result = cursor.fetchone()
                
                logger.debug(f"Retrieved user info for phone: {phone_number}")
                return result or {}
                
        except Exception as e:
            logger.error(f"Failed to get user info for {phone_number}: {e}")
            raise UserContextError(f"Failed to retrieve user info: {e}")
    
    @classmethod
    def set_selected_plan(cls, phone_number: str, plan_id: str) -> bool:
        """
        Store the selected plan for the user in user_context.
        Returns True if successful.
        """
        if not phone_number or not phone_number.strip():
            raise ValueError("Phone number cannot be empty")
        
        if not plan_id or not plan_id.strip():
            raise ValueError("Plan ID cannot be empty")
        
        try:
            result = cls.update_user_context(phone_number, {'selected_plan': plan_id})
            logger.info(f"Set selected plan for {phone_number}: {plan_id}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to set selected plan for {phone_number}: {e}")
            raise UserContextError(f"Failed to set selected plan: {e}")
    
    @classmethod
    def get_selected_plan(cls, phone_number: str) -> Optional[str]:
        """
        Retrieve the selected plan for the user from user_context.
        Returns plan_id or None.
        """
        if not phone_number or not phone_number.strip():
            raise ValueError("Phone number cannot be empty")
        
        try:
            context = cls.get_user_context(phone_number)
            selected_plan = context.get('selected_plan')
            
            logger.debug(f"Retrieved selected plan for {phone_number}: {selected_plan}")
            return selected_plan
            
        except Exception as e:
            logger.error(f"Failed to get selected plan for {phone_number}: {e}")
            raise UserContextError(f"Failed to retrieve selected plan: {e}")
    
    @classmethod
    def clear_user_context(cls, phone_number: str) -> bool:
        """
        Delete the user_context row for the user (for testing or reset).
        Returns True if row deleted.
        """
        if not phone_number or not phone_number.strip():
            raise ValueError("Phone number cannot be empty")
        
        try:
            with cls.get_db_connection() as (conn, cursor):
                cursor.execute(
                    "DELETE FROM user_context WHERE phone_number = %s",
                    (phone_number,)
                )
                conn.commit()
                
                rows_affected = cursor.rowcount
                logger.info(f"Cleared context for phone: {phone_number}, rows affected: {rows_affected}")
                return rows_affected > 0
                
        except Exception as e:
            logger.error(f"Failed to clear user context for {phone_number}: {e}")
            raise UserContextError(f"Failed to clear user context: {e}")


# =============================
# Backward Compatibility: Module-level Functions
# =============================
# These wrappers allow other modules to use the manager easily.
def ensure_user_context_row(phone_number: str) -> bool:
    """Backward compatibility wrapper"""
    return UserContextManager.ensure_user_context_row(phone_number)


def get_user_context(phone_number: str) -> Dict[str, Any]:
    """Backward compatibility wrapper"""
    return UserContextManager.get_user_context(phone_number)


def update_user_context(phone_number: str, updates: Dict[str, Any]) -> bool:
    """Backward compatibility wrapper"""
    return UserContextManager.update_user_context(phone_number, updates)


def save_user_data(
    phone_number: str,
    context_summary: str,
    chat_history: Optional[List[Dict[str, Any]]] = None
) -> bool:
    """Backward compatibility wrapper"""
    return UserContextManager.save_user_data(phone_number, context_summary, chat_history)


def load_user_data(phone_number: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Backward compatibility wrapper"""
    return UserContextManager.load_user_data(phone_number)


def get_user_info(phone_number: str) -> Dict[str, Any]:
    """Backward compatibility wrapper"""
    return UserContextManager.get_user_info(phone_number)


def set_selected_plan(phone_number: str, plan_id: str) -> bool:
    """Backward compatibility wrapper"""
    return UserContextManager.set_selected_plan(phone_number, plan_id)


def get_selected_plan(phone_number: str) -> Optional[str]:
    """Backward compatibility wrapper"""
    return UserContextManager.get_selected_plan(phone_number)


# =============================
# Utility Functions
# =============================
def get_user_conversation_state(phone_number: str) -> Optional[str]:
    """
    Get the current conversation state for a user from user_context.
    Returns state string or None.
    """
    try:
        context = get_user_context(phone_number)
        return context.get('conversation_state')
    except Exception as e:
        logger.error(f"Failed to get conversation state for {phone_number}: {e}")
        return None


def set_user_conversation_state(phone_number: str, state: str) -> bool:
    """
    Set the conversation state for a user in user_context.
    Returns True if successful.
    """
    try:
        return update_user_context(phone_number, {'conversation_state': state})
    except Exception as e:
        logger.error(f"Failed to set conversation state for {phone_number}: {e}")
        return False


def add_chat_message(phone_number: str, message_type: str, content: str) -> bool:
    """
    Add a message to the user's chat history and persist it.
    Returns True if successful.
    """
    try:
        context, chat_history = load_user_data(phone_number)
        
        new_message = {
            'type': message_type,
            'content': content,
            'timestamp': None  # You might want to add timestamp here
        }
        
        chat_history.append(new_message)
        return save_user_data(phone_number, context.get('context_summary', ''), chat_history)
        
    except Exception as e:
        logger.error(f"Failed to add chat message for {phone_number}: {e}")
        return False


def fetch_and_store_selected_plan_for_payment(phone_number: str, context: dict) -> dict:
    """
    Fetch the selected plan from user_context and store it in the provided context dict for payment processing.
    Returns the updated context dict.
    """
    try:
        selected_plan = get_selected_plan(phone_number)
        context['selected_plan'] = selected_plan
        return context
    except Exception as e:
        logger.error(f"Failed to fetch/store selected plan for payment for {phone_number}: {e}")
        context['selected_plan'] = None
        return context