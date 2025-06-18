"""
User Context Management Module

This module handles user context data including chat history, conversation state,
and selected plans. It provides a clean interface for database operations
related to user context management.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from contextlib import contextmanager

from .mysql_connector import get_mysql_connection

# Configure logging
logger = logging.getLogger(__name__)


class UserContextError(Exception):
    """Custom exception for user context operations"""
    pass


class DatabaseConnectionError(UserContextError):
    """Exception for database connection issues"""
    pass


class UserContextManager:
    """
    Manages user context operations with improved error handling and structure.
    """
    
    @staticmethod
    @contextmanager
    def get_db_connection(dictionary: bool = False):
        """
        Context manager for database connections with proper cleanup.
        
        Args:
            dictionary: Whether to return results as dictionaries
            
        Yields:
            Database connection and cursor
            
        Raises:
            DatabaseConnectionError: If connection fails
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
        Serialize chat history to JSON string.
        
        Args:
            chat_history: List of chat messages
            
        Returns:
            JSON string representation
            
        Raises:
            UserContextError: If serialization fails
        """
        try:
            return json.dumps(chat_history, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to serialize chat history: {e}")
            raise UserContextError(f"Failed to serialize chat history: {e}")
    
    @staticmethod
    def _deserialize_chat_history(chat_history_json: str) -> List[Dict[str, Any]]:
        """
        Deserialize chat history from JSON string.
        
        Args:
            chat_history_json: JSON string representation
            
        Returns:
            List of chat messages
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
        Ensure user context row exists in database.
        
        Args:
            phone_number: User's phone number
            
        Returns:
            True if row exists or was created successfully
            
        Raises:
            UserContextError: If operation fails
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
        Retrieve user context from database.
        
        Args:
            phone_number: User's phone number
            
        Returns:
            Dictionary containing user context data
            
        Raises:
            UserContextError: If retrieval fails
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
        Update user context with given data.
        
        Args:
            phone_number: User's phone number
            updates: Dictionary of fields to update
            
        Returns:
            True if update was successful
            
        Raises:
            UserContextError: If update fails
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
        Save context summary and chat history.
        
        Args:
            phone_number: User's phone number
            context_summary: Context summary to save
            chat_history: Optional chat history list
            
        Returns:
            True if save was successful
            
        Raises:
            UserContextError: If save fails
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
        Load user context and chat history.
        
        Args:
            phone_number: User's phone number
            
        Returns:
            Tuple of (context_dict, chat_history_list)
            
        Raises:
            UserContextError: If load fails
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
        Retrieve user info from database.
        
        Args:
            phone_number: User's phone number
            
        Returns:
            Dictionary containing user info
            
        Raises:
            UserContextError: If retrieval fails
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
        Store the selected plan for a user.
        
        Args:
            phone_number: User's phone number
            plan_id: ID of the selected plan
            
        Returns:
            True if plan was set successfully
            
        Raises:
            UserContextError: If operation fails
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
        Retrieve the selected plan for a user.
        
        Args:
            phone_number: User's phone number
            
        Returns:
            Selected plan ID or None if not set
            
        Raises:
            UserContextError: If retrieval fails
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
        Clear user context data (useful for testing or user reset).
        
        Args:
            phone_number: User's phone number
            
        Returns:
            True if context was cleared successfully
            
        Raises:
            UserContextError: If operation fails
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


# Backward compatibility - expose functions at module level
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


# Additional utility functions
def get_user_conversation_state(phone_number: str) -> Optional[str]:
    """
    Get the current conversation state for a user.
    
    Args:
        phone_number: User's phone number
        
    Returns:
        Current conversation state or None
    """
    try:
        context = get_user_context(phone_number)
        return context.get('conversation_state')
    except Exception as e:
        logger.error(f"Failed to get conversation state for {phone_number}: {e}")
        return None


def set_user_conversation_state(phone_number: str, state: str) -> bool:
    """
    Set the conversation state for a user.
    
    Args:
        phone_number: User's phone number
        state: New conversation state
        
    Returns:
        True if state was set successfully
    """
    try:
        return update_user_context(phone_number, {'conversation_state': state})
    except Exception as e:
        logger.error(f"Failed to set conversation state for {phone_number}: {e}")
        return False


def add_chat_message(phone_number: str, message_type: str, content: str) -> bool:
    """
    Add a single message to user's chat history.
    
    Args:
        phone_number: User's phone number
        message_type: Type of message ('human' or 'ai')
        content: Message content
        
    Returns:
        True if message was added successfully
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