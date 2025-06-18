import React, { useContext, useState } from 'react';
import { ChatContext } from './ChatContext';
import './ChatWidget.css';

function ChatWidget() {
  const { isOpen, toggleChat } = useContext(ChatContext);
  const [messages, setMessages] = useState([
    { text: 'Hi! I am here to help you with insurance today.', sender: 'bot' },
  ]);
  const [currentMessage, setCurrentMessage] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [phoneNumber, setPhoneNumber] = useState('');
  const [isPhoneNumberSet, setIsPhoneNumberSet] = useState(false);
  const [currentField, setCurrentField] = useState(null);
  const [pendingUserInfo, setPendingUserInfo] = useState([]); // Queue of user info questions
  const [collectingUserInfo, setCollectingUserInfo] = useState(false);

  const closeChat = async () => {
    try {
      if (isPhoneNumberSet) {
        await fetch('http://localhost:8000/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: 'exit', phone_number: phoneNumber }),
        });
      }
      setMessages([
        { text: 'Hi! How can I help you with insurance today?', sender: 'bot' },
      ]);
      setCurrentMessage('');
      setSelectedFile(null);
      setPhoneNumber('');
      setIsPhoneNumberSet(false);
      setCurrentField(null);
      setPendingUserInfo([]);
      setCollectingUserInfo(false);
      toggleChat();
    } catch (error) {
      console.error('Error closing chat:', error);
    }
  };

  const initializeUser = async (phoneNum) => {
    try {
      const response = await fetch(`http://localhost:8000/initialize_user/${phoneNum}`, {
        method: 'POST',
      });
      const data = await response.json();
      if (data.missing_fields && data.missing_fields.length > 0) {
        // Queue all missing fields
        setPendingUserInfo(data.missing_fields);
        setCollectingUserInfo(true);
        // Show the first question as a bot message
        setMessages(prev => [...prev, { text: data.missing_fields[0].question, sender: 'bot' }]);
        setCurrentField(data.missing_fields[0]);
      } else {
        setPendingUserInfo([]);
        setCollectingUserInfo(false);
        setCurrentField(null);
      }
      return true;
    } catch (error) {
      setMessages(prev => [...prev, {
        text: "Sorry, there was an error setting up your chat. Please try again.",
        sender: 'bot'
      }]);
      return false;
    }
  };

  const handlePhoneNumberSubmit = async (e) => {
    e.preventDefault();
    if (phoneNumber.trim()) {
      const success = await initializeUser(phoneNumber);
      if (success) {
        setIsPhoneNumberSet(true);
      }
    }
  };

  // Unified send handler for both user info and normal chat
  const handleSendMessage = async () => {
    if (!currentMessage.trim()) return;
    const userMsg = { text: currentMessage, sender: 'user' };
    setMessages(prev => [...prev, userMsg]);
    setCurrentMessage('');
    if (collectingUserInfo && currentField) {
      // Send user info answer
      try {
        const response = await fetch('http://localhost:8000/update_user_info', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            phone_number: phoneNumber,
            field: currentField.field,
            value: currentMessage
          }),
        });
        const data = await response.json();
        const remainingFields = data.missing_fields || [];
        if (remainingFields.length > 0) {
          setPendingUserInfo(remainingFields);
          setCurrentField(remainingFields[0]);
          setMessages(prev => [...prev, { text: remainingFields[0].question, sender: 'bot' }]);
        } else {
          setPendingUserInfo([]);
          setCurrentField(null);
          setCollectingUserInfo(false);
          // Now send the user's original message to the chat endpoint
          // Use the last user message in the messages array
          const lastUserMsg = userMsg.text;
          try {
            const chatResponse = await fetch('http://localhost:8000/chat', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ message: lastUserMsg, phone_number: phoneNumber }),
            });
            const chatData = await chatResponse.json();
            if (chatData.missing_fields && chatData.missing_fields.length > 0) {
              setPendingUserInfo(chatData.missing_fields);
              setCollectingUserInfo(true);
              setCurrentField(chatData.missing_fields[0]);
              setMessages(prev => [...prev, { text: chatData.missing_fields[0].question, sender: 'bot' }]);
            } else {
              setMessages(prev => [...prev, { text: chatData.response, sender: 'bot' }]);
            }
          } catch (error) {
            setMessages(prev => [...prev, {
              text: 'Sorry, there was an error connecting to the chatbot.',
              sender: 'bot'
            }]);
          }
        }
      } catch (error) {
        setMessages(prev => [...prev, { text: 'Error updating info. Try again.', sender: 'bot' }]);
      }
    } else {
      // Normal chat
      try {
        const response = await fetch('http://localhost:8000/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: userMsg.text, phone_number: phoneNumber }),
        });
        const data = await response.json();
        if (data.missing_fields && data.missing_fields.length > 0) {
          setPendingUserInfo(data.missing_fields);
          setCollectingUserInfo(true);
          setCurrentField(data.missing_fields[0]);
          setMessages(prev => [...prev, { text: data.missing_fields[0].question, sender: 'bot' }]);
        } else {
          setMessages(prev => [...prev, { text: data.response, sender: 'bot' }]);
        }
      } catch (error) {
        setMessages(prev => [...prev, {
          text: 'Sorry, there was an error connecting to the chatbot.',
          sender: 'bot'
        }]);
      }
    }
  };

  return (
    <div className="chat-widget">
      <button className="chat-button" onClick={toggleChat}>
        💬
      </button>
      {isOpen && (
        <div className="chat-window">
          <div className="chat-header">
            <span className="chat-title">Cogno AI Insurance Advisor</span>
            <button className="close-button" onClick={closeChat}>×</button>
          </div>
          <div className="chat-body">
            {!isPhoneNumberSet ? (
              <form onSubmit={handlePhoneNumberSubmit} className="phone-form">
                <input
                  className="phone-input"
                  type="tel"
                  placeholder="Enter your phone number"
                  value={phoneNumber}
                  onChange={(e) => setPhoneNumber(e.target.value)}
                  required
                  autoFocus
                />
                <button className="start-chat-button" type="submit">Start Chat</button>
              </form>
            ) : (
              <>
                {messages
                  .filter((message) => {
                    const hiddenBotMessages = [
                      'Error updating info. Try again.',
                      'Sorry, there was an error setting up your chat. Please try again.',
                      'Sorry, there was an error connecting to the chatbot.'
                    ];
                    if (hiddenBotMessages.includes(message.text)) return false;
                    if (!message.text || !message.text.trim()) return false;
                    return true;
                  })
                  .map((message, index) => (
                    <div key={index} className={`message-row ${message.sender === 'user' ? 'user-row' : 'bot-row'}`}>
                      {message.sender === 'bot' && (
                        <div className="avatar bot-avatar">🤖</div>
                      )}
                      <div className={`message ${message.sender}-message`}>
                        {message.text}
                      </div>
                      {message.sender === 'user' && (
                        <div className="avatar user-avatar">🧑</div>
                      )}
                    </div>
                  ))}
              </>
            )}
          </div>
          {isPhoneNumberSet && (
            <div className="chat-footer">
              <input
                type="text"
                className="footer-input"
                placeholder="Type your message..."
                value={currentMessage}
                onChange={(e) => setCurrentMessage(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    handleSendMessage();
                  }
                }}
                autoFocus
                disabled={collectingUserInfo && !currentField}
              />
              <button className="footer-send-btn" onClick={handleSendMessage} disabled={collectingUserInfo && !currentField}>Send</button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default ChatWidget;
