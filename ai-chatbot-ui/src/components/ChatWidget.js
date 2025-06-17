import React, { useContext, useState } from 'react';
import { ChatContext } from './ChatContext';
import './ChatWidget.css';

function ChatWidget() {
  const { isOpen, toggleChat } = useContext(ChatContext);
  const [messages, setMessages] = useState([
    { text: 'Hi! How can I help you with insurance today?', sender: 'bot' },
  ]);
  const [currentMessage, setCurrentMessage] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [phoneNumber, setPhoneNumber] = useState('');
  const [isPhoneNumberSet, setIsPhoneNumberSet] = useState(false);
  const [currentField, setCurrentField] = useState(null);
  // Track indices of user messages entered during user info input phase
  const [userInfoInputIndices, setUserInfoInputIndices] = useState([]);

  const closeChat = async () => {
    try {
      if (isPhoneNumberSet) {
        // Send exit message to backend
        await fetch('http://localhost:8000/chat', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ 
            message: 'exit',
            phone_number: phoneNumber 
          }),
        });
      }
      
      // Reset all states
      setMessages([
        { text: 'Hi! How can I help you with insurance today?', sender: 'bot' },
      ]);
      setCurrentMessage('');
      setSelectedFile(null);
      setPhoneNumber('');
      setIsPhoneNumberSet(false);
      setCurrentField(null);
      
      // Close the chat window
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
        //  setMessages(prev => [...prev,
        // {
          
        //   sender: 'bot'
        // }
        // ]);
        
        setCurrentField(data.missing_fields[0]);
      } else {
        setCurrentField(null);
      }
      return true;
    } catch (error) {
      console.error('Error initializing user:', error);
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

  // Update handleFieldSubmit to record user message indices
  const handleFieldSubmit = async (value) => {
    if (!currentField) return;
    if (!value.trim()) return; // Prevent empty submissions
    setMessages(prev => {
      const newMessages = [...prev, { text: value, sender: 'user' }];
      setUserInfoInputIndices(indices => [...indices, newMessages.length - 1]);
      return newMessages;
    });
    setCurrentMessage('');
    try {
      const response = await fetch('http://localhost:8000/update_user_info', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          phone_number: phoneNumber,
          field: currentField.field,
          value: value
        }),
      });
      const data = await response.json();
      if (data.success) {
        const remainingFields = data.missing_fields || [];
        setCurrentField(remainingFields[0] || null);
        if (remainingFields.length === 0) {
          setMessages(prev => [...prev, { text: data.response, sender: 'bot' }]);
        }
      }
    } catch (error) {
      setMessages(prev => [...prev, { text: 'Error updating info. Try again.', sender: 'bot' }]);
    }
  };

  const handleSendMessage = async () => {
    if (currentMessage.trim() || selectedFile) {
      const newMessage = { text: currentMessage, sender: 'user' };
      setMessages(prev => [...prev, newMessage]);
      setCurrentMessage('');
      setSelectedFile(null);

      try {
        const response = await fetch('http://localhost:8000/chat', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ 
            message: newMessage.text,
            phone_number: phoneNumber 
          }),
        });
        const data = await response.json();
        
        if (data.missing_fields && data.missing_fields.length > 0) {
          setCurrentField(data.missing_fields[0]);
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
            ) : currentField ? (
              <div className="field-form">
                <p className="field-question">{currentField.question}</p>
                <div className="field-input-row">
                  <input
                    type="text"
                    className="field-input"
                    value={currentMessage}
                    onChange={(e) => setCurrentMessage(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') {
                        handleFieldSubmit(currentMessage);
                        setCurrentMessage('');
                      }
                    }}
                    placeholder="Type your answer here..."
                    autoFocus
                  />
                  <button className="submit-btn" onClick={() => {
                    handleFieldSubmit(currentMessage);
                    setCurrentMessage('');
                  }}>
                    Submit
                  </button>
                </div>
              </div>
            ) : (
              <>
                {messages
                  .filter((message, index) => {
                    // Hide error/info messages
                    const hiddenBotMessages = [
                      'Error updating info. Try again.',
                      'Sorry, there was an error setting up your chat. Please try again.',
                      'Sorry, there was an error connecting to the chatbot.'
                    ];
                    if (hiddenBotMessages.includes(message.text)) return false;
                    // Hide user messages that were entered during user info input phase
                    if (userInfoInputIndices && userInfoInputIndices.includes(index) && message.sender === 'user') return false;
                    // Hide empty or whitespace-only messages
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
          {isPhoneNumberSet && !currentField && (
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
              />
              <button className="footer-send-btn" onClick={handleSendMessage}>Send</button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default ChatWidget;
