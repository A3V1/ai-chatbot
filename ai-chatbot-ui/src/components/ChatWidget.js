import React, { useContext, useState } from 'react';
import { ChatContext } from './ChatContext';
import './ChatWidget.css';

function ChatWidget() {
  const { isOpen, toggleChat } = useContext(ChatContext);
  const [messages, setMessages] = useState([
    { text: 'Hi! I’m Cogno AI. How can I help you with insurance today?', sender: 'bot' },
  ]);
  const [currentMessage, setCurrentMessage] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);

  const handleSendMessage = async () => {
    if (currentMessage.trim() || selectedFile) {
      const newMessage = { text: currentMessage, sender: 'user' };
      setMessages([...messages, newMessage]);
      setCurrentMessage('');
      setSelectedFile(null); // Clear selected file after sending

      // Send message to FastAPI backend
      try {
        const response = await fetch('http://localhost:8000/chat', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ message: newMessage.text }),
        });
        const data = await response.json();
        setMessages((prev) => [...prev, { text: data.response, sender: 'bot' }]);
      } catch (error) {
        setMessages((prev) => [...prev, { text: 'Sorry, there was an error connecting to the chatbot.', sender: 'bot' }]);
      }
    }
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      if (file.size <= 2 * 1024 * 1024) { // 2MB limit
        setSelectedFile(file);
      } else {
        alert('File size exceeds 2MB limit.');
        setSelectedFile(null);
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
            Cogno AI
            <button className="close-button" onClick={toggleChat}>×</button>
          </div>
          <div className="chat-body">
            {messages.map((message, index) => (
              <div key={index} className={`message ${message.sender}`}>
                {message.text}
              </div>
            ))}
            {selectedFile && (
              <div className="message user file-attachment">
                Attached: {selectedFile.name} ({Math.round(selectedFile.size / 1024)} KB)
              </div>
            )}
          </div>
          <div className="chat-footer">
            <input
              type="file"
              id="file-input"
              style={{ display: 'none' }}
              onChange={handleFileChange}
            />
            <button className="attach-button" onClick={() => document.getElementById('file-input').click()}>
              +
            </button>
            <input
              type="text"
              placeholder="Type your message..."
              value={currentMessage}
              onChange={(e) => setCurrentMessage(e.target.value)}
              onKeyPress={(e) => {
                if (e.key === 'Enter') {
                  handleSendMessage();
                }
              }}
            />
            <button onClick={handleSendMessage}>Send</button>
          </div>
        </div>
      )}
    </div>
  );
}

export default ChatWidget;
