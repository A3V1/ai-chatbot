import React, { createContext, useState } from 'react';

export const ChatContext = createContext();

export const ChatProvider = ({ children }) => {
  const [isOpen, setIsOpen] = useState(false);

  const toggleChat = () => {
    setIsOpen(!isOpen);
  };

  return (
    <ChatContext.Provider value={{ isOpen, toggleChat }}>
      {children}
    </ChatContext.Provider>
  );
};
