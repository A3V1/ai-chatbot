import React, { useContext, useState, useEffect } from 'react';
import { ChatContext } from './ChatContext';
import { useNavigate } from 'react-router-dom';
import './ChatWidget.css';

// Clear localStorage on window load
window.addEventListener('load', () => {
  localStorage.clear();
});

function ChatWidget() {
  const { isOpen, toggleChat } = useContext(ChatContext);
  const navigate = useNavigate();
  const [messages, setMessages] = useState([
    { text: 'Hi! I am here to help you with insurance today.', sender: 'bot' },
  ]);
  const [currentMessage, setCurrentMessage] = useState('');
  const [selectedFile, setSelectedFile] = useState(null); // for file upload
  const [phoneNumber, setPhoneNumber] = useState('');
  const [isPhoneNumberSet, setIsPhoneNumberSet] = useState(false);
  const [currentField, setCurrentField] = useState(null);
  const [pendingUserInfo, setPendingUserInfo] = useState([]); // for onboarding questions
  const [collectingUserInfo, setCollectingUserInfo] = useState(false);
  const [recommendedPlan, setRecommendedPlan] = useState(null); // set when bot recommends a plan
  const [planAmount, setPlanAmount] = useState(null); // set when bot recommends a plan
  const [showPaymentConfirm, setShowPaymentConfirm] = useState(false);
  const [pendingPaymentData, setPendingPaymentData] = useState(null);

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
      setPhoneNumber('');
      setIsPhoneNumberSet(false);
      setCurrentField(null);
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
        setCollectingUserInfo(true);
        // Show the first question as a bot message
        setMessages(prev => [...prev, { text: data.missing_fields[0].question, sender: 'bot' }]);
        setCurrentField(data.missing_fields[0]);
      } else {
        setCurrentField(null);
        setCollectingUserInfo(false);
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
    // List of profile fields allowed in user_info table
    const profileFields = [
      'age',
      'desired_coverage',
      'premium_budget',
      'premium_payment_mode',
      'preferred_add_ons',
      'has_dependents',
      'policy_duration_years',
      'insurance_experience_level'
    ];
    if (collectingUserInfo && currentField) {
      if (profileFields.includes(currentField.field)) {
        // Send user info answer to /update_user_info
        try {
          const response = await fetch('http://localhost:8000/update_user_info', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              phone_number: phoneNumber,
              field: currentField.field,
              value: userMsg.text
            }),
          });
          const data = await response.json();
          const remainingFields = data.missing_fields || [];
          if (remainingFields.length > 0) {
            setCurrentField(remainingFields[0]);
            setMessages(prev => [...prev, { text: remainingFields[0].question, sender: 'bot' }]);
          } else {
            setCurrentField(null);
            setCollectingUserInfo(false);
            // Now send the user's original message to the chat endpoint
            const lastUserMsg = userMsg.text;
            try {
              const chatResponse = await fetch('http://localhost:8000/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: lastUserMsg, phone_number: phoneNumber }),
              });
              const chatData = await chatResponse.json();
              if (chatData.missing_fields && chatData.missing_fields.length > 0) {
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
      } else if (currentField.field === 'interested_policy_type') {
        // Handle interested_policy_type separately
        try {
          const response = await fetch('http://localhost:8000/save_interested_policy_type', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              phone_number: phoneNumber,
              interested_policy_type: userMsg.text
            }),
          });
          const data = await response.json();
          if (data.success) {
            // After saving interested_policy_type, send the original message to chat endpoint
            const chatResponse = await fetch('http://localhost:8000/chat', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ message: userMsg.text, phone_number: phoneNumber }),
            });
            const chatData = await chatResponse.json();
            if (chatData.missing_fields && chatData.missing_fields.length > 0) {
              setCollectingUserInfo(true);
              setCurrentField(chatData.missing_fields[0]);
              setMessages(prev => [...prev, { text: chatData.missing_fields[0].question, sender: 'bot' }]);
            } else {
              setCurrentField(null);
              setCollectingUserInfo(false);
              setMessages(prev => [...prev, { text: chatData.response, sender: 'bot' }]);
            }
          } else {
            setMessages(prev => [...prev, { text: 'Error saving interested policy type. Try again.', sender: 'bot' }]);
          }
        } catch (error) {
          setMessages(prev => [...prev, { text: 'Error saving interested policy type. Try again.', sender: 'bot' }]);
        }
      } else {
        // Policy-specific answer: send to /save_policy_specific_answers
        try {
          const response = await fetch('http://localhost:8000/save_policy_specific_answers', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              phone_number: phoneNumber,
              answers: { [currentField.field]: userMsg.text }
            }),
          });
          // After saving, ask next question or continue chat
          // For simplicity, fetch next missing field from /chat endpoint
          const chatResponse = await fetch('http://localhost:8000/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: '', phone_number: phoneNumber }),
          });
          const chatData = await chatResponse.json();
          if (chatData.missing_fields && chatData.missing_fields.length > 0) {
            setCollectingUserInfo(true);
            setCurrentField(chatData.missing_fields[0]);
            setMessages(prev => [...prev, { text: chatData.missing_fields[0].question, sender: 'bot' }]);
          } else {
            setCurrentField(null);
            setCollectingUserInfo(false);
            setMessages(prev => [...prev, { text: chatData.response, sender: 'bot' }]);
          }
        } catch (error) {
          setMessages(prev => [...prev, { text: 'Error saving policy-specific answer. Try again.', sender: 'bot' }]);
        }
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

  // Helper to handle payment navigation with confirmation
  const handleProceedToPayment = (data) => {
    setPendingPaymentData(data);
    setShowPaymentConfirm(true);
  };

  const confirmPayment = () => {
    setShowPaymentConfirm(false);
    if (pendingPaymentData) {
      // Add a delay to ensure backend has updated selected_plan
      setTimeout(() => {
        navigate('/payment', { state: pendingPaymentData });
        setPendingPaymentData(null);
      }, 700); // 700ms delay
    }
  };

  const cancelPayment = () => {
    setShowPaymentConfirm(false);
    setPendingPaymentData(null);
  };

  // When bot recommends a plan, update recommendedPlan and planAmount
  useEffect(() => {
    const lastBotMsg = messages.filter(m => m.sender === 'bot').slice(-1)[0];
    if (lastBotMsg && lastBotMsg.text) {
      // Try to extract plan name and amount from bot message
      const planMatch = lastBotMsg.text.match(/\*\*(.+?)\*\*/); // e.g. **HomeSecure Essentials**
      if (planMatch) {
        setRecommendedPlan(planMatch[1]);
      }
      // Try to extract amount (₹ or Rs)
      const amtMatch = lastBotMsg.text.match(/(?:₹|Rs\.?)[ ]?([\d,]+)/);
      if (amtMatch) {
        setPlanAmount(parseInt(amtMatch[1].replace(/,/g, '')));
      }
    }
  }, [messages]);

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
                  .map((message, index) => {
                    // Redirect automatically if bot says 'Redirecting to payment page...'
                    if (
                      message.sender === 'bot' &&
                      message.text &&
                      message.text.toLowerCase().includes('redirecting to payment page')
                    ) {
                      setTimeout(() => {
                        handleProceedToPayment({ plan: recommendedPlan, amount: planAmount, phone_number: phoneNumber });
                      }, 500); // 0.5 second delay for UX
                    }
                    // Show payment button if bot says payment is being processed or ready to pay
                    const showPaymentBtn =
                      message.sender === 'bot' &&
                      (
                        message.text.toLowerCase().includes('ready to secure your policy') ||
                        message.text.toLowerCase().includes('ready to make payment?') ||
                        message.text.toLowerCase().includes("type 'yes' to proceed with payment")
                      );
                    return (
                      <div key={index} className={`message-row ${message.sender === 'user' ? 'user-row' : 'bot-row'}`}>
                        {message.sender === 'bot' && (
                          <div className="avatar bot-avatar">🤖</div>
                        )}
                        <div className={`message ${message.sender}-message`}>
                          {message.text}
                          {/* Show Proceed to Payment button after payment confirmation */}
                          {showPaymentBtn && (
                            <button className="footer-send-btn" style={{margin:'10px 0'}} onClick={() => handleProceedToPayment({ plan: recommendedPlan, amount: planAmount, phone_number: phoneNumber })}>
                              Proceed to Payment
                            </button>
                          )}
                          {/* Show Proceed to Payment button right after recommendation */}
                          {message.sender === 'bot' && recommendedPlan && planAmount && message.text.includes(recommendedPlan) && (
                            <button
                              className="footer-send-btn"
                              style={{ margin: '10px 0' }}
                              onClick={() => {
                                // Debug log for payment navigation
                                window.console && console.debug && console.debug('Proceed to Payment clicked:', { plan: recommendedPlan, amount: planAmount, phone_number: phoneNumber });
                                handleProceedToPayment({ plan: recommendedPlan, amount: planAmount, phone_number: phoneNumber });
                              }}
                            >
                              Proceed to Payment
                            </button>
                          )}
                        </div>
                        {message.sender === 'user' && (
                          <div className="avatar user-avatar">🧑</div>
                        )}
                      </div>
                    );
                  })}
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
              {/* File upload button */}
              <input
                type="file"
                id="file-upload"
                style={{ display: 'none' }}
                onChange={e => {
                  const file = e.target.files[0];
                  if (file) {
                    if (file.size > 2 * 1024 * 1024) {
                      alert('File size exceeds 2MB limit.');
                      e.target.value = '';
                      setSelectedFile(null);
                    } else {
                      setSelectedFile(file);
                    }
                  }
                }}
              />
              <button
                className="footer-send-btn"
                type="button"
                onClick={() => document.getElementById('file-upload').click()}
                style={{ marginLeft: 8 }}
              >
                Add File
              </button>
              {selectedFile && (
                <span style={{ marginLeft: 8, fontSize: 12 }}>
                  {selectedFile.name}
                </span>
              )}
            </div>
          )}
          {/* Payment Confirmation Modal */}
          {showPaymentConfirm && (
            <div className="payment-confirm-modal" style={{position:'fixed',top:0,left:0,right:0,bottom:0,background:'rgba(0,0,0,0.4)',display:'flex',alignItems:'center',justifyContent:'center',zIndex:1000}}>
              <div style={{background:'#fff',padding:30,borderRadius:8,minWidth:300,textAlign:'center',boxShadow:'0 2px 12px rgba(0,0,0,0.2)'}}>
                <h3>Confirm Payment</h3>
                <p>You are about to proceed to payment for:</p>
                <p><b>Plan:</b> {pendingPaymentData?.plan || 'N/A'}</p>
                <p><b>Amount:</b> ₹{pendingPaymentData?.amount || 'N/A'}</p>
                <div style={{marginTop:20}}>
                  <button className="footer-send-btn" style={{marginRight:10}} onClick={confirmPayment}>Confirm</button>
                  <button className="footer-send-btn" onClick={cancelPayment}>Cancel</button>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default ChatWidget;
