import React, { useContext } from 'react';
import { ChatContext } from './ChatContext';

window.addEventListener('load', () => {
  localStorage.clear();
});

function HeroSection() {
  const { toggleChat } = useContext(ChatContext);

  return (
    <section className="hero">
      <div className="hero-content">
        <div className="hero-text">
          <h1>Cogno AI is here for you, your peace of mind, and your future</h1>
          <p>Plan, protect, and stay informed with Cogno AI — your virtual insurance advisor.</p>
          <span className="avatar" onClick={toggleChat}>💬</span>
        </div>
        <div className="hero-image"></div>
      </div>
    </section>
  );
}

export default HeroSection;
