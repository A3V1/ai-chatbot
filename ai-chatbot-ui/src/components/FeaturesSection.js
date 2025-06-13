import React from 'react';

function FeaturesSection() {
  return (
    <section className="features">
      <div className="feature-card">
        <span>🔍</span>
        <h3>Policy Lookup</h3>
        <p>Get answers about your current policy</p>
      </div>
      <div className="feature-card">
        <span>🤖</span>
        <h3>Instant Claims Help</h3>
        <p>Understand claim steps & documents</p>
      </div>
      <div className="feature-card">
        <span>🛡️</span>
        <h3>Plan Comparison</h3>
        <p>Know which plan fits you best</p>
      </div>
      <div className="feature-card">
        <span>💬</span>
        <h3>Real-Time Support</h3>
        <p>Ask anything, anytime</p>
      </div>
    </section>
  );
}

export default FeaturesSection;
