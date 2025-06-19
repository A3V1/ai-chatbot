import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import HeroSection from './components/HeroSection';
import FeaturesSection from './components/FeaturesSection';
import HowItWorks from './components/HowItWorks';
import ChatWidget from './components/ChatWidget';
import PaymentPage from './components/PaymentPage';

import Tagline from './components/Tagline';
import './App.css';

window.addEventListener('load', () => {
  localStorage.clear();
});

function App() {
  return (
    <Router>
      <div className="App">
        <Header />
        <Routes>
          <Route path="/" element={
            <>
              <HeroSection />
              <FeaturesSection />
              <HowItWorks />
              <Tagline />
              <ChatWidget />
            </>
          } />
          <Route path="/payment" element={<PaymentPage />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
