import React from 'react';
import Header from './components/Header';
import HeroSection from './components/HeroSection';
import FeaturesSection from './components/FeaturesSection';
import HowItWorks from './components/HowItWorks';
import ChatWidget from './components/ChatWidget';

import Tagline from './components/Tagline';
import './App.css';

function App() {
  return (
    <div className="App">
      <Header />
      <HeroSection />
      <FeaturesSection />
      <HowItWorks />
      <Tagline />
      <ChatWidget />
      
    </div>
  );
}

export default App;
