import React, { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import './paymentpage.css';

function PaymentPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const [selectedPlan, setSelectedPlan] = useState('');
  const [amount, setAmount] = useState('');
  const [loading, setLoading] = useState(true);
  const [debugInfo, setDebugInfo] = useState({});
  const [usedPhone, setUsedPhone] = useState('');

  useEffect(() => {
    let phone_number = localStorage.getItem('phone_number');
    setUsedPhone(phone_number || (location.state && location.state.phone_number) || '');
    console.debug('PaymentPage: location.state', location.state); // Debug: log location.state
    let timeoutId = setTimeout(() => {
      setLoading(false);
      setDebugInfo(prev => ({...prev, timeout: 'Fetch payment details timed out'}));
    }, 10000); // 10 seconds timeout
    if (!phone_number && location.state && location.state.phone_number) {
      phone_number = location.state.phone_number;
      localStorage.setItem('phone_number', phone_number);
    }
    if (!phone_number) {
      alert('No phone number found. Please start chat again.');
      navigate('/');
      return;
    }
    // Fetch selected plan and amount from backend
    fetch(`http://localhost:8000/api/payment-details?phone_number=${phone_number}`)
      .then(res => {
        if (!res.ok) throw new Error('Network response was not ok');
        return res.json();
      })
      .then(data => {
        console.debug('PaymentPage: payment-details response', data); // Debug: log backend response
        // Extra debug: show time and phone number
        console.debug('Fetched at:', new Date().toISOString(), 'for phone:', phone_number);
        setSelectedPlan(data.selected_plan || (location.state && location.state.plan) || '');
        setAmount(data.amount || (location.state && location.state.amount) || '');
        setDebugInfo(data);
        setLoading(false);
        clearTimeout(timeoutId);
      })
      .catch((err) => {
        console.error('PaymentPage: fetch error', err); // Debug: log fetch error
        setDebugInfo({ error: err.message });
        alert('Could not fetch payment details.');
        setLoading(false);
        clearTimeout(timeoutId);
        navigate('/');
      });
    return () => clearTimeout(timeoutId);
  }, [navigate, location.state]);

  if (loading) return <div>Loading payment details...<br/>Debug: {JSON.stringify(debugInfo)}</div>;
  if (!selectedPlan || amount == null || amount === '') {
    return <div>No payment details found. <button onClick={() => navigate('/')}>Go Home</button></div>;
  }

  return (
    <div className="payment-container">
      <h2 className="payment-title"></h2>
      <p className="payment-detail"><b>Selected Plan:</b> {selectedPlan}</p>
      <p className="payment-detail payment-amount"><b>Premium Amount:</b> ₹{amount}</p>
      <button className="pay-now-btn" onClick={() => alert('Payment functionality next step')}>Pay Now</button>
      {/* Debug Info Section
      <div className="payment-debug">
        <b>Debug Info:</b>
        <div><b>Used Phone:</b> {usedPhone}</div>
        <div><b>location.state:</b> {JSON.stringify(location.state)}</div>
        <div><b>Backend Response:</b> {JSON.stringify(debugInfo)}</div>
      </div> */}
    </div>
  );
}

export default PaymentPage;
