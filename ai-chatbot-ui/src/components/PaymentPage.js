import React, { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

function PaymentPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const [selectedPlan, setSelectedPlan] = useState('');
  const [amount, setAmount] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let phone_number = localStorage.getItem('phone_number');
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
        setSelectedPlan(data.selected_plan);
        setAmount(data.amount);
        setLoading(false);
      })
      .catch((err) => {
        alert('Could not fetch payment details.');
        navigate('/');
      });
  }, [navigate, location.state]);

  if (loading) return <div>Loading payment details...</div>;
  if (!selectedPlan || amount === null || amount === undefined) {
    return <div>No payment details found. <button onClick={() => navigate('/')}>Go Home</button></div>;
  }

  return (
    <div style={{ padding: 40, textAlign: 'center' }}>
      <h2>Proceed to Payment</h2>
      <p><b>Selected Plan:</b> {selectedPlan}</p>
      <p><b>Premium Amount:</b> ₹{amount}</p>
      <button style={{ padding: '10px 30px', fontSize: 18, background: '#c85f44', color: '#fff', border: 'none', borderRadius: 6 }} onClick={() => alert('Payment functionality coming soon!')}>Pay Now</button>
    </div>
  );
}

export default PaymentPage;
