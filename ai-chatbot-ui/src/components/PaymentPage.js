import React, { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

function PaymentPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const [plan, setPlan] = useState('');
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
    fetch(`http://localhost:8000/plan_details/${phone_number}`)
      .then(res => res.json())
      .then(data => {
        setPlan(data.plan);
        setAmount(data.amount);
        setLoading(false);
      })
      .catch(() => {
        alert('Could not fetch plan details.');
        navigate('/');
      });
  }, [navigate, location.state]);

  if (loading) return <div>Loading payment details...</div>;
  if (!plan || !amount) {
    return <div>No payment details found. <button onClick={() => navigate('/')}>Go Home</button></div>;
  }

  return (
    <div style={{ padding: 40, textAlign: 'center' }}>
      <h2>Proceed to Payment</h2>
      <p><b>Plan:</b> {plan}</p>
      <p><b>Amount:</b> ₹{amount}</p>
      <button style={{ padding: '10px 30px', fontSize: 18, background: '#c85f44', color: '#fff', border: 'none', borderRadius: 6 }} onClick={() => alert('Payment functionality coming soon!')}>Pay Now</button>
    </div>
  );
}

export default PaymentPage;
