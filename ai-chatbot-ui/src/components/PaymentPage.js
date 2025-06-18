import React, { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

function PaymentPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const [plan, setPlan] = useState('');
  const [amount, setAmount] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Dynamically load Razorpay script
    const script = document.createElement('script');
    script.src = 'https://checkout.razorpay.com/v1/checkout.js';
    script.async = true;
    document.body.appendChild(script);
    return () => {
      document.body.removeChild(script);
    };
  }, []);

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

  const handlePayment = async () => {
    try {
      const response = await fetch('http://localhost:8000/create_payment', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ amount: Number(amount) }),
      });
      const data = await response.json();
      if (!data.order_id) {
        alert('Payment initiation failed.');
        return;
      }
      const options = {
        key: 'YOUR_KEY_ID', // Replace with your Razorpay key
        amount: data.amount,
        currency: data.currency,
        order_id: data.order_id,
        handler: function (response) {
          alert('Payment successful! Payment ID: ' + response.razorpay_payment_id);
          navigate('/');
        },
        theme: { color: '#c85f44' },
        modal: {
          ondismiss: function () {
            setLoading(false);
            alert('Payment popup closed.');
          }
        }
      };
      setLoading(true);
      if (window.Razorpay) {
        const rzp = new window.Razorpay(options);
        rzp.open();
      } else {
        alert('Razorpay SDK not loaded. Please try again.');
      }
      setLoading(false);
    } catch (error) {
      setLoading(false);
      alert('Error initiating payment.');
    }
  };

  if (loading) return <div>Loading payment details...</div>;
  if (!plan || !amount) {
    return <div>No payment details found. <button onClick={() => navigate('/')}>Go Home</button></div>;
  }

  return (
    <div style={{ padding: 40, textAlign: 'center' }}>
      <h2>Proceed to Payment</h2>
      <p><b>Plan:</b> {plan}</p>
      <p><b>Amount:</b> ₹{amount}</p>
      <button onClick={handlePayment} style={{ padding: '10px 30px', fontSize: 18, background: '#c85f44', color: '#fff', border: 'none', borderRadius: 6 }}>Pay Now</button>
    </div>
  );
}

export default PaymentPage;
