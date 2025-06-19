from fastapi import APIRouter, Body
from pydantic import BaseModel
from typing import Optional

router = APIRouter()

class PaymentRequest(BaseModel):
    phone_number: str
    plan: str
    amount: float

class PaymentResponse(BaseModel):
    status: str
    message: str
    plan: Optional[str] = None
    amount: Optional[float] = None

@router.post("/payment/initiate", response_model=PaymentResponse)
def initiate_payment(payment: PaymentRequest = Body(...)):
    # Here you would integrate with a real payment gateway (e.g., Razorpay, Stripe)
    # For now, just echo back the details for confirmation
    return PaymentResponse(
        status="success",
        message=f"Payment initiated for plan '{payment.plan}' (₹{payment.amount}) for user {payment.phone_number}.",
        plan=payment.plan,
        amount=payment.amount
    )
