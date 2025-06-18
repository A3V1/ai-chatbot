# AI Chatbot for Insurance Advisory

## Overview

This project is an end-to-end AI-powered insurance advisory chatbot. It consists of a React-based frontend and a FastAPI backend with a conversational AI engine, user context management, and MySQL database integration. The chatbot helps users choose, compare, and apply for insurance policies, guiding them through the process interactively.

---

## Features

- **Conversational AI**: Natural language chat for insurance queries and recommendations
- **User Context**: Remembers user profile, preferences, and chat history
- **Policy Recommendation**: Suggests policies based on user needs and eligibility
- **Document Retrieval**: Provides brochures, claim process, exclusions, and more
- **Application Guidance**: Guides users through application and payment steps
- **Frontend UI**: Modern React chat widget and landing page
- **Backend API**: FastAPI with endpoints for chat, user info, and initialization
- **Database**: MySQL for user profiles and policy data

---

## Architecture

```
[React Frontend] <-> [FastAPI Backend] <-> [MySQL Database]
```

- **Frontend**: `ai-chatbot-ui/` (React, chat widget, user interface)
- **Backend**: `backend/` (FastAPI, chatbot logic, user info, policy retrieval)
- **Database**: MySQL (user_info, insurance_policies tables)

---

## Setup Instructions

### 1. Clone the Repository

```sh
git clone <repo-url>
cd ai-chatbot
```

### 2. Backend Setup (Python/FastAPI)

- Go to the backend folder:
  ```sh
  cd backend
  ```
- Install dependencies:
  ```sh
  pip install -r requirements.txt
  ```
- Set up environment variables (create a `.env` file):
  ```env
  MYSQL_HOST=localhost
  MYSQL_USER=root
  MYSQL_PASSWORD=yourpassword
  MYSQL_DATABASE=insurance_bot
  OPENAI_API_KEY=your_openai_key
  PINECONE_API_KEY=your_pinecone_key
  PINECONE_ENVIRONMENT=your_pinecone_env
  ```
- Start the FastAPI server:
  ```sh
  uvicorn main:app --reload --host 0.0.0.0 --port 8000
  ```

### 3. Frontend Setup (React)

- Go to the UI folder:
  ```sh
  cd ../ai-chatbot-ui
  ```
- Install dependencies:
  ```sh
  npm install
  ```
- Start the React app:
  ```sh
  npm start
  ```
- The app runs at [http://localhost:3000](http://localhost:3000)

---

## Usage

- Open the React app in your browser.
- Enter your phone number to start a chat session.
- The bot will ask for required info (age, policy type, etc.) and recommend policies.
- You can ask for brochures, claim process, exclusions, or proceed to apply and pay.
- All chat and user info is stored in MySQL for context.

---

## Backend API Endpoints

- `POST /chat` — Main chat endpoint
- `POST /initialize_user/{phone_number}` — Initialize user profile
- `POST /update_user_info` — Update user info fields

---

## Project Structure

```
ai-chatbot/
├── ai-chatbot-ui/         # React frontend
├── backend/               # FastAPI backend
│   ├── main.py            # API entrypoint
│   ├── cbot.py            # Chatbot logic
│   ├── requirements.txt   # Python dependencies
│   └── data_processing/   # Data and DB utilities
└── README.md              # Project documentation
```

---

## Contributing

- Fork the repo and create a feature branch
- Follow code style and add docstrings/comments
- Test your changes locally
- Submit a pull request with a clear description

---

## Troubleshooting

- **CORS errors**: Ensure backend `origins` allow your frontend URL
- **DB errors**: Check MySQL credentials and running status
- **API keys**: Make sure `.env` is set up with valid keys
- **Port conflicts**: Default ports are 3000 (frontend) and 8000 (backend)

---

## License

MIT License

---

## Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain)
- [Pinecone](https://www.pinecone.io/)
- [OpenAI](https://openai.com/)
- [Create React App](https://create-react-app.dev/)

---

## Payment Gateway Integration

To integrate a payment gateway into this project:

### 1. Choose a Payment Gateway
Popular options: Razorpay, Stripe, PayPal, Paytm. For India, Razorpay is common; for global, Stripe or PayPal are good choices.

### 2. Backend Integration (FastAPI)
- Install the SDK for your chosen gateway (e.g., `pip install razorpay`) or use their REST API.
- Create a new API endpoint in your FastAPI backend to initiate a payment. This endpoint should:
  - Receive payment details (amount, user info, etc.)
  - Create a payment order/session using the gateway’s API
  - Return the payment link or session ID to the frontend

**Example (Razorpay):**
```python
import razorpay
from fastapi import APIRouter

router = APIRouter()
razorpay_client = razorpay.Client(auth=("YOUR_KEY_ID", "YOUR_KEY_SECRET"))

@router.post("/create_payment")
def create_payment(amount: int, currency: str = "INR"):
    payment = razorpay_client.order.create({
        "amount": amount * 100,  # Amount in paise
        "currency": currency,
        "payment_capture": 1
    })
    return {"order_id": payment["id"]}
```

### 3. Frontend Integration (React)
- When the user wants to pay, call the backend `/create_payment` endpoint.
- Use the payment gateway’s JavaScript SDK to open the payment modal or redirect the user.
- Handle payment success/failure callbacks and update the backend accordingly.

**Example (Razorpay JS):**
```js
const options = {
  key: "YOUR_KEY_ID",
  amount: order.amount,
  currency: order.currency,
  order_id: order.order_id,
  handler: function (response) {
    // Notify backend of payment success
  }
};
const rzp = new window.Razorpay(options);
rzp.open();
```

### 4. Update Chatbot Flow
- In your chatbot logic, when the user is ready to pay, trigger the payment process and provide the payment link or open the payment modal.
- After payment, confirm the transaction and update the user’s status in your database.

**Summary:**
- Add a payment API in FastAPI backend
- Use the payment gateway’s SDK in both backend and frontend
- Update chatbot flow to handle payment steps

If you want a step-by-step implementation for a specific gateway (e.g., Razorpay or Stripe), see their official docs or request a tailored guide.

---

For questions or support, open an issue or contact the maintainer.
