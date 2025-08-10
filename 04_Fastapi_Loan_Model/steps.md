
pip install -r requirements.txt

Train the ML Model.
python train.py

Start API Server
python main.py

Check logs that server is started on port 8000

Open postman, create new request

URL:  http://localhost:8000/predict-loan
Type: Post
Body:  {
            "income" : 400000,
            "credit_score": 750,
            "loan_amount" : 457675,
            "employment_years": 5
}

Change body type using radio buttons on top to Raw and select JSON in the dropdown on right.

Then send the request to server.

You will get response like : {
    "approved": true,
    "message": "Loan approved",
    "confidence": 1.0
}