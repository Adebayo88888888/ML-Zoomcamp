import pickle
from fastapi import FastAPI, Request # text: ignore
from fastapi.responses import JSONResponse

model_file = 'good_bad_trader_log_reg.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = FastAPI(title="Trader Classification Service")

@app.post("/predict")
async def predict(request: Request):
    trader = await request.json()

    X = dv.transform([trader]) # Transform input for the model
    y_pred = model.predict_proba(X)[0, 1]  # Probability of being a 'good trader'

    result = {
        'good_trader_probability': float(y_pred),
        'is_good_trader': bool(y_pred >= 0.5)
    }

    return JSONResponse(result)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
