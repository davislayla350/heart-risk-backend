# import required fastapi classes and functions
from fastapi import FastAPI, Request, Form, HTTPException
# used to send HTML and file responses
from fastapi.responses import HTMLResponse, FileResponse
# serve static files like HTML/CSS/JS
from fastapi.staticfiles import StaticFiles
# allow CORS for frontend-backend interaction
from fastapi.middleware.cors import CORSMiddleware
# for loading ML model
import joblib
# for working with numeric arrays
import numpy as np
# for file path handling
import os

# create the FastAPI app instance
app = FastAPI()

# enable CORS so frontend can call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins
    allow_methods=["*"],  # allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # allow all headers
)

# mount static folder to serve frontend files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# load the ML model when the app starts
try:
    model = joblib.load("model/heart_disease_model.pkl")
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # fallback in case of error

# serve index.html on root URL
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    try:
        file_path = os.path.join("frontend", "index.html")
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="index.html not found")
        return FileResponse(file_path)
    except Exception as e:
        print(f"Error serving index: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# serve info.html when accessed via /info.html
@app.get("/info.html", response_class=HTMLResponse)
async def serve_info():
    try:
        file_path = os.path.join("frontend", "info.html")
        print(f"Looking for info.html at: {os.path.abspath(file_path)}")
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            raise HTTPException(status_code=404, detail="info.html not found")
        print("Serving info.html")
        return FileResponse(file_path)
    except Exception as e:
        print(f"Error serving info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# handle form submission and return prediction
@app.post("/predict")
async def predict(
    Age: int = Form(...),
    Sex: int = Form(...),
    ChestPainType: int = Form(...),
    RestingBP: int = Form(...),
    Cholesterol: int = Form(...),
    FastingBS: int = Form(...),
    RestingECG: int = Form(...),
    MaxHR: int = Form(...),
    ExerciseAngina: int = Form(...),
    Oldpeak: float = Form(...),
    ST_Slope: int = Form(...)
):
    # if model isn't loaded, throw error
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # put all inputs into a numpy array
        features = np.array([[Age, Sex, ChestPainType, RestingBP, Cholesterol,
                              FastingBS, RestingECG, MaxHR, ExerciseAngina,
                              Oldpeak, ST_Slope]])
        
        print("Input features:", features.tolist())
        
        # make prediction using the model
        prediction = model.predict(features)
        # check if model supports probabilities
        prediction_proba = model.predict_proba(features) if hasattr(model, 'predict_proba') else None
        
        print("Raw prediction:", prediction[0])
        if prediction_proba is not None:
            print("Prediction probabilities:", prediction_proba[0])
        
        # interpret the model's result
        result = "High Risk" if prediction[0] == 1 else "Low Risk"
        print("Final result:", result)
        
        # return the result as JSON
        return {"prediction": result}
        
    except Exception as e:
        print(f" Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# debug route to check which frontend files exist
@app.get("/debug")
async def debug_files():
    try:
        # list frontend files if the folder exists
        frontend_files = os.listdir("frontend") if os.path.exists("frontend") else []
        return {
            "frontend_exists": os.path.exists("frontend"),
            "frontend_files": frontend_files,
            "current_directory": os.getcwd(),
            "index_exists": os.path.exists("frontend/index.html"),
            "info_exists": os.path.exists("frontend/info.html")
        }
    except Exception as e:
        return {"error": str(e)}

# run the app if this file is executed directly
if __name__ == "__main__":
    import uvicorn  # import uvicorn server
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)  # start server
