from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

from src.services.llm.builder import QuestionBuilder

app = FastAPI(title="Pdf Question API")

class PdfResponse(BaseModel):
    answer: str

class PdfRequest(BaseModel):
    question: str

@app.post("/pdf/ask", response_model=PdfResponse)
async def ask_question_async(request: PdfRequest):
    try:
        print(request)
        question = request.question
        builder = QuestionBuilder()
        resp = builder.build(question)
        return PdfResponse(answer=resp)

    except requests.RequestException as e:
        # Handle network or request errors
        raise HTTPException(status_code=500, detail=f"Error calling external API: {str(e)}")
    except ValueError as e:
        # Handle JSON parsing errors
        raise HTTPException(status_code=500, detail=f"Invalid response format: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Welcome to the PFG API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3557)