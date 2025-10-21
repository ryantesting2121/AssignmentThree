API ENdpoints description:
This API serves a sentiment analysis model built with FastAPI, allowing users to analyze the sentiment of movie reviews. It includes endpoints for health checks, sentiment prediction (with and without probability), and retrieving sample reviews from the training dataset.

BUILDING INSTRUCTIONS:

1.
- Do this step only if needed on your machine please
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

2.
.\.venv\Scripts\Activate.ps1  

3.
cd C:\Users\shawn\OneDrive\Desktop\AssignmenThree

4.
python main.py

5. 
uvicorn main:app --reload


FASTAPI Link:
http://127.0.0.1:8000/docs