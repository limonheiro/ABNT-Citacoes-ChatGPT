from typing import List
from fastapi import FastAPI, Request, UploadFile, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import uvicorn
import argparse
import os

from abntcite import get_pdf, get_resume_text, openai_response

app = FastAPI(root_path=".")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory='templates')

origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://0.0.0.0:8000",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Item(BaseModel):
    file: UploadFile
    tag: str

##############################################
# ------------GET Request Routes--------------
##############################################


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse('home.html', {"request": request})


@app.post("/")
async def article_ref(request: Request, file: UploadFile, tag: str = Form()):
    content = await file.read()
    filename = file.filename

    dir_input = "static/pdf/"
    os.makedirs(dir_input, exist_ok=True)
    new_file = f"{dir_input + filename}"

    with open(new_file, "wb") as f:
        f.write(content)

    text_pdf, metadata = get_pdf(new_file)
    resume_text = get_resume_text(text_pdf, tag)

    citation = openai_response(resume_text, tag)

    print(citation)
    return templates.TemplateResponse('home.html', {"request": request, "citation": citation.choices[0].text})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', default=8000)
    opt = parser.parse_args()

    app_str = 'api:app'  # make the app string equal to whatever the name of this file is
    uvicorn.run(app_str, host=opt.host, port=opt.port, reload=True)