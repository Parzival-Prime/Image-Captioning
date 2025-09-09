
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import json
from supporter import ImageCaptioningModel, VOCAB_SIZE, device, model_dict, generate_caption

app = FastAPI()
port = int(os.environ.get("PORT", 8000))

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

with open("vocabulary.json", "r") as file:
  vocabulary = json.load(file)
itos = vocabulary['idx2word']
stoi = vocabulary['word2idx']

model = ImageCaptioningModel(vocab_size=VOCAB_SIZE).to(device)
model.load_state_dict(model_dict['model_state_dict'])


@app.get('/')
def home():
    return JSONResponse(status_code=200, content="Okay! API is working.")

@app.get('/healthz')
def health():
    return JSONResponse(status_code=200, content="working!")

@app.post('/generate-caption')
async def generator(file: UploadFile = File(...)):
    try:
        image = await file.read()
        caption = generate_caption(model, image, itos)
        return JSONResponse(status_code=201, content={"caption": caption})
    except HTTPException as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__=="__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=port)