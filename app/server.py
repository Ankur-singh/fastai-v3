from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO
import requests

from fastai import *
from fastai.vision import *

file_id = '1b9LyUGaGazjlDS4ysEiMQJwdiHG98pAK'
export_file_name = 'final.pkl'

classes = ['affenpinscher','afghan_hound', 'african_hunting_dog', 'airedale', 'american_staffordshire_terrier', 'appenzeller', 'australian_terrier',
 'basenji', 'basset', 'beagle', 'bedlington_terrier', 'bernese_mountain_dog', 'black-and-tan_coonhound', 'blenheim_spaniel', 'bloodhound', 'bluetick',
 'border_collie', 'border_terrier', 'borzoi', 'boston_bull', 'bouvier_des_flandres', 'boxer', 'brabancon_griffon', 'briard', 'brittany_spaniel', 'bull_mastiff', 
 'cairn', 'cardigan','chesapeake_bay_retriever', 'chihuahua', 'chow', 'clumber', 'cocker_spaniel', 'collie', 'curly-coated_retriever', 'dandie_dinmont', 'dhole',
 'dingo', 'doberman', 'english_foxhound', 'english_setter', 'english_springer','entlebucher', 'eskimo_dog', 'flat-coated_retriever', 'french_bulldog', 
 'german_shepherd', 'german_short-haired_pointer', 'giant_schnauzer', 'golden_retriever', 'gordon_setter', 'great_dane','great_pyrenees', 'greater_swiss_mountain_dog', 
 'groenendael', 'ibizan_hound', 'irish_setter', 'irish_terrier', 'irish_water_spaniel', 'irish_wolfhound','italian_greyhound', 'japanese_spaniel', 'keeshond', 'kelpie', 
 'kerry_blue_terrier', 'komondor', 'kuvasz', 'labrador_retriever', 'lakeland_terrier', 'leonberg', 'lhasa','malamute', 'malinois', 'maltese_dog', 'mexican_hairless', 
 'miniature_pinscher', 'miniature_poodle','miniature_schnauzer', 'newfoundland', 'norfolk_terrier', 'norwegian_elkhound', 'norwich_terrier', 'old_english_sheepdog', 
 'otterhound', 'papillon','pekinese', 'pembroke', 'pomeranian', 'pug', 'redbone', 'rhodesian_ridgeback', 'rottweiler', 'saint_bernard', 'saluki', 'samoyed', 'schipperke', 
 'scotch_terrier','scottish_deerhound', 'sealyham_terrier', 'shetland_sheepdog', 'shih-tzu', 'siberian_husky', 'silky_terrier', 'soft-coated_wheaten_terrier',
 'staffordshire_bullterrier', 'standard_poodle', 'standard_schnauzer', 'sussex_spaniel', 'tibetan_mastiff', 'tibetan_terrier', 'toy_poodle', 'toy_terrier', 'vizsla', 
 'walker_hound', 'weimaraner', 'welsh_springer_spaniel', 'west_highland_white_terrier', 'whippet', 'wire-haired_fox_terrier','yorkshire_terrier']

path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


def download_file_from_google_drive(id, destination):
    if destination.exists(): 
        return

    session = requests.Session()
    
    URL = "https://docs.google.com/uc?export=download"
    
    response = session.get(URL, params = {'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params= {'id':id, 'confirm': token}
        response = session.get(URL, params= params, stream=True)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def setup_learner():
    download_file_from_google_drive(file_id, path/export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise

learn = setup_learner()

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open(encoding='utf-8').read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    img_bytes = await (data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    return JSONResponse({'result': str(prediction)})

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app=app, host='0.0.0.0', port=5042)
