import torch
from fastapi import FastAPI, UploadFile, File
from io import BytesIO
from lavis.models import load_model_and_preprocess
from llama_cpp import Llama
from PIL import Image

# Load BLIP and starling models as global state
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="large_coco", is_eval=True, device=device)
llm = Llama(
      model_path="../../llama.cpp/models/starling-lm-7b-alpha.Q5_K_M.gguf",
      n_gpu_layers=40,
      n_ctx=4096,
)

# LLM Prompt for producing David Attenborough sentences
prompt = "GPT4 User: Rephrase the following description into a single sentence as if it were a narration out of one of david attenborough nature documentary. Keep it brief. Only provide a single and most importantly short sentence. CONTEXT: XXX<|end_of_turn|>GPT4 Assistant:"

app = FastAPI()
app.context = []

@app.get("/process-image/")
async def process_image(image_file: UploadFile = File(...)):
    # Read the image and process it for the model
    image_content = await image_file.read()
    image = Image.open(BytesIO(image_content))
    image = vis_processors["eval"](image).unsqueeze(0).to(device)

    # Execute BLIP and retrieve image caption
    description = model.generate({"image": image})
    joined_description = " ".join(description)

    # Add context if we have any
    temp_prompt = ""
    if len(app.context) != 0:
        use_context = " Try to build upon the following documentation context:\n\n " + " ".join(app.context) + " "
        temp_prompt = prompt.replace("CONTEXT", use_context)
    else:
        temp_prompt = prompt.replace("CONTEXT", "")

    print(temp_prompt)

    # Place the image caption inside the LLM prompt
    ready_prompt = temp_prompt.replace("XXX", joined_description)
        
    # Execute llama.cpp with loaded model
    output = llm(
        ready_prompt,
        max_tokens=3000,
        echo=True
    )
    # Retrieve wanted text output
    answer = output['choices'][0]['text']

    # Find the starting index of the duplicate part
    duplicate_index = answer.find("GPT4 Assistant: ")
    # Remove the duplicate part
    sentence = answer[duplicate_index + len("GPT4 Assistant: "):]

    app.context.append(sentence)
    if len(app.context) > 10:
        app.context = app.context[1:]
    
    return sentence
