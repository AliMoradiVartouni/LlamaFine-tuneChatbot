import os
import torch
from transformers import LlamaForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GenerationConfig
from time import perf_counter

from fastapi import FastAPI, Request, HTTPException
import uvicorn
from pydantic import BaseModel

app = FastAPI()

class LocalRadmanBot:
    def __init__(self):
        # Paths to your local model directory
        output_dir = "/home/ali/moradi/models/Radman-Llama-3.2-3B/extra"

        # Ensure the directory exists
        if not os.path.exists(output_dir):
            raise ValueError(f"The directory does not exist: {output_dir}")

        # Load the tokenizer from the local directory
        self.tokenizer = AutoTokenizer.from_pretrained(output_dir)

        # Load the LlamaForCausalLM model from the local directory
        self.model = LlamaForCausalLM.from_pretrained(output_dir, torch_dtype=torch.float16)
        print("Model loaded successfully")
        # Set the device (either 'cuda' for GPU or 'cpu' for CPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Move model to the correct device (GPU if available)
        self.model = self.model.to(self.device)

    def formatted_prompt(self, question) -> str:
        return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant:"

    def generate_response(self, user_input):
        prompt = self.formatted_prompt(user_input)

        generation_config = GenerationConfig(
            max_new_tokens=2500,
            top_p=0.9,
            penalty_alpha=0.6,
            do_sample=True,
            top_k=5,
            temperature=0.7,
            repetition_penalty=1.2,
            # Remove max_new_tokens from here
            pad_token_id=self.tokenizer.eos_token_id
        )

        start_time = perf_counter()
        inputs = self.tokenizer(prompt, return_tensors="pt").to('cuda')

        outputs = self.model.generate(
            **inputs,
            generation_config=generation_config,
            # max_new_tokens=60,  # Only specify this, not max_length
            no_repeat_ngram_size=3
        )

        decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Improved response cleaning
        # Remove the original question
        if user_input in decoded_output:
            response = decoded_output.split(user_input)[-1].strip()
        else:
            response = decoded_output.strip()

        # Remove any role prefixes and their responses
        roles = ["assistant:", "admin:", "user:", "Bot:", "AI:"]
        for role in roles:
            if role in response:
                response = response.split(role)[-1].strip()

        # # Remove any text after a role appears (to prevent self-conversation)
        # for role in roles:
        #     if role in response:
        #         response = response.split(role)[0].strip()

        output_time = perf_counter() - start_time

        return {
            'response': response,
            'output_time': output_time,
            'device': self.device
        }

    # generate_response(user_input='How can I give invitation letter?')
# Define request and response models
class UserInput(BaseModel):
    message: str

class BotResponse(BaseModel):
    response: str
# Initialize the bot
bot = LocalRadmanBot()
@app.get("/")
def read_root():
    return {"message": "Bot is ready! Type 'quit' to exit. (Note: Use POST /chat for interaction)"}


@app.post("/chat", response_model=BotResponse)
async def main(user_input: UserInput):
    try:
        print(f"Received message: {user_input.message}")  # Debug print
        # Get the bot response
        result = bot.generate_response(user_input.message)
        print(f"Bot response: {result}")  # Debug print
        return BotResponse(response=result['response'])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# uvicorn FairFace:app --reload
if __name__ == "__main__":
    uvicorn.run(app, host="46.4.82.183", port=8000)
