import os
import torch
from transformers import LlamaForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GenerationConfig
from time import perf_counter

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

def main():
    try:
        # Get the current directory
        current_dir = os.getcwd()
        print(f"Current working directory: {current_dir}")


        # Initialize the bot
        bot = LocalRadmanBot()

        print("\nBot is ready! Type 'quit' to exit.")
        print("-" * 50)


        while True:
            # Get user input
            user_input = input("You: ").strip()
            # Read the JSON body

            if user_input.lower() == 'quit':
                break

            # Process message and get response
            result = bot.generate_response(user_input)

            # print(f"\nBot: {result['response']}")
            print(f"{result['response']}")
            # print(f"(Intent: {result['intent']}, Confidence: {result['confidence']:.2f})")
            print("-" * 50)

    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nPlease check all and try again.")


# uvicorn FairFace:app --reload
if __name__ == "__main__":
    main()