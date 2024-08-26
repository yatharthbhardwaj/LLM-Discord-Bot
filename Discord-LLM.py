#NzMzNjMwNTg2OTcwNzAxODU1.GgK6lF.lmuqHbhbReDlEz60aSU_lHB3lrnhZ1dCNvUmAQ
# pip install accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import discord
from discord.ext import commands


# Ensure you have your Hugging Face token set as an environment variable
HUGGINGFACE_TOKEN = "Your Huggin Face Token"

# Load the tokenizer and model with authentication
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it", use_auth_token=HUGGINGFACE_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b-it",
    use_auth_token=HUGGINGFACE_TOKEN,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# Check if CUDA is available and set the device accordingly
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

bot = commands.Bot(command_prefix='>', self_bot=True)

ALLOWED_CHANNEL_IDS = [1268207657207206010, 1252709125448794154, 1176974285806518434, 1258549103823032444]

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return  
    if message.channel.id in ALLOWED_CHANNEL_IDS:
        print(f"Message from {message.author.display_name} in {message.channel.name}: {message.content}")

        response = generate_text_response(message.content)
        

        await message.channel.send(response)

def generate_text_response(text):
    # text generation logic
    input_text = text
    input_ids = tokenizer(input_text, return_tensors="pt").to(device)

    outputs = model.generate(**input_ids, max_new_tokens=320)
    return tokenizer.decode(outputs[0])



bot.run("OTgwNDQyMjM0NzYxNjYyNDg0.GgsKzu.V_mpHFcP41H8c-fk3faNiJn9pp28btAY_b9pp4")
