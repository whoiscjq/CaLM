# import openai
from openai import OpenAI
import time
MODEL = "o1-mini"

def startup(api_key):
    return api_key


def query(context, query_text, dry_run=False):
    
    client = OpenAI(api_key = context,base_url='https://api.claudeshop.top/v1/chat/completions')

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": query_text}
            ],
        temperature=0,
    )
    print(response)
    return response.choices[0].message.content

