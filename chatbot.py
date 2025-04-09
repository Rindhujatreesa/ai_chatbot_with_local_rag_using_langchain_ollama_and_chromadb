# A simple illustration of Ollama API
from ollama import chat
from ollama import ChatResponse

response: ChatResponse = chat(model='gemma3:1b', messages=[
  {
      'role':'system',
      'content':'Give the answer in a single sentence'
  },
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])
print(response['message']['content'])
# or access fields directly from the response object
#print(response.message.content)