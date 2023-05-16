import os
import openai
import time
import subprocess

openai.api_key = "<openai-api-key>"
os.environ['OPENAI_API_KEY'] = "<openai-api-key>"

subprocess.run("openai tools fine_tunes.prepare_data -f training_data.jsonl".split())

upload_response = openai.File.create(
  file=open("training_data.jsonl", "rb"),
  purpose='fine-tune'
)
# print(upload_response)
file_id = upload_response.id

# print(file_id)

ft_response = openai.FineTune.create(training_file=file_id, model='davinci', n_epochs=20, suffix = 'ex1')

found = False
while not found:
    rt_response = openai.FineTune.retrieve(id=ft_response.id)
    time.sleep(100)
    print(rt_response)
    if rt_response.fine_tuned_model is not None:
        found = True

new_prompt = "Which team won the Superbowl in 2022? ->"

print(rt_response.fine_tuned_model)
answer = openai.Completion.create(
  model=rt_response.fine_tuned_model,
  prompt=new_prompt,
  max_tokens=100,
  stop=[".\n"],
)
print(answer['choices'][0]['text'])
