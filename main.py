from dotenv import load_dotenv
import os
import boto3
import json

load_dotenv(dotenv_path=".env")
os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('AWS_ACCESS_KEY')
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('AWS_SECRET_ACCESS_KEY')

bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="ap-southeast-2"
)

response = bedrock.invoke_model(
    modelId="amazon.nova-micro-v1:0",
    body=json.dumps({
        "messages": [{"role": "user", "content": [{"text": "Say hello!"}]}],
        "inferenceConfig": {"maxTokens": 100, "temperature": 0.2}
    })
)

output = json.loads(response["body"].read())
print(output["output"]["message"]["content"][0]["text"])