from src.config import get_settings
from moorcheh_sdk import MoorchehClient
import inspect

try:
    settings = get_settings()
    client = MoorchehClient(api_key=settings.moorcheh_api_key)
    print("Client attributes:", dir(client))
    if hasattr(client, 'vectors'):
        print("Vectors attributes:", dir(client.vectors))
        # Check if fetch exists
        if hasattr(client.vectors, 'fetch'):
             print("Fetch signature:", inspect.signature(client.vectors.fetch))
except Exception as e:
    print(e)
