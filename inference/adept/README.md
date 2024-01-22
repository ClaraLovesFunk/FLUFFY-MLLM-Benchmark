Implementation Instructions/Example:
https://huggingface.co/adept/fuyu-8b

To run the model you'll need to install its dependencies. It's recommended to use a virtual environment (in case you do not see a requirements.txt file in this directory: it should be uploaded by mid febuary 2024):

```bash
python3 -m venv venvs/adept
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

FYI: 
In Summer 2023 the following problem was encountered: ImportError for FuyuProcessor in Transformers v4.34.1. The problem was solved in this forum discussion: https://huggingface.co/adept/fuyu-8b/discussions/30

Temperature: 
not adjustable