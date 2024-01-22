Implementation:
Inference of blip2 and instructblip conducted by following the instructions and implementation example of Salesforce/Lavis: https://github.com/salesforce/LAVIS/tree/main

To run the model you'll need to install its dependencies. It's recommended to use a virtual environment (in case you do not see a requirements.txt file in this directory: it should be uploaded by mid febuary 2024):

```bash
python3 -m venv venvs/instructblip
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

FYI: 
-

Temperature: 
0