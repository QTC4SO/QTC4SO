import string
import torch
import pandas as pd
from tqdm import tqdm
from transformers.models.t5 import T5ForConditionalGeneration, T5Tokenizer
from gen import get_result

if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = T5ForConditionalGeneration.from_pretrained("shaoyuyoung/QTC4SO")
    tokenizer = T5Tokenizer.from_pretrained("shaoyuyoung/QTC4SO")
    model.to(DEVICE)
    prefix = "Python"
    incomplete_title = "How can"
    body = """I have a Python script which uses tkinter.messagebox to display an error message with traceback details 
    if an unexpected exception occurs. Displaying tracebacks this way has a few drawbacks.Traceback details aren't 
    helpful for the average user.Testers can't easily select and copy text from a messageboxComplex errors can have 
    large tracebacks which span dozens of lines.Instead of displaying error details by default, I would like to add a 
    "show details" button which would display more information in a read-only text field.How can I add a "show 
    details" button to a tkinter messagebox? """

    code = """
    import tkinter.messagebox as tm
    import traceback

    try:
        1/0
    except Exception as error:
        tm.showerror(title="Error",
                    message="An error has occurred: '" + str(error) + "'.",
                    detail=traceback.format_exc())
    """
    input_text = ' '.join(incomplete_title.split()[:16]) + " <body> " + ' '.join(
        body.split()[:256]) + " <code> " + ' '.join(code.split()[:256])
    result = get_result(prefix, input_text)
    title = incomplete_title + ' '+result
    print(title)
