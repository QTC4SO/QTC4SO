import string
import torch
import pandas as pd
from rouge import FilesRouge
from tqdm import tqdm
from transformers.models.t5 import T5ForConditionalGeneration, T5Tokenizer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = T5ForConditionalGeneration.from_pretrained("shaoyuyoung/QTC4SO")
tokenizer = T5Tokenizer.from_pretrained("shaoyuyoung/QTC4SO")
model.to(DEVICE)


def get_result(prefix, input_text):
    input_ids = tokenizer(str(prefix) + ": " + str(input_text), return_tensors="pt", max_length=512,
                          padding="max_length", truncation=True)

    summary_text_ids = model.generate(
        input_ids=input_ids["input_ids"].to(DEVICE),
        attention_mask=input_ids["attention_mask"].to(DEVICE),
        bos_token_id=model.config.bos_token_id,
        eos_token_id=model.config.eos_token_id,
        length_penalty=1.2,
        top_k=5,
        top_p=0.95,
        max_length=48,
        min_length=2,
        num_beams=3,
    )
    result = tokenizer.decode(summary_text_ids[0], skip_special_tokens=True)
    if result[-1] in string.punctuation:
        result = result[:-1] + " " + result[-1]
    return result


if __name__ == '__main__':

    data_path = "../datasets/"
    lans = ['java', 'python', 'html', 'javascript', 'c#', 'php', 'ruby', 'go']

    for lan in lans:

        test_df = pd.read_csv(data_path + lan + '/test.csv')
        test_df.columns = ["prefix", "input_text", "target_text"]
        result_list = []
        result_guo = []
        input = test_df['input_text'].tolist()
        results = test_df['target_text'].tolist()
        prefixs = test_df['prefix'].tolist()
        for i in tqdm(range(len(results))):
            input_text = input[i]
            try:
                result = get_result(prefixs[i], input_text)
            except:
                result = ''
            result_list.append(result)
            result_guo.append(results[i])
        print(lan)
        print(len(result_list))

        df = pd.DataFrame(result_list)
        preFile = data_path + '/predict_' + lan + '.csv'
        df.to_csv(preFile, header=False, index=False)
        goldenFile = data_path + '/Golden_' + lan + '.csv'
        df = pd.DataFrame(result_guo)
        df.to_csv(goldenFile, header=False, index=False)
