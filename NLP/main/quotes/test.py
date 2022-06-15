import  quotes
import spacy 
import pandas as pd

nlp = spacy.load('en_core_web_lg')
df = pd.read_csv(r"C:\Users\tlebr\OneDrive - pku.edu.cn\Thesis\data\scmp\2021.csv")
row1 = df.iloc[1]
doc = nlp(row1.Body)#, user_data=row1.drop("Body").to_dict())

final_quotes = quotes.extract_quotes("1", doc)