import re
import spacy
import string
import pandas as pd


# load Note datasets
df_notes = pd.read_csv('/Users/mohammadshiri/Downloads/NOTEEVENTS.csv.gz', nrows=10000)

#  Preprocessing
def preprocess1(x):
    y = re.sub('\\[(.*?)\\]', '', x)  # remove de-identified brackets
    y = re.sub('[0-9]+\.', '', y)  # remove 1.2. since the segmenter segments based on this
    y = re.sub('dr\.', 'doctor', y)
    y = re.sub('m\.d\.', 'md', y)
    y = re.sub('admission date:', '', y)
    y = re.sub('discharge date:', '', y)
    y = re.sub('--|__|==', '', y)

    # remove, digits, spaces
    y = y.translate(str.maketrans("", "", string.digits))
    y = " ".join(y.split())
    return y

def preprocessing(df_notes):
    df_notes['TEXT'] = df_notes['TEXT'].fillna(' ')
    df_notes['TEXT'] = df_notes['TEXT'].str.replace('\n', ' ')
    df_notes['TEXT'] = df_notes['TEXT'].str.replace('\r', ' ')
    df_notes['TEXT'] = df_notes['TEXT'].apply(str.strip)
    df_notes['TEXT'] = df_notes['TEXT'].str.lower()

    df_notes['TEXT'] = df_notes['TEXT'].apply(lambda x: preprocess1(x))

    return df_notes

df_processed = preprocessing(df_notes)

# Notes to Sentences
from spacy.lang.en import English
nlp = English()  # just the language with no model
nlp.add_pipe('sentencizer')
# nlp.add_pipe(nlp.create_pipe('sentencizer'))

# nlp praser may not work when there is only one token. In these cases, we just remove them as note that has length 1 usually is some random stuff

def toSentence(x):
    doc = nlp(x)
    text=[]
    try:
        for sent in doc.sents:
            st=str(sent).strip()
            if len(st)<20:
                #a lot of abbreviation is segmented as one line. But these are all describing the previous things
                #so I attached it to the sentence before
                if len(text)!=0:
                    text[-1]=' '.join((text[-1],st))
                else:
                    text=[st]
            else:
                text.append((st))
    except:
        print(doc)
    return text

sents=df_processed['TEXT'].apply(lambda x: toSentence(x))

df_sents = pd.DataFrame(sents)
df_sents.columns = ['sentences']
df_sents.to_csv('sents.csv', index=False)

# Replace the abbreviations
def replace(tokens, sequence, dict):
    rep = False

    for token in tokens:
        if token.__str__() in ['a', 'in', 'to', '.', 'be', 'as']:
            break
        if dict.Abbreviation.eq(token.__str__()).any() \
                or dict.Abbreviation.eq(token.__str__() + '.').any():
            rep = True
            try:
                row = dict[dict.Abbreviation.eq(token.__str__())].iloc[0, :]
            except:
                row = dict[dict.Abbreviation.eq(token.__str__() + '.')].iloc[0, :]
            sequence = sequence.replace(token.__str__(), row.Meaning.split('(')[0])
    if rep:
        return sequence
    else:
        return 0

# Combine dictionaries and create personalized ones
dict1 = pd.read_excel('Med_Abbreviations.xlsx')
dict2 = pd.read_excel('Med_Abbreviations_copy.xlsx')
dict = pd.concat([dict1, dict2], ignore_index=True)
dict.Abbreviation = dict.Abbreviation.str.split(',')
dict = dict.explode('Abbreviation', ignore_index=True)
dict.Meaning = dict.Meaning.str.split(',')
dict = dict.explode('Meaning', ignore_index=True)
dict.Meaning = dict.Meaning.str.split('/')
nobody = dict.explode('Meaning', ignore_index=True)
nobody.drop_duplicates(inplace=True)
nurse = nobody.sample(frac=0.6)
doctor = nurse.sample(frac=0.6)

df_sents = df_sents.explode('sentences', ignore_index=True)
output = pd.DataFrame()
output['sentences'] = None
output['labels'] = None
labels = {'non-medical': '', 'doctor': '', 'nurse': ''}
sents = {'non-medical': '', 'doctor': '', 'nurse': ''}
max = 0
nlp = spacy.load('en_core_web_sm')

# Iterate over each sentence
for i, sequence in df_sents.sentences.iteritems():
    sequence = sequence.replace('\\', ' ')
    tokens = nlp(sequence)
    dicts = {'non-medical': nobody, 'doctor': doctor, 'nurse': nurse}
    for name, dict in dicts.items():
        modified = replace(tokens, sequence, dict)
        if modified:
            labels[name] = ' '.join([labels[name], modified])
            sents[name] = ' '.join([sents[name], sequence])
            max += tokens.__len__()

    if max > 500:
        for name in labels.keys():
            output = output.append({'sentences': f'"{name}" {sents[name]}',
                                    'labels': f'"{name}" {labels[name]}'}, ignore_index=True)
        labels = {'non-medical': '', 'doctor': '', 'nurse': ''}
        sents = {'non-medical': '', 'doctor': '', 'nurse': ''}
        max = 0

output.to_csv('labeled.csv', index=False)
