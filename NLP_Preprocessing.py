# nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# nltk download
nltk.download('punkt') # token
nltk.download('stopwords')

import pandas as pd
import spacy
import re
import os

class NLP_Preprocessing:
    def __init__(self, fi, drop_cols):
        self.fi = fi
        self.df = pd.read_csv(fi)
        self.drop_cols = drop_cols

    # Just for print information
    def quick_view(self, lst, num):
        print(f'[*] Quick view(total {num}):')
        for i in range(10):
            print(f'\t{i + 1}.{lst[i]}')
    
    def Drop_col(self):
        for col in self.drop_cols:
            self.df.drop(col, axis=1, inplace=True)

        # Drop all na
        self.df.dropna(inplace=True)
    
    def lowercasing(self):
        # Lowercasing text
        for column in self.df.columns:
            # Lowercasing just 'object' Dtype
            if self.df[column].dtype == 'object':
                print(f'[*] Lowercasing {column}...')
                self.df[column] = self.df[column].str.lower()
            
            else:
                print(f'[*] {self.df[column].dtype} - {column}: PASS')
    
    def remove_noise(self):
        # Remove everything except alphabetic and number
        # Output: clean_texts -> list()
        print('[*] Remove noise...')
        pattern = re.compile(r'[^a-zA-Z0-9\s]')

        clean_texts = []
        for i in self.df.index:
            print(f'\r[*] Progress: {round(round((i + 1)/self.df.shape[0], 3)*100, 2)}%', end='')
            clean_text = re.sub(pattern, ' ', self.df.loc[i, 'review'])
            clean_texts.append(clean_text)

        clean_texts_num = len(clean_texts)
        print('\n[*] Done.')
        self.quick_view(clean_texts, clean_texts_num)

        return clean_texts
    
    def remove_stopwds(self, clean_texts):
        # Remove stopwords from clean_texts base on nltk library
        # Output: clean_texts -> list()
        print('[*] Remove stop words...')

        # Stop words table (from nltk)
        stop_wds = stopwords.words('english') 

        tmp = []
        clean_texts_num = len(clean_texts)
        for i in range(clean_texts_num):
            print(f'\r[*] Progress: {round(round((i + 1)/clean_texts_num, 3)*100, 2)}%', end='')
            
            clean_text = ' '.join([wd for wd in clean_texts[i].split(' ') if wd not in stop_wds])
            tmp.append(clean_text)

        clean_texts = tmp[:]
        tmp = []

        clean_texts_num = len(clean_texts)
        print('\n[*] Done.')
        self.quick_view(clean_texts, clean_texts_num)

        return clean_texts

    def recognize_entity(self, clean_texts):
        # Identifying important entities and save them to entities_catalog.csv
        # Output: entities_catalog -> dict()
        print('[*] Recognize entity...')

        # Entities table (from spacy)
        recognition = spacy.load("en_core_web_sm")

        entities_catalog = {}
        clean_texts_num = len(clean_texts)
        for i in range(clean_texts_num):
            print(f'\r[*] Progress: {round(round((i + 1)/clean_texts_num, 3)*100, 2)}%', end='')
            
            doc = recognition(clean_texts[i]) # recognize core entities
            for ent in doc.ents:
                entities_catalog.setdefault(ent.label_, [])
                entities_catalog[ent.label_].append(ent.text)

        # Print data info
        entities_catalog_num = len(entities_catalog.keys())
        print('\n[*] Done.')
        print(f'[*] Quick view(total {entities_catalog_num}):')
        for i in range(10):
            curr_k = list(entities_catalog.keys())[i]
            print(f'\t{i + 1}.{curr_k} - {entities_catalog[curr_k][1:10]}')

        # Save file
        fo_path = os.path.join(os.path.dirname(self.fi), 'entities_catalog.csv')
        print(f'[*] Generating entities_catalog.csv...')
        entities_catalog_df = pd.DataFrame({'entity name': entities_catalog.keys(), 'entity contents': entities_catalog.values()})
        entities_catalog_df.to_csv(fo_path, index=False)
    
    def tokenization(self, clean_texts):
        # Tokenize data
        # Output: tokens -> list()
        print('[*] Tokenizing...')

        tokens = []
        clean_texts_num = len(clean_texts)
        for i in range(clean_texts_num):
            print(f'\r[*] Progress: {round(round((i + 1)/clean_texts_num, 3)*100, 2)}%', end='')
            
            token_wd = word_tokenize(clean_texts[i])
            tokens.append(token_wd)

        tokens_num = len(tokens)
        print('\n[*] Done.')
        self.quick_view(tokens, tokens_num)
        
        return tokens
    
    def normalization(self, tokens):
        # Lemmatization with nltk
        # Output: tokens -> list()
        print('[*] Normalizing...')
        stemmer = PorterStemmer()

        tmp = []
        tokens_num = len(tokens)
        for i in range(tokens_num):
            print(f'\r[*] Progress: {round(round((i + 1)/tokens_num, 3)*100, 2)}%', end='')

            lemmatized_token = [stemmer.stem(wd) for wd in tokens[i]]
            tmp.append(lemmatized_token)

        lemmatized_tokens = tmp[:]
        tmp = []

        tokens_num = len(lemmatized_tokens)
        print('\n[*] Done.')
        self.quick_view(lemmatized_tokens, tokens_num)

        return lemmatized_tokens
    
    def filter_tokens(self, tokens):
        # Filter all non-alphabetic word
        # Output: alpha_tokens -> list()
        print('[*] Filtering tokens...')

        alpha_tokens = []
        tokens_num = len(tokens)
        for i in range(tokens_num):
            print(f'\r[*] Progress: {round(round((i + 1)/tokens_num, 3)*100, 2)}%', end='')
            
            alpha_tokens.append([wd for wd in tokens[i] if wd.isalpha()])

        tokens_num = len(alpha_tokens)
        print('\n[*] Done.')
        self.quick_view(alpha_tokens, tokens_num)

        return alpha_tokens
    
    def save_file(self, clean_texts, tokens, alpha_tokens):
        # New data
        # clean_texts -> list()
        # tokens -> list()
        # alpha_tokens -> list()
        print('[*] Saving files...')

        self.df.insert(self.df.shape[1], 'clean_texts', clean_texts, allow_duplicates=True)
        self.df.insert(self.df.shape[1], 'tokens', tokens, allow_duplicates=True)
        self.df.insert(self.df.shape[1], 'alpha_tokens', alpha_tokens, allow_duplicates=True)

        fo_path = os.path.join(os.path.dirname(self.fi), f'preprocessed_{os.path.basename(self.fi)}')
        self.df.to_csv(fo_path, index=False)

        print('[*] Done.')
    
    def main(self):
        '''Clean Texts'''
        # Drope nan and unnecessary columns
        self.Drop_col()

        # Lowercasing text
        self.lowercasing()

        # Remove everything except alphabetic and number
        clean_texts = self.remove_noise()

        # Remove stopwords from clean_texts base on nltk library
        clean_texts = self.remove_stopwds(clean_texts)

        '''Entity Recognition'''
        # Identifying important entities and save them to entities_catalog.csv
        self.recognize_entity(clean_texts)

        '''Token'''
        # Tokenize data
        tokens = self.tokenization(clean_texts)

        # Lemmatization
        tokens = self.normalization(tokens)

        # Filter all non-alphabetic word
        alpha_tokens = self.filter_tokens(tokens)

        '''Save File'''
        self.save_file(clean_texts, tokens, alpha_tokens)

if __name__ == '__main__':
    fi = 'drugsComTrain_raw.csv'

    # Droped columns
    drop_cols = ['date', 
                 'uniqueID'] 

    pipeline = NLP_Preprocessing(fi, drop_cols)
    pipeline.main()
