import os
import re
import random
from dotenv import load_dotenv
from transformers import T5ForConditionalGeneration, T5Tokenizer
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from langid import classify
import stanza
import pymorphy2

morph = pymorphy2.MorphAnalyzer()

load_dotenv()

stanza.download('en')
nlp_en = stanza.Pipeline('en', processors='tokenize,mwt,pos,lemma,ner')

def load_input(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()


def save_output(filename, text):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(text)


def clean_text(text):
    return ' '.join(text.split())


def lemmatize_text(text, lang='en'):
    if lang == 'ru':
        morph = pymorphy2.MorphAnalyzer()
        text = text.split(" ")
        for i in range(len(text)):
            text[i] = morph.parse(text[i])[0]
        lemmatized_text = ' '.join([word.normal_form for word in text])
    elif lang == 'en':
        doc = nlp_en(text)
        lemmatized_text = ' '.join([word.lemma for sent in doc.sentences for word in sent.words])
    else:
        lemmatized_text = ''
    return lemmatized_text

def lemmatize_text_russian(text):
    tokens = text.split()
    lemmatized_tokens = []
    for token in tokens:
        parsed_token = morph.parse(token)[0]
        lemma = parsed_token.normal_form
        lemmatized_tokens.append(lemma)
    lemmatized_text = ' '.join(lemmatized_tokens)
    return lemmatized_text

class MarkovChain:
    def __init__(self):
        self.chain = defaultdict(list)

    def tokenize(self, text):
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token.isalpha()]
        return tokens

    def train(self, text, k=1):
        tokens = self.tokenize(text)
        for i in range(len(tokens) - k):
            key = tuple(tokens[i:i + k])
            self.chain[key].append(tokens[i + k])


    def generate(self, length=50, k=2):
        if not self.chain:
            return ""

        start = list(self.chain.keys())[0]
        result = list(start)

        for i in range(length - k):
            key = tuple(result[-k:])
            if key in self.chain and self.chain[key]:
                next_word = random.choice(self.chain[key])
                result.append(next_word)
            else:
                break
        return ' '.join(result)

def generate_humanized_text_en(text, model_name="t5-base"):
    try:
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)

        cleaned_text = clean_text(text)

        prompt = f"{cleaned_text}"

        input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)

        output = model.generate(
            input_ids,
            max_length=200,
            min_length=0,
            num_return_sequences=1,
            temperature=0.9,
            top_k=50,
            top_p=0.85,
            repetition_penalty=6.0,
            no_repeat_ngram_size=4,
            length_penalty=1.0,
            do_sample=True,
            num_beams=10,
        )

        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text

    except Exception as e:
        print(f"Error generating humanized text: {e}")
        return None

def generate_humanized_text_ru(text, model_name="cointegrated/rut5-base-multitask"):
    try:
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)

        cleaned_text = clean_text(text)

        prompt = f"{cleaned_text}"

        input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)


        output = model.generate(
            input_ids,
            max_length=200,
            min_length=0,
            num_return_sequences=1,
            temperature=0.9,
            top_k=50,
            top_p=0.85,
            repetition_penalty=6.0,
            no_repeat_ngram_size=4,
            length_penalty=1.0,
            do_sample=True,
            num_beams=10,
        )

        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text

    except Exception as e:
        print(f"Error generating humanized text: {e}")
        return None


if __name__ == "__main__":
    input_file = 'AI_text_rewriting/input.txt'
    output_file = 'AI_text_rewriting/output.txt'

    input_text = load_input(input_file)

    if input_text:
        nltk.download('punkt')
        nltk.download('stopwords')

        input_text = input_text.replace('?', '?.').replace('!', '!.').replace('...', '.').replace(';', '.')

        input_text2 = input_text.split('\n')
        for i in range(len(input_text2)):
            input_text2[i] = input_text2[i].split('.')

        s = ''
        for j in range(len(input_text2)):
            for i in range(len(input_text2[j])):
                lang = classify(input_text2[j][i])[0]
                
                if input_text2[j][i]:
                    # Лемматизация текста в зависимости от языка
                    lemmatized_text = lemmatize_text(input_text2[j][i], lang=lang)

                markov_chain = MarkovChain()
                markov_chain.train(lemmatized_text, k=4)
                markov_generated_text = markov_chain.generate(length=10, k=4)
                if (lang == 'ru'):
                    humanized_text = generate_humanized_text_ru(markov_generated_text)
                else:
                    humanized_text = generate_humanized_text_en(markov_generated_text)
                if humanized_text:
                    humanized_text_bukva = humanized_text[0]
                    humanized_text_bukva = humanized_text_bukva.upper()
                    s += humanized_text_bukva + humanized_text[1:] + " "
            s += "\n"

        if s:
            save_output(output_file, s)
            print(f"Output saved to {output_file}")
        else:
            print("Processing failed.")
    else:
        print(f"No input text found in {input_file}.")
