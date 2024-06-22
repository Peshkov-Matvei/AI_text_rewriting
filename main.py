import os
import re
import random
from dotenv import load_dotenv
from transformers import T5ForConditionalGeneration, T5Tokenizer
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


load_dotenv()


def load_input(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()


def save_output(filename, text):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(text)


def clean_text(text):
    return ' '.join(text.split())


class MarkovChain:
    def __init__(self):
        self.chain = defaultdict(list)

    def tokenize(self, text):
        tokens = word_tokenize(text.lower())
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

        start = random.choice(list(self.chain.keys()))
        result = list(start)

        for _ in range(length - k):
            key = tuple(result[-k:])
            if key in self.chain and self.chain[key]:
                next_word = random.choice(self.chain[key])
                result.append(next_word)
            else:
                break

        return ' '.join(result)


def generate_humanized_text(text, model_name="t5-base"):
    try:
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)

        cleaned_text = clean_text(text)

        prompt = f"{cleaned_text}"

        input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)

        output = model.generate(
            input_ids,
            max_length=1024,
            min_length=100,
            num_return_sequences=1,
            temperature=0.9,
            top_k=2000,
            top_p=0.9,
            repetition_penalty=2.5,
            no_repeat_ngram_size=2,
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

        markov_chain = MarkovChain()
        markov_chain.train(input_text, k=2)
        markov_generated_text = markov_chain.generate(length=100, k=2)

        humanized_text = generate_humanized_text(markov_generated_text)
        if humanized_text:
            save_output(output_file, humanized_text)
            print(f"Output saved to {output_file}")
        else:
            print("Processing failed.")
    else:
        print(f"No input text found in {input_file}.")
