import os
from dotenv import load_dotenv
from transformers import T5ForConditionalGeneration, T5Tokenizer

load_dotenv()

def load_input(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()

def save_output(filename, text):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(text)

def clean_text(text):
    return ' '.join(text.split())

def generate_humanized_text(text, model_name="t5-base"):
    try:
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)

        cleaned_text = clean_text(text)

        prompt = f"Rewrite the following text to make it sound more natural and human-like: {cleaned_text}"

        input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)

        output = model.generate(
            input_ids,
            max_length=1024,
            min_length=100,
            num_return_sequences=1,
            temperature=0.9,  # Температура для управления креативностью
            top_k=2000,  # Top-k sampling
            top_p=0.9,  # Nucleus sampling
            repetition_penalty=2.5,
            no_repeat_ngram_size=2,  # Размер n-грамм, которые не должны повторяться
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
        humanized_text = generate_humanized_text(input_text)
        if humanized_text:
            save_output(output_file, humanized_text)
            print(f"Output saved to {output_file}")
        else:
            print("Processing failed.")
    else:
        print(f"No input text found in {input_file}.")
