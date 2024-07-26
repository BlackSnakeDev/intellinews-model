from transformers import MarianMTModel, MarianTokenizer

model_name = 'Helsinki-NLP/opus-mt-fr-en'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate(text, tokenizer, model):
    tokens = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**tokens)
    translation = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
    return translation

french_text = "Hamza Namira: ''Monter sur la scène de Carthage est un honneur''"
english_translation = translate(french_text, tokenizer, model)
print(f"French: {french_text}")
print(f"English: {english_translation}")
