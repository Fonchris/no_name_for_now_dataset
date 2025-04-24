# Ghomala Tokenizer using Hugging Face Tokenizers (BPE, Word-level)

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence
from tokenizers.processors import TemplateProcessing
from tokenizers.decoders import BPEDecoder
import json

# Step 1: Load Ghomala corpus from JSON
with open("C:/Users/AFC/PycharmProjects/no_name_for_now_dataset/Ghomala-datasets/BIBLE_EXTENDED_CORPUS.json", "r", encoding="utf-8") as f:
    data = json.load(f)

ghomala_sentences = [entry["Ghomala translation"] for entry in data if "Ghomala translation" in entry]

# Step 2: Save sentences to a temporary file for training
with open("ghomala_corpus.txt", "w", encoding="utf-8") as f:
    for sentence in ghomala_sentences:
        f.write(sentence.strip() + "\n")

# Step 3: Configure and train the tokenizer

# Normalizer: Normalize to NFD, lowercase, strip accents
normalizer = Sequence([NFD(), Lowercase(), StripAccents()])

# Initialize the tokenizer with BPE model
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.normalizer = normalizer

# Use whitespace pre-tokenizer
tokenizer.pre_tokenizer = Whitespace()

# Special tokens
special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

# Trainer configuration
trainer = BpeTrainer(vocab_size=8000, min_frequency=2, show_progress=True, special_tokens=special_tokens)

# Train on the corpus
tokenizer.train(["ghomala_corpus.txt"], trainer)

# Set post-processing for input/output templates (like BERT)
tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[("[CLS]", tokenizer.token_to_id("[CLS]")), ("[SEP]", tokenizer.token_to_id("[SEP]"))],
)

# Set decoder for inverse transform
tokenizer.decoder = BPEDecoder()

# Save the tokenizer locally
tokenizer.save("ghomala_tokenizer.json")

print("Tokenizer training complete. Saved to 'ghomala_tokenizer.json'.")
