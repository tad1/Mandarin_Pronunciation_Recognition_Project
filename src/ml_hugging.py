#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datasets import load_dataset

dataset = load_dataset("mozilla-foundation/common_voice_11_0", "zh-CN", split="train[:1%]", trust_remote_code=True)


# In[2]:


from datasets import Audio

dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))

def preprocess(batch):
    batch["input_values"] = batch["audio"]["array"]
    batch["input_length"] = len(batch["input_values"])
    return batch

dataset = dataset.map(preprocess, remove_columns=["audio"])


# In[3]:


from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")


# In[4]:


model.config.vocab_size = 5


# In[5]:


from tones import get_tones

def prepare_dataset(batch):
    batch["input_values"] = processor(batch["input_values"], sampling_rate=16_000).input_values[0]
    batch["labels"] = [tone.value-1 for tone in get_tones(batch["sentence"])]
    return batch

dataset = dataset.map(prepare_dataset, remove_columns=["client_id", "path", "sentence", "up_votes", "down_votes", "age", "gender", "accent", "locale", "segment", "input_length"])


# In[6]:


train_dataset = dataset.train_test_split(test_size=0.1)["train"]
eval_dataset = dataset.train_test_split(test_size=0.1)["test"]


# In[7]:


import torch
from torch.nn.utils.rnn import pad_sequence

class CustomDataCollatorCTCWithPadding:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        input_features = [torch.tensor(feature["input_values"]) for feature in features]
        label_features = [torch.tensor(feature["labels"]) for feature in features]

        input_features_padded = pad_sequence(input_features, batch_first=True, padding_value=self.processor.feature_extractor.padding_value)
        labels_padded = pad_sequence(label_features, batch_first=True, padding_value=-100)
        attention_masks = torch.zeros_like(input_features_padded).masked_fill(input_features_padded != self.processor.feature_extractor.padding_value, 1)

        return {
            "input_values": input_features_padded,
            "labels": labels_padded,
            "attention_mask": attention_masks
        }

data_collator = CustomDataCollatorCTCWithPadding(processor=processor)


# In[8]:


from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./wav2vec2",
    group_by_length=True,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    fp16=True,
    save_steps=500,
    eval_steps=500,
    logging_steps=500,
    learning_rate=1e-4,
    warmup_steps=500,
    save_total_limit=2,
)


# In[ ]:


from transformers import Trainer

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor.feature_extractor,
)

trainer.train()

# In[ ]:


metrics = trainer.evaluate()

