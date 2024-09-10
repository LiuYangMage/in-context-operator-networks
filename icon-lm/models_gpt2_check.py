import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer, GPT2Config
from models_gpt2_source import GPT2Model as CustomGPT2Model



'''
The code of GPT-2 model in models_gpt2_source.py is slightly modified, for the purpose of multi-modal learning in ICON.
1, the positional embedding is removed from the forward method, so we need to manually add it before call gpt2 model.
2, the attention mask is 2D matrix (excluding the batch dimension)
In this script, we will test the custom GPT-2 model by comparing the last hidden state of the original GPT-2 model and the custom GPT-2 model.
'''

class OriginalGPT2(nn.Module):
    def __init__(self, model_name='gpt2'):
        super(OriginalGPT2, self).__init__()
        
        # Load the pre-trained GPT-2 model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        gpt_config = GPT2Config.from_pretrained(model_name)
        gpt_config.resid_pdrop = 0
        gpt_config.attn_pdrop = 0
        gpt_config.embd_pdrop = 0
        gpt_config.summary_first_dropout = 0
        self.gpt2 = GPT2Model.from_pretrained(model_name, config=gpt_config) # without LM Head
        # Define the language modeling head and tie its weights to the token embeddings
        self.lm_head = nn.Linear(self.gpt2.config.n_embd, self.tokenizer.vocab_size, bias=False)
        self.lm_head.weight = self.gpt2.wte.weight

    def forward(self, input_ids):
        last_hidden_state = self.gpt2(input_ids = input_ids).last_hidden_state
        output = self.lm_head(last_hidden_state)
        
        return last_hidden_state, output

  

class CustomGPT2(nn.Module):
    def __init__(self, model_name='gpt2'):
        super(CustomGPT2, self).__init__()
        
        # Load the pre-trained GPT-2 model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        gpt_config = GPT2Config.from_pretrained(model_name)
        gpt_config.resid_pdrop = 0
        gpt_config.attn_pdrop = 0
        gpt_config.embd_pdrop = 0
        gpt_config.summary_first_dropout = 0
        self.gpt2 = CustomGPT2Model.from_pretrained(model_name, config=gpt_config) # without LM Head
        # Define the language modeling head and tie its weights to the token embeddings
        self.lm_head = nn.Linear(self.gpt2.config.n_embd, self.tokenizer.vocab_size, bias=False)
        self.lm_head.weight = self.gpt2.wte.weight

    def forward(self, input_ids, attention_mask=None):
        # mannually add the positional embedding
        gpt2_embeddings = self.gpt2.wte(input_ids) # (batch_size, seq_length, hidden_size)
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0).expand(input_ids.size(0), -1) # (batch_size, seq_length)
        position_embeddings = self.gpt2.wpe(position_ids) # (batch_size, seq_length, hidden_size)
        transformed_input = gpt2_embeddings + position_embeddings # (batch_size, seq_length, hidden_size)
        last_hidden_state = self.gpt2(inputs_embeds=transformed_input, attention_mask = attention_mask)[0]
        output = self.lm_head(last_hidden_state)
        
        return last_hidden_state, output

  


def test_gpt2(input_text = None):
  # Initialize your model

  origin_model = OriginalGPT2()
  origin_model.eval()
  custom_model = CustomGPT2()
  custom_model.eval()

  if input_text is None:
    input_text = "My name is Alice and I"
  tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
  input_ids = torch.tensor(tokenizer.encode(input_text)).unsqueeze(0) # [1, seq_length]

  causal_mask = torch.tril(torch.ones(input_ids.size(1), input_ids.size(1), dtype = bool)).unsqueeze(0) # (1, seq_length, seq_length)
  print(causal_mask[0,:,:])
  with torch.no_grad(): 
      last_hidden_state_origin, output_logits_origin = origin_model(input_ids) # [1, seq_length, vocab_size]
      last_hidden_state_custom, output_logits_custom = custom_model(input_ids)
      last_hidden_state_custom_mask, output_logits_custom_mask = custom_model(input_ids, attention_mask = causal_mask)
      assert torch.allclose(last_hidden_state_origin, last_hidden_state_custom, atol=1e-4)
      assert torch.allclose(last_hidden_state_origin, last_hidden_state_custom_mask, atol=1e-4)
      print("Last hidden state is the same for the original and custom GPT2 model")



if __name__ == "__main__":
  test_gpt2()


