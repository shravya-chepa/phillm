# Fine-Tuning an LLM for Philosophy

By: Shravya Chepa

**Goal**: Fine-tune a model to write philosophical queries and receive responses focusing on various schools of thought like: Stoicism, Nihilism, Rationalism, Existentialism, Utilitarianism, Buddhism, Hinduism, Taoism, and Hedonism.

## *Installs*
The following modules need to be installed:
```pip install bitsandbytes 'accelerate>=0.26.0' peft datasets torch```

## *Process*

### 1. Dataset Preparation and Formatting: 
Gather texts from Stoicism, Hinduism, Taoism, and other schools of philosophy that are in the public domain. Make sure the books are public domain and free to use. The following books were used for this purpose:
- [Letters from a Stoic by Seneca (Stoicism)](https://archive.org/details/letters-from-a-stoic-1)
- [Beyond Good and Evil by Friedrich Nietzsche (Nihilism)](https://archive.org/details/beyondgoodandevi00nietuoft)
- [Ethics by Spinoza (Rationalism)](https://archive.org/details/in.ernet.dli.2015.263056)
- [Fear and Trembling and The Sickness Unto Death by Søren Kierkegaard (Existentialism)](https://archive.org/details/fear-and-trembling-and-the-sickness-unto-death)
- [Utilitarianism by John Stuart Mill (Utilitarianism)](https://archive.org/details/utilitarianism00millrich)
- [The Sacred Books of the East edited by Max Muller (Buddhism)](https://www.gutenberg.org/ebooks/2017)
- [The Song Celestial by Sir Edwin Arnold (Hinduism)](https://archive.org/details/FoZf_the-song-celestial-bhagavad-gita-by-sir-edwin-arnold-jaico-publishing-house)
- [Tao Te Ching by Laozi (Taoism)](https://archive.org/details/laozi_tao-te-ching)
- [On the Nature of Things by Lucretius (Hedonism)](https://archive.org/details/lucretiusonnatu00lucr)

Convert text into prompts and responses format. Use a semi-automated approach to generate these pairs. Use GPT o4 to generate these pairs from the books. Then structure the dataset in a consistent instruction-response format. Each data point is a JSON object with "instruction" field, an "input" field, and an "output" field. Since we need simple Q&A, the "input" is an empty string and "output" is the answer.

```
{
    "instruction": "What Stoic advice is given for dealing with life’s hardships such as poverty and death?",
    "input": "",
    "output": "Stoicism advises that each day, one should reflect on a single, meaningful idea that strengthens the mind against hardship. By regularly contemplating themes like poverty and mortality, one builds resilience and learns to face such conditions with calm acceptance."
}
```

Store the dataset in the data folder. There are two datasets that I experimented with: main_data.json (1035 objects) and expanded_data.json (2129 objects). These only differ in the number of examples I used for training.

### 2. Model Selection (Choosing a Base LLM):
I chose Phi-2 model because it is known for its strong performance despite its small-scale size. This model is then fine-tuned using LoRA.

### 3. Fine-Tuning Methodology:
Utilize supervised instruction tuning with parameter-efficient fine-tuning (PEFT) via Low-Rank Adaptation (LoRA). LoRA tunes only a small percentage of the model's weights by inserting low-rank adapters, instead of updating all the billions of parameters. Can also combine LoRA with 4-bit quantization (QLoRA). The modules transformers, peft and bitsandbytes are used for this purpose. 
- Environment setup.
- Load the base model and tokenizer with 4-bit precision loading enabled.
- Prepare LoRA adapters (prepare_model_for_kbit_training).
- Prepare the data for training. Tokenize the prompts and responses for the model. Format the instruction-response pairs so that only the response tokens contribute to the loss, by masking the prompt tokens during training (InstructionResponseCollator class defined).
Alpaca format style looks like the following:
```
### Instruction: [instruction text]
### Response: [response text]
```
- Set the training hyperparameters and run the training loop.

### 4. Use the Fine-Tuned Model
Once the training is complete, the model's LoRA adapter weights are saved within a folder. These weights can be loaded on top of the base model using PEFT library. I have three folders containing weights of three different models.
- philosophy_model0: This contains the weights of the training when I did not specify EOS (End of Sequence) token to every response during training so the model did not learn when to end the response. So I fixed that for subsequent training and tests.
- philosophy_model: This contains the weights of the training with the original dataset of 1035 entries. I found this model to be very overfit and wanted to improve performance against new unseen queries.
- philosophy_model2: This contains the weights after training on an expanded dataset with a 1000 more question-answer pairs.

### 5. Test Models with Prompts
Test the base model, model1 (1035 entries) and model2 (2129 entries) with new random questions.

### 6. Visualize Training Loss Curve
Read the training log from trainer_state.json and extract loss at each logging step. Then plot a line chart showing how the training loss decreases over time, visualizing the model's convergence.

### 7. Evaluation
Evaluate the model's answers. For this purpose, since the answers are open-ended, use GPT o3 to evaluate correctness, coherence and alignment to the specific philosophy of the responses.
Create new set of questions and compare the results from base model and the fine-tuned model.

### 8. Hugging Face Upload and Demo Application
Upload the model weights to Hugging Face and create a sample application to interact with the LLM model:
[https://huggingface.co/spaces/shravya-chepa/phi2-philosophy](https://huggingface.co/spaces/shravya-chepa/phi2-philosophy)

## *Hardware Used*
The experiments were conducted on an **NVIDIA A100 80GB PCIe GPU** with **CUDA 12.4** and **driver version 550.127.05**. MIG (Multi-Instance GPU) mode was **enabled**, with one instance active during testing.


## *References*

The following resources were used extensively to come up with strategy and to implement instruction fine tuning on open source LLMs.

- Raschka, S. (2023). *Build a large language model (from scratch)* [Book]. [https://learning.oreilly.com/library/view/build-a-large/9781633437166/](https://learning.oreilly.com/library/view/build-a-large/9781633437166/)

- Raschka, S. (n.d.). *Build a large language model (from scratch)* [Video series]. YouTube. [https://youtube.com/playlist?list=PLTKMiZHVd_2IIEsoJrWACkIxLRdfMlw11](https://youtube.com/playlist?list=PLTKMiZHVd_2IIEsoJrWACkIxLRdfMlw11)

- OpenAI. (2023). *ChatGPT* [Large language model]. [https://chatgpt.com/](https://chatgpt.com/)

- Shah, V. (2023, July 25). *The Hitchhiker’s guide to instruction tuning large language models*. Medium. [https://medium.com/@veer15/the-hitchhikers-guide-to-instruction-tuning-large-language-models-d6441dbf1413](https://medium.com/@veer15/the-hitchhikers-guide-to-instruction-tuning-large-language-models-d6441dbf1413)


## *Index for the Notebook*
1. Importing Modules and Initial Checks
2. Load Base Model
3. Test Base Model with a Prompt
4. Define InstructionResponseCollator to Format Dataset
5. Train Model with Dataset1 (1035 Entries)
6. Test Model1 with Prompts
7. Plot Training Loss Curve for Model1
8. Train Model with Dataset2 (2129 Entries)
9. Test Model2 with Prompts
10. Plot Training Loss Curve for Model2
11. Evaluation
    1. Evaluating the answers according to coherence, correctness and alignment with the philosophy
    2. Comparing Base Model VS Fine-Tuned Model Responses
12. Result
