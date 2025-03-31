from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def first(messages: [object]) -> str:
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        device_map=None,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        max_new_tokens=500,
        do_sample=False,
    )

    return generator(messages)


def first_example():
    message = {"role": "user", "content": "Create a funny joke about chickens."}
    output = first([message])
    print(output[0])


def word2vec(word: str) -> [(str, float)]:
    import gensim.downloader as api

    # model = api.load("word2vec-google-news-300")
    model = api.load("glove-wiki-gigaword-50")
    return model.most_similar([model['king']], topn=11)


def main():
    first_example()
    similar_words = word2vec('king')
    print(similar_words)
    pass


if __name__ == '__main__':
    main()
