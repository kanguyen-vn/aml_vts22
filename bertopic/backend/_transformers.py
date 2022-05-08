import numpy as np
from tqdm import tqdm
from typing import List
from bertopic.backend import BaseEmbedder
import torch
from transformers import AutoTokenizer


class TransformersBackend(BaseEmbedder):
    """ BERTweet embedding model

    The BERTweet embedding model used for generating document and
    word embeddings.

    Arguments:
        embedding_model: A BERTweet embedding model (subclassing transformers RobertaModel)

    Usage:

    To create a BERTweet backend, you need to create an transformers object via AutoModel and
    pass it through this backend:

    ```python
    import spacy
    from bertopic.backend import TransformersBackend

    nlp = spacy.load("en_core_web_md", exclude=['tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])
    spacy_model = SpacyBackend(nlp)
    ```

    To load in a transformer model use the following:

    ```python
    import spacy
    from thinc.api import set_gpu_allocator, require_gpu
    from bertopic.backend import SpacyBackend

    nlp = spacy.load("en_core_web_trf", exclude=['tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])
    set_gpu_allocator("pytorch")
    require_gpu(0)
    spacy_model = SpacyBackend(nlp)
    ```

    If you run into gpu/memory-issues, please use:

    ```python
    import spacy
    from bertopic.backend import SpacyBackend

    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_trf", exclude=['tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])
    spacy_model = SpacyBackend(nlp)
    ```
    """

    def __init__(self, embedding_model):
        super().__init__()

        if "transformers" in str(type(embedding_model)):
            self.embedding_model = embedding_model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.embedding_model.config._name_or_path
            )
        else:
            raise ValueError(
                "Please select a correct Transformers model by either using a string such as 'en_core_web_md' "
                "or create a nlp model using: `nlp = spacy.load('en_core_web_md')"
            )

    def embed(self, documents: List[str], verbose: bool = False) -> np.ndarray:
        """ Embed a list of n documents/words into an n-dimensional
        matrix of embeddings

        Arguments:
            documents: A list of documents or words to be embedded
            verbose: Controls the verbosity of the process

        Returns:
            Document/words embeddings with shape (n, m) with `n` documents/words
            that each have an embeddings size of `m`
        """

        # Extract embeddings from a transformer model

        embeddings = []
        for doc in tqdm(documents, position=0, leave=True, disable=not verbose):
            try:
                input_ids = torch.tensor([self.tokenizer.encode(doc)])
                with torch.no_grad():
                    features = self.embedding_model(input_ids)
                embedding = features.pooler_output.numpy().flatten()
            except:
                input_ids = torch.tensor([self.tokenizer.encode("An empty document")])

                with torch.no_grad():
                    features = self.embedding_model(input_ids)
                embedding = features.pooler_output.numpy().flatten()
            embeddings.append(embedding)
        embeddings = np.array(embeddings)

        return embeddings
