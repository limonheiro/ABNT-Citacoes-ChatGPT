import openai
# from bs4 import BeautifulSoup
from pypdf import PdfReader
from gensim.models import TfidfModel, OkapiBM25Model
from gensim.corpora import Dictionary
from gensim.similarities import SparseMatrixSimilarity
import numpy as np
import tracemalloc
tracemalloc.start()

# Autenticando a API do OpenAI
openai.api_key = "YOU_KEY_API"


# Lendo o arquivo PDF e extraindo o texto
def get_pdf(filename):
    text_pdf = []
    with open(filename, 'rb') as pdf_file:
        pdf_reader = PdfReader(pdf_file)
        metadata = pdf_reader.metadata
        for number_page in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[0]
            text_pdf += page.extract_text().split("\n")

    return text_pdf, metadata


def get_resume_text(text_article, tag_query, n=5):
    # Criando um resumo do texto
    tokenized_corpus = np.unique(text_article)
    tokenized_corpus = [doc.lower().split(" ") for doc in tokenized_corpus]
    dictionary = Dictionary(tokenized_corpus)
    tokenized_query = tag_query.lower().split(" ")

    bm25_model = OkapiBM25Model(dictionary=dictionary)
    bm25_corpus = bm25_model[list(map(dictionary.doc2bow, tokenized_corpus))]
    bm25_index = SparseMatrixSimilarity(bm25_corpus, num_docs=len(tokenized_corpus), num_terms=len(dictionary),
                                        normalize_queries=False, normalize_documents=False)

    tfidf_model = TfidfModel(dictionary=dictionary, smartirs='bnn')  # Enforce binary weighting of queries
    tfidf_query = tfidf_model[dictionary.doc2bow(tokenized_query)]
    similarities = bm25_index[tfidf_query]

    tokenized_corpus_str = [tokenized_corpus[i] for i in similarities.argsort()][::-1][:n]
    tokenized_corpus_str = " ".join([" ".join(t) for t in tokenized_corpus_str])

    return tokenized_corpus_str


def openai_response(text, tag, frases=3, tipo_citacao='indireta', estilo_citacao='ABNT', autor='Fulano de Tal', ano='2099'):
    # Interpretando o significado do texto com a API do OpenAI
    citation = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"Citação {tipo_citacao} do texto estilo {estilo_citacao} do Autor {autor}, "
                                      f"Ano {ano} sobre o tema {tag} "
                                      f"{' com' + str(frases) + 'frases tudo no mesmo paragrafo em portugues.' if frases > 0 else ''}: {text}",

        max_tokens=1024,
        temperature=0.2,
    )
    return citation
