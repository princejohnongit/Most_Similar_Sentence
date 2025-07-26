from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
potta_words=stopwords.words('english')
vectorizer= TfidfVectorizer(stop_words=potta_words,lowercase=True)

corpus=["This is the first. document.",
"This document is the second document.",
"And this is the third one.",
"Is this the first document?"]
tf_matrix=vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names_out())
print(tf_matrix.toarray())
#to do cosine similarity
print('---------------------------')
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
query=["second document"]
tf_query=vectorizer.transform(query)
#print(tf_query.toarray())

values=[]
for tf_doc in tf_matrix:
  values.append(cosine_similarity(tf_query,tf_doc))

print(*values)
'''index_of_high=np.argmax(values)
print(corpus[index_of_high])'''
similarity_scores = [v[0][0] for v in values]

print(values)
print(similarity_scores)
index_of_values_sorted=np.argsort(similarity_scores)
print(index_of_values_sorted)
#print(f"Cosine Simiarity betweeen {tf_query} and {tf_doc} is {value}")

'''# to print summary
for i in similarity_scores:
   print(docum[i])

print(document(similarity_score[0]))
print(document())'''
