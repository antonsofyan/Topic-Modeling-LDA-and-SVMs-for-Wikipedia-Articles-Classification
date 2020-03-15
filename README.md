# Topic Modeling and Latent Dirichlet Allocation for Wikipedia Articles Classification

In this work, we aim to develop a binary classifier that allows discriminating between the documents of two different categories of articles on the English Wikipedia. For this, the Latent Dirichlet Allocation (LDA) algorithm is used to obtain the topics of each article and a Support Vector Machine (SVM) classifier to distinguish the articles from the distribution of LDA topics.

The steps to follow are:

1. Selection of two work categories, and download of data corpus, including at least 100 documents per category.
2. Preprocessing of articles.
3. Modeling of downloaded articles using the LDA topic extraction algorithm.
4. Binary classification of documents taking their subcategories as labels, and the distribution of LDA topics as input representation. Analysis of classifier performance.

The two selected subcategories belong to the Quantity category of Wikipedia. These are:
- Physical quantities
- Vacuum tubes

## Acknowledgements 

University Carlos III of Madrid, Data Processing.
