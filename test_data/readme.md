**test_data package**

This package contains example data which are used in several tests in this respository.<br/>
*train_data.txt* and *valid_data.txt* can be used to construct training and validation data that can be used as input to feed into the composition models.<br/>
*embeddings.txt* are used to lookup the corresponding embeddings for the words in the example data.

*embeddings.txt*:<br/>
conatins 18 words with two-dimensional, normalized word embeddings.<br/>
Line 6 contains a vector for the unknown word (<unk>). The other lines contain modifier, heads or compounds.<br/>
The unknown word has the embedding [0.0 1.0] so that it can be identified easily
for test cases. The embeddings can be loaded and used by the gensim package.

*train_data.txt*:<br/>
6 lines of:<br/>
modifier head compound<br/>

*valid_data.txt*:<br/>
4 lines of:<br/>
modifier head compound<br/>
line 1 contains a modifier (Zitrone) that is not in the training data<br/>

*gold_standard.txt*<br/>
This file represents a gold standard prediction file were compounds and the 
predicted embeddings match the original embeddings. <br/>
8 lines of:<br/>
compound embedding


