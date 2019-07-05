---
layout:     post
title:      "''Analogies Explained'' ... Explained"
date:       2019-07-01 15:44:00 +0100
author:     Carl Allen
paper-link: https://arxiv.org/abs/1901.09813
link-text:  "[arXiv]"
categories: NLP

---

> This post provides a `less maths, more intuition' explanation of [Analogies Explained: Towards Understanding Word Embeddings][paper] (ICML, 2019, Best Paper Honourable Mention) with the aim of being accessible to a general machine learning audience, and others (e.g. NLP, linguists) that may prefer a less formal "theorem-proof" approach than the original paper. The outline follows the [ICML presentation][presentation].

[comment]: # (`)

## Background [<font size="3">[Skip to main result]</font>](#proof)

### Word Embeddings

*Word embeddings* are numerical vector representations of words. Each entry, or dimension, of a word embedding can be thought of as capturing some semantic or syntactic feature of the word, and the full vector can be considered as co-ordinates of the word in a high-dimensional space. Word embeddings can be generated explicitly, e.g. from rows of word co-occurrence statistics (or low-rank approximations of such statistics); or by *neural network* methods such as *[Word2Vec]* (W2V) or *[Glove]*.

[comment]: # ( [[Mikolov et al., 2013]][Word2Vec] `[[Pennington et al.,2014]][glove]`)

The latter, *neural word embeddings*, are found to be highly useful in natural language processing (NLP) tasks, such as evaluating word similarity, identifying named entities and assessing positive or negative sentiment in a passage of text (e.g. a customer review).

### Analogies

An intriguing property of neural word embeddings is that *analogies* can often be solved simply by adding/subtracting word embeddings. For example, analogy:

$$``man$$ is to $$king$$ as $$woman$$ is to $$queen\!"$$
{: style="text-align: center"}

[comment]: # (``)

can be *solved* using word embeddings by finding that closest to $$\mathbf{w}_{king} - \mathbf{w}_{man} + \mathbf{w}_{woman}$$, which turns out to be $$ \mathbf{w}_{queen} $$
($\mathbf{w}_{x}$ denotes the embedding of word $x$). Note that words in the question (e.g. *man*, *woman* and *king*) are typically omitted from the search.
In other words,

$$
\begin{equation}
  \mathbf{w}_{king} - \mathbf{w}_{man} + \mathbf{w}_{woman} \approx \mathbf{w}_{queen}
  \tag{1}\label{eq:one}
\end{equation}
$$

That is, ***word embeddings of analogies approximately form parallelograms:***
^
![word embeddings of analogies approximately form a paralellogram](/assets/parallelogram5.png){:height="150px"}.
{: style="text-align: center"}

Whilst fitting intuition, this phenomenon is intriguing since word embeddings are not trained to achieve it!

In practice, the word embeddings of analogies do not perfectly form a parallelogram:

![word embeddings of analogies not quite a paralellogram](/assets/analogy_embeddings.png){:height="250px"}
{: style="text-align: center"}
This shows the (exact) parallelogram formed by $$\mathbf{w}_{king}, \mathbf{w}_{man}$$, $$\mathbf{w}_{woman}$$ and  $$\mathbf{w}_{king} {\small-} \mathbf{w}_{man} {\small+} \mathbf{w}_{woman}$$ fixed in the $xy$-plane and a selection of word embeddings shown relative to them. We see that the embedding of queen $$\mathbf{w}_{queen}$$ does not sit at the corner, but is the closest to it. Word embeddings of related words, e.g. *prince* and *lord*, lie relatively close by and random unrelated words further away.

|Our work explains the gap (indicated) between $$\mathbf{w}_{queen}$$ and $$\mathbf{w}_{king} {\small-} \mathbf{w}_{man} {\small+} \mathbf{w}_{woman}$$; and why that gap is small, often smallest, for the word that completes the analogy.|
{:.mbtablestyle style="text-align: center"}

To understand why *semantic* relationships between words give rise to *geometric* relationships between word embeddings, we first consider what W2V embeddings learn.

### Word2Vec

![W2V architecture](/assets/neural_network.png){:style="float: right; margin-right: 0px; margin-top: -10px;" height="250px"}

W2V (specifically, SkipGram with negative sampling) is an algorithm that generates word embeddings by training the weights of a 2-layer "neural network"
to predict $$c_j$$ the *context words* (words that fall within a fixed size context window) around each word  $$w_i$$ (referred to as a *target word*) accross a text corpus.

Whilst predicting $$p(c_j\!\mid\! w_i)$$ for all $$c_j$$ in a dictionary of unique words $$\mathcal{E}$$ using a softmax function was initially considered, a sigmoid function and negative sampling are instead used due to computational cost.

[Levy & Goldberg (2014)][levy-goldberg] showed that, as a result, the two weight matrices $$\mathbf{W}, \mathbf{C}$$ (columns of which form the word embeddings $$\mathbf{w}_i, \mathbf{c}_j$$) factorise a matrix of *shifted* Pointwise Mutual Information (PMI), i.e.

$$
\begin{equation}
\mathbf{W}^\top\mathbf{C} \approx \textbf{PMI} - \log k
\end{equation}
$$

where $k$ is the chosen number of negative samples and
$$
\begin{equation}
\textbf{PMI}_{i,j} = \text{PMI}(w_i, c_j) = \log\tfrac{p(w_i,\, c_j)}{p(w_i)p(c_j)}
\end{equation}\ .
$$

Dropping the *shift* term ($$\log k$$), an artefact of the *W2V* algorithm, the relationship
$$
\begin{equation}
\mathbf{W}^\top\!\mathbf{C} \!\approx\! \textbf{PMI}
\end{equation}
$$
shows an embedding $$\mathbf{w}_i$$ to be a *low-dimensional projection* of a row of the PMI matrix, $$\text{PMI}_i$$ (a *PMI vector*).

---
---
<br>
{: #proof}
## Proving the embedding relationship of analogies

From above, we see that the additive relationship between word embeddings of analogies \eqref{eq:one} follows if
an equivalent relationship exists between PMI vectors, i.e.

$$
\begin{equation}
  \text{PMI}_{king} - \text{PMI}_{man} + \text{PMI}_{woman} \approx \text{PMI}_{queen}\ ;
  \tag{2}\label{eq:two}
\end{equation}
$$

and vector addition is sufficiently preserved under the low-rank projection induced by the loss function -- as readily achieved by a least squares loss function and approximately achieved by *W2V* and *Glove*.

|To prove that relationship \eqref{eq:two} arises between PMI vectors of an analogy, we show that \eqref{eq:two} follows from a particular **paraphrase** relationship, which is then shown to be equivalent to an analogy.|
{:.mbtablestyle  style="text-align: center"}


### Paraphrases

When we say a word $$w_*$$ *paraphrases* a set of words $$\mathcal{W}$$, we mean intuitively that they are *semantically interchangeable* in the text. For example, where $$king$$ appears, we might instead see  $$man$$ <u>and</u>  $$royal$$ close together.
Mathematically, a best choice paraphrase word $$w_*$$ can be defined as that which maximises the likelihood of the context words observed around $$\mathcal{W}$$.
In other words, the distribution of words observed around $w_*$ defined over the dictionary $$p(\mathcal{E}\!\mid\!w_*)$$, should be similar to that around $$\mathcal{W}$$ $$p(\mathcal{E}\!\mid\!\mathcal{W})$$, as measured by Kullback-Leibler (KL) divergence. Whilst these distributions are discrete and unordered, we can picture this as:

![Paraphrase distributions](/assets/distributions.png){:height="120px"}
{: style="text-align: center"}

> Formally, we say $$w_*$$ **paraphrases** $\mathcal{W}$ if the **paraphrase error** $${\rho}^{\mathcal{W}, w_*}\!\in\! \mathbb{R}^{n}$$
 is (element-wise) small:
$$
\begin{equation}
    {\rho}^{\mathcal{W}, w_*}_j = \log \tfrac{p(c_j|w_*)}{p(c_j|\mathcal{W})}\ ,   \quad c_j\!\in\!\mathcal{E}.
    \tag{3}\label{eq:three}
\end{equation}
$$

To see the relevance of paraphrases, we compare the sum of PMI vectors of two words $$w_1, w_2$$ to that of **<u>any word</u>** $$w_*$$ by considering each ($$j^{th}$$) component of the difference vector $$\text{PMI}_* - (\text{PMI}_1 + \text{PMI}_2)$$:

![W2V architecture](/assets/equation_PMI.png){:height="230px"}
{: style="text-align: center"}

We see that the difference can be written as a *paraphrase error*, small only if $$w_*$$ paraphrases $$\{w_1, w_2\}$$, and *dependence error* terms that are inherent to $$w_1$$ and $$w_2$$ and do not depend on $$w_*$$.
Formally, for context window size $$l$$, we have:

> Lemma 1:
For any word $$w_*\!\in\!\mathcal{E}$$ and word set $$\mathcal{W}\!\subseteq\!\mathcal{E}$$, $$|\mathcal{W}|\!<\!l$$:
>
>$$
  \begin{equation}
    \text{PMI}_*
    \ =\
    \sum_{w_i\in\mathcal{W}} \text{PMI}_{i}
     \,+\, \rho^{\mathcal{W}, w_*}
     \,+\, \sigma^{\mathcal{W}}
     \,-\, \tau^\mathcal{W}\mathbf{1}\ .
\end{equation}
$$

[comment]: # (![Paraphrase distributions](/assets/lemma1-1.png){:height="100px"} {: style="text-align: left; margin-left: 30px"})

This connects paraphrasing to PMI vector *addition*, as appears in \eqref{eq:two}. To extend this, paraphrasing can be generalised to a relationship between any *two word sets*, replacing $$w_*$$ by  $$\mathcal{W}_*$$. The underlying principle remains the same: word sets paraphrase one another if the distributions of context words around them are similar. (Note: if $$\mathcal{W}_*$$ contains a single word, the previous paraphrase definition is recovered.) Analogously to above, we find that:

> Lemma 2:
For any word sets $$\mathcal{W}, \mathcal{W}_*\!\subseteq\!\mathcal{E}$$; $$|\mathcal{W}|, |\mathcal{W}_*|\!<\!l$$:
>
>$$
  \begin{equation}
    \sum_{w_i\in\mathcal{W}_*} \text{PMI}_{i}
    \ =\
    \sum_{w_i\in\mathcal{W}} \text{PMI}_{i}
     \,+\, \rho^{\mathcal{W}, \mathcal{W}_*}
     \,+\, (\sigma^{\mathcal{W}} - \sigma^{\mathcal{W}_*})
     \,-\, (\tau^\mathcal{W} - \tau^{\mathcal{W}_*})\mathbf{1}
\end{equation}
$$

[comment]: # (![Paraphrase distributions](/assets/lemma2-1.png){:height="115px"} {: style="text-align: left; margin-left: 30px"})

Turning to our example, setting $$\mathcal{W} \!=\! \{woman, king\}$$ and $$\mathcal{W}_* \!=\!  \{man, queen\}$$, we have:

$$
\begin{equation}
  \text{PMI}_{king} - \text{PMI}_{man} + \text{PMI}_{woman} \ \approx\ \text{PMI}_{queen}\ ,
\end{equation}
$$

[comment]: # ($$\underbrace{\sigma^{\mathcal{W}} -  \sigma^{\mathcal{W_*}} - ( \tau^{\mathcal{W}} -  \tau^{\mathcal{W_*}})\mathbf{1}}_{\text{net dependence error}}$$ ![Paraphrase distributions](/assets/eqn_paraphrase.png){:height="115px"} {: style="text-align: left; margin-left: 30px"})

if $$\mathcal{W}$$ paraphrases $$\mathcal{W}_*$$ (meaning $$\rho^{\mathcal{W}, \mathcal{W}_*}$$ is small), subject to *net* statistical dependencies of words within $$\mathcal{W}$$ and $$\mathcal{W}_*$$, i.e. $$\sigma^{\mathcal{W}} - \sigma^{\mathcal{W}_*}$$ and $$\tau^\mathcal{W} - \tau^{\mathcal{W}_*}$$. Thus,

|Relation \eqref{eq:two} holds, subject to dependence error, if $$\{woman, king\}$$ paraphrases $$\{man, queen\}$$.|
{:.mbtablestyle style="text-align: center"}

This gives a semantic relationship as a sufficient condition for the geometric relationship we aim to explain -- but that semantic relationship is not an analogy.
What remains then, is to explain why a general analogy $$``w_a$$ is to $$w_{a^*}$$ as $$w_b$$ is to $$w_{b^*}\!\!"$$ implies that $$\{w_b, w_{a^*}\!\}$$ paraphrases $$\{w_a, w_{b^*}\!\}$$.
We show that these conditions are in fact equivalent by reinterpreting paraphrases as *word transformations*.

[comment]: # (``)

### Word Transformation

From [above](#paraphrases), the paraphrase of a word set $$\mathcal{W}$$ by a word $$w_*$$ can be thought of as drawing semantic equivalence between $$\mathcal{W}$$ and $$w_*$$. Alternatively, we can choose a particular word $w$ in $\mathcal{W}$ and view the paraphrase as indicating words (i.e. all others in $$\mathcal{W}$$, denoted $$\mathcal{W}^+$$) that, when added to $$w$$, make it "more like" -- or *transform* it to -- $$w_*$$. For example, the paraphrase of $$\{man, royal\}$$  by $$king$$ can be interpreted as a *word transformation* from $$man$$ to $$king$$ by adding $$royal$$. In effect, the added words *narrow the context*. More precisely, they alter the distribution of context words found around $$w$$ to more closely align with that of $$w_*$$. Denoting a paraphrase by $$\approx_p$$, we can represent this as:

![W2V architecture](/assets/word_trans_1.png){:width="300px"}
{: style="text-align: center"}

in which the paraphrase can be seen as the "glue" in a relationship between, or rather, *from* $$w$$ *to* $$w_*$$.
Thus, if $$w\!\in\!\mathcal{W}$$ and $$\mathcal{W}^+ \!=\! \mathcal{W}\!\setminus\!\!\{w\}$$, to say "$$w_*$$ paraphrases $$\mathcal{W}$$" is equivalent to saying "there exists a word transformation from $$w$$ to $$w_*$$ by adding $$\mathcal{W}^+$$".
To be clear, nothing changes other than perspective.

Extending the concept to paraphrases between word sets $$\mathcal{W}$$ and $$\mathcal{W}_*$$, we can choose any $$w \!\in\! \mathcal{W}$$ and $$w_* \!\in\! \mathcal{W}_*$$ and view the paraphrase as defining a relationship between $$w$$ and $$w_*$$ in which $$\mathcal{W}^+ \!=\! \mathcal{W}\!\setminus\!\!\{w\}$$ is added to $$w$$ and $$\mathcal{W}^- \!=\! \mathcal{W_*}\!\!\setminus\!\!\{w_*\}$$ to $$w_*$$:

![W2V architecture](/assets/word_trans_2.png){:width="300px"}
{: style="text-align: center"}

This is not a word transformation as above, since it lacks the same notion of direction *from* $$w$$ *to* $$w_*$$. To overcome this, we consider $$\mathcal{W}^-$$, the words added to $$w_*$$, as being *subtracted* from $$w$$ (hence the naming convention chosen):

![W2V architecture](/assets/word_trans_3.png){:width="300px"}
{: style="text-align: center"}

Where added words narrow context, subtracted words can be thought of as *broadening* the context.

> We say there exists a ***word transformation*** from word $$w$$ to word $$w_*$$, with ***transformation parameters*** $$\mathcal{W}^+\!\!$$, $$\mathcal{W}^-\!\subseteq\mathcal{E}\ $$    *iff*   $$\ \{w\}\!\cup\!\mathcal{W}^+ \approx_\text{P} \{w_*\}\!\cup\!\mathcal{W}^-$$.

#### Intuition

The intuition behind word transformations mirrors simple algebra, e.g. 8 is made *equivalent* to 5 by adding 3 to the right, or subtracting 3 from the left. Analogously, with paraphrasing as a measure of *equivalence*, we can identify words ($$\mathcal{W}^+, \mathcal{W}^-$$) that when added to/subtracted from $$w$$, make it equivalent to $$w_*$$.
In doing so, just as 3 *describes the difference* between 8 and 5 in the numeric example, we find words that describe the difference between $$w$$ and $$w_*$$, that is, how "$$w$$ is to $$w_*$$".

Where, in our initial case, words could only be *added* to $$w$$, limiting the tools available to describe its semantic difference to $$w_*$$ to discrete words in $$\mathcal{E}$$. Now, "differences between other words" can also be used, providing a far richer toolkit, e.g. the difference between $$man$$ and $$king$$ can be approximately explained by, say, $$royal$$ or $crown$, but can be more accurately described by the difference between $woman$ and $queen$.

### Interpreting Analogies

We can now mathematically interpret the language of an analogy:

> We say *$$\ `\!`w_a$$ is to $$w_{a^*}$$ as $$w_b$$ is to $$w_{b^*}\!$$"* *iff* there exist $$\mathcal{W}^+\!, \mathcal{W}^-\!\subseteq\!\mathcal{E}$$ that serve as transformation parameters that transform both $$w_a$$ to $$w_{a^*}$$ and $$w_b$$ to $$w_{b^*}$$.

That is, we interpret analogy wording: each instance of "is to" refers to the parameters of a word transformation; and the "as" implies their equality. Thus the semantic differences  within each word pair, as captured by the transformation parameters, are the same -- fitting intuition and now defined explicitly.

So, an analogy is a pair of word transformations with common parameters $$\mathcal{W}^+, \mathcal{W}^-$$, but what are those parameters? Fortunately, we need not search or guess. We show in the [paper] (Sec 6.4) that if an analogy holds, then *any* parameters that transform one word pair, e.g. $$w_a$$ to $$w_{a^*}$$, *must also transform the other pair*.
As such, we can chose $$\mathcal{W}^+\!=\!\{w_{a^*}\!\}$$, $$\mathcal{W}^-\!=\!\{w_{a}\}$$, which perfectly transform $$w_a$$ to $$w_{a^*}$$ since $$\{w_a, w_{a^*}\!\}$$ paraphrases $$\{w_{a^*\!}, w_a\}$$ exactly (note that ordering is irrelevant in paraphrases, e.g. $$\{man$$, $$royal\}$$ paraphrases  $$\{royal$$, $$man\}$$). But, if the analogy holds, those same parameters must also transform $$w_b$$ to $$w_{b^*}$$, meaning that $$\{w_b, w_{a^*}\}$$ paraphrases $$\{w_{b^*}, w_a\}$$. Thus,

|*$$`\!`w_a$$ is to $$w_{a^*}$$ as $$w_b$$ is to $$w_{b^*}\!\!"$$* $$\ $$ if and only if $$\ $$ $$\{w_b, w_{a^*}\}$$ paraphrases $$\{w_{b^*}, w_a\}$$.|
{:.mbtablestyle style="text-align: center"}

This completes the chain:
 - analogies are equivalent to word transformations with common transformation parameters that describe the semantic difference,
 - those word transformations are equivalent to paraphrases, one of which is rendered trivial under a particular choice of transformation parameters,
 - the other paraphrase leads to a geometric relationship between PMI vectors \eqref{eq:two}, subject to the accuracy of the paraphrase ($$\rho$$) and dependence error terms ($$\sigma, \tau$$),
 - under low-dimensional projection (induced by the loss function), the same geometric relationship manifests in word embeddings of analogies \eqref{eq:one}, as seen in word embeddings of *W2V* and *Glove*.

Returning to an earlier plot, we can now explain the "gap" in terms of paraphrase ($$\rho$$) and dependence $$(\sigma, \tau$$) error terms, and understand why it is small, often smallest, for the word completing the analogy.

![W2V architecture](/assets/solution.png){:height="300px"}
{: style="text-align: center"}

[paper]: https://arxiv.org/abs/1901.09813
[presentation]: https://icml.cc/media/Slides/icml/2019/104(13-11-00)-13-11-00-4883-analogies_expla.pdf
[Word2Vec]: https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
[Glove]: https://www.aclweb.org/anthology/D14-1162
[levy-goldberg]: https://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization.pdf
