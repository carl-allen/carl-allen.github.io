---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
title:  "About me..."
---

I am a [Laplace Junior Chair](https://data-ens.github.io/jobs/) in Machine Learning at École Normale Supérieure, Paris in the group of Stéphane Mallat, Giulio Biroli and Garbiele Peyré. I was previously a postdoctoral fellow at [ETH Zurich](https://ai.ethz.ch/) and finished my PhD in Machine Learning in 2021 at the [University of Edinburgh](https://www.ed.ac.uk/informatics), supervised by Profs [Tim 
Hospedales](https://homepages.inf.ed.ac.uk/thospeda/) and [Iain Murray](https://homepages.inf.ed.ac.uk/imurray2/bio.html).

**Research Interest**: My core interest is to mathematically understand mechanisms behind successful machine learning methods, e.g. neural networks, similarly to how physics describes the laws of nature. Recent machine learning models produce incredible results, but what is actually learned is often a mystery with unexplained failures cases and unknown risks in downstream applications. Scientific curiosity aside, by unpicking these models and the properties of the data they leverage, my research aims towards better performing, more interpretable and more reliable algorithms; and potentially a more fundamental understanding of the natural phenomena behind the data itself. 

**Current topics**: I consider how machine learning models exploit aspects of the data distribution from a probabilistic, often latent variable, perspective. Recent projects include: [explaining how VAEs *disentangle* independent factors of the data](https://arxiv.org/pdf/2410.22559), [identifying the mathematical model behind recent *self-supervised learning*](https://openreview.net/pdf?id=QEwz7447tR), and [deriving a probabilistic interpretation of *softmax classification*](https://openreview.net/pdf?id=EWv9XGOpB3).

My PhD ([_Towards a Theoretical Understanding of Word and Relation Representation_](https://arxiv.org/pdf/2202.00486.pdf)) investigated *neural representations* of discrete objects and their relationships, focusing on word embeddings learned from large text corpora (e.g. [word2vec](https://arxiv.org/pdf/1301.3781) or [GloVe](https://aclanthology.org/D14-1162.pdf)); and entity embeddings learned from large collections of facts ("_subject-relation-object_") in a knowledge graph. My main result shows [how word embeddings can seemingly be added and subtracted](http://proceedings.mlr.press/v97/allen19a/allen19a.pdf), e.g. *queen ≈ king - man + woman*, which received Best Paper (honourable mention) at ICML 2019.

During an internship at [Samsung AI Centre, Cambridge](https://research.samsung.com/aicenter_cambridge) I worked at the interestection of representation learning and logical reasoning.

**Background**: I moved to Artificial Intelligence/Machine Learning research after a number of years in [Project Finance](https://tenor.com/view/why-why-why-dee-pqperplqnes-gif-20613669).
I have a BSc in Mathematics & Chemistry (<a href="https://www.southampton.ac.uk/">University of Southampton</a>),
an MSc Mathematics and the Foundations of Computer Science (MFoCS, <a href="https://www.ox.ac.uk/">University of Oxford</a>) and
MScs in Artificial Intelligence and Data Science (<a href="https://www.ed.ac.uk/">University of Edinburgh</a>).

**Awards & Invited Talks**: Aside from the Best Paper (hon. mention) I have been awarded a research grant from the [Hasler Foundation](https://haslerstiftung.ch/) and have given several invited talks, including at Harvard Center of Mathematical Sciences & Applications and Astra-Zeneca.

## Publications

  <p><strong>Unpicking Data at the Seams: VAEs, Disentanglement and Independent Components</strong>
    <a href="https://arxiv.org/pdf/2410.22559">[paper]</a> <br />
  <u>C Allen</u>;
  <em> under review</em>, 2024 <br />
  </p>

  <p><strong>A Probabilistic Model behind Self-Supervised Learning</strong>
    <a href="https://openreview.net/pdf?id=QEwz7447tR">[paper]</a> <br />
  A Bizeul, B Schölkopf, <u>C Allen</u>;
  <em> TMLR</em>, 2024 <br />
  </p>

  <p><strong>Variational Classification: A Probabilistic Generalization of the Softmax Classifier</strong>
    <a href="https://arxiv.org/pdf/2305.10406">[paper]</a> <br />
  S Dhuliawala, M Sachan, <u>C Allen</u>;
  <em> TMLR</em>, 2023 <br />
  </p>

  <p><strong>Learning to Drop Out: An Adversarial Approach to Training Sequence VAEs</strong>
    <a href="https://proceedings.neurips.cc/paper_files/paper/2022/file/3ed57b293db0aab7cc30c44f45262348-Paper-Conference.pdf">[paper]</a> <br />
  Đ Miladinović, K Shridhar, K Jain, M Paulus, JM Buhmann, <u>C Allen</u>;
  <em> NeurIPS</em>, 2022 <br />
  </p>

  <p><strong>Adapters for Enhanced Modelling of Multilingual Knowledge and Text</strong>
    <a href="https://arxiv.org/pdf/2210.13617">[paper]</a> <br />
  Y Hou, W Jiao, M Liu, <u>C Allen</u>, Z Tu, M Sachan;
  <em> EMNLP</em>, 2022 <br />
  </p>

  <p><strong>Interpreting Knowledge Graph Relation Representation from Word Embeddings</strong>
    <a href="https://arxiv.org/pdf/1909.11611">[paper]</a> <br />
  <u>C Allen</u>*, I Balažević*, T Hospedales;
  <em> ICLR</em>, 2021 <br />
  </p>

  <p><strong>Multi-scale Attributed Embedding of Networks</strong>
    <a href="https://arxiv.org/abs/1909.13021">[paper]</a> <a href="https://github.com/benedekrozemberczki/MUSAE">[github]</a><br />
  B Rozemberczki, <u>C Allen</u>, R Sarkar;
  <em> Journal of Complex Networks </em>, 2021 <br />
  </p>

  <p><strong>A Probabilistic Model for Discriminative & Neuro-Symbolic Semi-Supervised Learning</strong>
    <a href="https://arxiv.org/pdf/2006.05896">[paper]</a><br />
  <u>C Allen</u>, I Balažević, T Hospedales;
  2020 <br />
  </p>

  <p><strong>What the Vec? Towards Probabilistically Grounded Embeddings</strong>
    <a href="https://arxiv.org/abs/1805.12164">[paper]</a><br />
  <u>C Allen</u>, I Balažević, T Hospedales;
  <em> NeurIPS</em>, 2019 <br />
  </p>

  <p><strong>Multi-relational Poincaré Graph Embeddings</strong>
    <a href="https://arxiv.org/abs/1905.09791">[paper]</a> <a href="https://github.com/ibalazevic/multirelational-poincare">[github]</a><br />
  I Balažević, <u>C Allen</u>, T Hospedales;
  <em> NeurIPS</em>, 2019 <br />
  </p>


  <p><strong>Analogies Explained: Towards Understanding Word Embeddings</strong>
    <a href="https://arxiv.org/abs/1901.09813">[paper]</a>
    <a href="https://carl-allen.github.io/nlp/2019/07/01/explaining-analogies-explained.html">[blog post]</a>
    <a href="/assets/Analogies_Explained_slides_ICML.pdf">[slides]</a><br />
  <u>C Allen</u>, T Hospedales;
  <em> ICML</em>, 2019 <strong>(Best Paper, honorable mention)</strong><br />
  </p>

  <p><strong>TuckER: Tensor Factorization for Knowledge Graph Completion</strong>
    <a href="https://arxiv.org/abs/1901.09590">[paper]</a> <a href="https://github.com/ibalazevic/TuckER">[github]</a><br />
  I Balažević, <u>C Allen</u>, T Hospedales;
  <em> EMNLP</em>, 2019 <strong>(oral)</strong> <br />
  </p>

  <p><strong>Hypernetwork Knowledge Graph Embeddings</strong>
    <a href="https://arxiv.org/abs/1808.07018">[paper]</a> <a href="https://github.com/ibalazevic/HypER">[github]</a><br />
  I Balažević, <u>C Allen</u>, T Hospedales;
  <em> ICANN</em>, 2019 <strong>(oral)</strong> <br />
  </p>

---
