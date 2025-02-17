This is a fork from https://github.com/sfu-discourse-lab/GenderGapTracker to make the quotation extraction part of the pipeline pip installable. 

From `NLP/main` run `pip install .`  to create a package "quotes" which you can import as `from quotes import quote_extractor` and run `quote_extractor("1", doc)` where 1 is an arbitrary doc_id (does not get used but is legacy signature) and doc is a spacy doc from  `en_core_web_lg` (e.g. `nlp = spacy.load("en_core_web_lg"); doc = nlp('some text')`).


__Status: V5.3__ (Code provided as-is; only sporadic updates expected).

# The Gender Gap Tracker Previous

This repo contains the code and framework used in our study on gender bias in the media. We present the [Gender Gap Tracker (GGT)](https://gendergaptracker.informedopinions.org/), an automated system that measures men and women’s voices on
seven major Canadian news outlets in real time. We analyze the rich information in news articles using Natural Language Processing (NLP) and quantify the discrepancy in proportions of men and women quoted. Our larger goals through this project are
to enhance awareness of women’s portrayal in public discourse through hard evidence, and to  encourage news organizations to provide a more diverse set of voices in their reporting.

The Gender Gap Tracker is a collaboration between [Informed Opinions](https://informedopinions.org/), a non-profit dedicated to amplifying women’s voices in media and Simon Fraser University, through the [Discourse Processing Lab](https://www.sfu.ca/discourse-lab.html) and the [Big Data Initiative](https://www.sfu.ca/big-data/big-data-sfu).

## Publications
1. Asr FT, Mazraeh M, Lopes A, Gautam V, Gonzales J, Rao P, Taboada M. (2021) The Gender Gap Tracker: Using Natural Language Processing to measure gender bias in media. *PLoS ONE 16(1): e0245533*. https://doi.org/10.1371/journal.pone.0245533
2. Rao P, Taboada M. (2021), Gender bias in the news: A scalable topic modelling and visualization framework. *Frontiers in Artificial Intelligence, 4(82)*. https://doi.org/10.3389/frai.2021.664737

## Contributors

See [CONTRIBUTORS.md](CONTRIBUTORS.md)

## Contents of this repo

* `dashboard_for_research`: [Research dashboard and apps](https://gendergaptracker.research.sfu.ca/) that allow us to explore the GGT data in more detail.
* `NLP`: NLP modules for performing quote extraction and entity gender annotation on the news articles.
* `scraping_and_aggregation`: Modules for scraping news articles from various Canadian news organizations' websites, and the code we use to extract aggregate statistics from the database.

## Data

The data was downloaded from public and subscription websites of newspapers, under the ‘fair dealing’ provision in Canada’s Copyright Act. This means that the data can be made available (upon signing a licence agreement) **only** for non-commercial and/or research purposes.

## Future directions

In future versions of the software, we are planning to visualize more fine-grained information about who is being quoted, separating politicians, witnesses and/or victims, from experts (as informed sources of analysis, context and opinion). We are also looking into different ways of separating wire copy from the original publication of each news outlet in order to provide a clearer view of the gender gap in Canadian media, produced by the news outlets themselves.

From a research perspective, questions of salience and space arise, i.e., whether quotes by men are presented more prominently in an article, and whether men are given more space in average (perhaps counted in number of words). More nuanced questions that involve language analysis include whether the quotes are presented differently in terms of endorsement or distance from the content of the quote (*stated* vs. *claimed*). Analyses of transitivity structure in clauses can yield further insights about the type of roles women are portrayed in, complementing some of our studies' findings via dependency analyses.

Finally, we are mindful of and acknowledge the relative lack of work in NLP, topic modelling and gender equality for corpora in **languages other than English**. Our hope is that we can at least play a small role here, through our analyses of Canadian French-language news, which we have also been scraping and storing in our database since 2018. We believe that this work will yield not only interesting methodological insights (for example, the relative benefits of stemming vs. lemmatization on topic keyword interpretability for non-English corpora), but also reveal whether the same gender disparities we observed in our English corpus are present in French. While we plan to pursue such additional areas of inquiry, we also invite researchers to join in this effort!


## Contact

For more information about the research methodology and for questions regarding collaboration, please contact Dr. Maite Taboada.

> **Maite Taboada**  
mtaboada@sfu.ca  
Director, Discourse Processing Lab  
Simon Fraser University  
Burnaby, British Columbia, Canada  
