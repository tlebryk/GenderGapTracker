"""
FOR MY USE: 
We want to use mergenes as is use by update_existing_collection
except we need to escape the collection abstraction [comment out post collection stuff. ]


"""
import logging
import re
from itertools import chain

import neuralcoref
import pandas as pd
import spacy
from spacy.pipeline import EntityRuler
from thesisutils import utils as ut
from tqdm import tqdm

import quote_extractor
import utils
from config import config

app_logger = utils.create_logger('entity_gender_annotator_logger', log_dir='logs', logger_level=logging.INFO, file_log_level=logging.INFO)
# %%
MONGO_ARGS = config['MONGO_ARGS']
AUTHOR_BLOCKLIST = config['NLP']['AUTHOR_BLOCKLIST']
NAME_PATTERNS = config['NLP']['NAME_PATTERNS']
MAX_BODY_LENGTH = config['NLP']['MAX_BODY_LENGTH']
# %%
print('Loading spaCy language model...')
nlp = spacy.load('en_core_web_lg')
# Add custom named entity rules for non-standard person names that spaCy doesn't automatically identify
ruler = EntityRuler(nlp, overwrite_ents=True).from_disk(NAME_PATTERNS)
nlp.add_pipe(ruler)
print('Finished loading.')

coref = neuralcoref.NeuralCoref(nlp.vocab, max_dist=200)
nlp.add_pipe(coref, name='neuralcoref')


ner_filter = [
    "DATE",
    "WORK_OF_ART",
    "PERCENT",
    "QUANTITY",
    "TIME",
    "MONEY",
    # "LAW",
    "LANGUAGE"
    "ORDINAL",
    "CARDINAL",

]

# ========== Named Entity Merging functions ==========

# merge nes is a two step unification process:
# 1- Merge NEs based on exact match
# 2- merge NEs based on partial match
def merge_nes(doc_coref):
    # ne_dict and ne_cluster are dictionaries which keys are PERSON named entities extracted from the text and values
    #  are mentions of that named entity in the text. Mention clusters come from coreference clustering algorithm.
    ne_dict = {}
    ne_clust = {}
    # It's highly recommended to clean nes before merging them. They usually contain invalid characters
    person_nes = [x for x in doc_coref.ents if x.label_ == 'PERSON']
    # in this for loop we try to merge clusters detected in coreference clustering

    # ----- Part A: assign clusters to person named entities
    for ent in person_nes:
        # Sometimes we get noisy characters in name entities
        # TODO: Maybe it's better to check for other types of problems in NEs here too

        ent_cleaned = utils.clean_ne(str(ent))
        if (len(ent_cleaned) == 0) or utils.string_contains_digit(ent_cleaned):
            continue

        ent_set = set(range(ent.start_char, ent.end_char))
        found = False
        # if no coreference clusters is detected in the document
        if doc_coref._.coref_clusters is None:
            ne_dict[ent] = []
            ne_clust[ent] = -1

        else:
            for cluster in doc_coref._.coref_clusters:
                for ment in cluster.mentions:
                    ment_set = set(range(ment.start_char, ment.end_char))
                    if has_coverage(ent_set, ment_set):
                        ne_dict[ent] = cluster
                        ne_clust[ent] = cluster.i

                        found = True
                        break
                # End of for on mentions
                if found:
                    break

            # End of for on clusters

            if not found:
                ne_dict[ent] = []
                ne_clust[ent] = -1

    # ----- Part B: Merge clusters in ne_dict based on exact match of their representative (PERSON named entities)
    merged_nes = {}
    for ne, cluster in zip(ne_dict.keys(), ne_dict.values()):

        ne_clean_text = utils.clean_ne(str(ne))

        if not cluster:
            cluster_id = [-1]
            mentions = []
        else:
            cluster_id = [cluster.i]
            mentions = cluster.mentions

        # check if we already have a unique cluster with same representative
        if ne_clean_text in merged_nes.keys():
            retrieved = merged_nes[ne_clean_text]
            lst = retrieved['mentions']
            lst = lst + [ne] + mentions
            cls = retrieved['cluster_id']
            cls = cls + cluster_id
            merged_nes[ne_clean_text] = {'mentions': lst, 'cluster_id': cls}
        else:
            tmp = [ne] + mentions
            merged_nes[ne_clean_text] = {'mentions': tmp, 'cluster_id': cluster_id}

    # ----- Part C: do a complex merge
    complex_merged_nes, changed = complex_merge(merged_nes)

    return complex_merged_nes


# This is the last try to merge named entities based on multi-part ne merge policy
def complex_merge(ne_dict):
    merged_nes = {}
    changed = {}
    for ne in ne_dict.keys():
        found = False
        for merged in merged_nes.keys():
            if can_merge_nes(str(ne), str(merged)):
                if len(ne) > len(merged):
                    merged_nes[ne] = merged_nes[merged] + ne_dict[ne]['mentions']
                    changed[ne] = 1
                    del merged_nes[merged]
                elif len(ne) < len(merged):
                    changed[merged] = 1
                    merged_nes[merged] = merged_nes[merged] + ne_dict[ne]['mentions']

                found = True

                break
        if not found:
            changed[ne] = 0
            merged_nes[ne] = ne_dict[ne]['mentions']

    return merged_nes, changed


def has_coverage(s1, s2):
    return len(s1.intersection(s2)) >= 2


# This function checks whether we can do maltipart merge for two named entities.
def can_merge_nes(ne1, ne2):
    can_merge = False
    # To get rid of \n and empty tokens
    ne1 = ne1.strip()
    ne2 = ne2.strip()
    if len(ne1) > len(ne2):
        ne_big = ne1
        ne_small = ne2
    else:
        ne_big = ne2
        ne_small = ne1

    ne_big = ne_big.split(' ')
    ne_small = ne_small.split(' ')

    # Check for merging a two part name with a one part first name
    if len(ne_big) == 2 and len(ne_small) == 1:
        first_name_match = (ne_big[0] == ne_small[0]) and \
                            ne_big[0][0].isupper() and \
                            ne_small[0][0].isupper() and \
                            ne_big[1][0].isupper()

        can_merge = first_name_match
    # Check for merging a three part and a two part
    elif len(ne_big) == 3 and len(ne_small) == 2:
        last_middle_name_match = (ne_big[-1] == ne_small[-1]) and \
                                 (ne_big[-2] == ne_small[-2]) and \
                                  ne_big[0][0].isupper() and \
                                  ne_big[1][0].isupper() and \
                                  ne_big[2][0].isupper()
        can_merge = last_middle_name_match
    # Check for merging a three part and a one part
    elif len(ne_big) == 3 and len(ne_small) == 1:
        last_name_match = (ne_big[-1] == ne_small[-1]) and \
                           ne_big[-1][0].isupper() and \
                           ne_big[0][0].isupper()

        can_merge = last_name_match

    app_logger.debug('ne1: {0}\tne2: {1}\tComplex Merge Result: {2}'.format(ne1, ne2, can_merge))

    return can_merge


def remove_invalid_nes(unified_nes):
    final_nes = {}
    for key, value in zip(unified_nes.keys(), unified_nes.values()):
        # to remove one part NEs after merge
        # Todo: should only remove singltones?
        representative_has_one_token = (len(key.split(' ')) == 1)
        key_is_valid = not (representative_has_one_token)
        if key_is_valid:
            final_nes[key] = value

    return final_nes


def get_named_entity(doc_coref, span_start, span_end):
    span_set = set(range(span_start, span_end))

    for x in doc_coref.ents:
        x_start = x.start_char
        x_end = x.end_char
        x_set = set(range(x_start, x_end))
        if has_coverage(span_set, x_set):
            return str(x), x.label_

    return None, None


# This function assignes quotes to nes based on overlap of quote's speaker span and the names entity span
def quote_assign(nes, quotes, doc_coref):
    quote_nes = {}
    quote_no_nes = []
    index_finder_pattern = re.compile(r'.*\((\d+),(\d+)\).*')

    aligned_quotes_indices = []

    for q in quotes:
        regex_match = index_finder_pattern.match(q['speaker_index'])
        q_start = int(regex_match.groups()[0])
        q_end = int(regex_match.groups()[1])
        q_set = set(range(q_start, q_end))

        quote_aligned = False
        # search in all of the named entity mentions in it's cluster for the speaker span.
        for ne, mentions in zip(nes.keys(), nes.values()):
            if quote_aligned:
                break
            for mention in mentions:
                mention_start = mention.start_char
                mention_end = mention.end_char
                mention_set = set(range(mention_start, mention_end))

                if has_coverage(q_set, mention_set):
                    alignment_key = '{0}-{1}'.format(q_start, q_end)
                    aligned_quotes_indices.append(alignment_key)
                    q['is_aligned'] = True
                    q['named_entity'] = str(ne)
                    q['named_entity_type'] = 'PERSON'
                    quote_aligned = True

                    if ne in quote_nes.keys():
                        current_ne_quotes = quote_nes[ne]
                        current_ne_quotes.append(q)
                        quote_nes[ne] = current_ne_quotes
                    else:
                        quote_nes[ne] = [q]

                    break  # Stop searching in mentions. Go for next quote

        if not quote_aligned:
            q['is_aligned'] = False
            ne_text, ne_type = get_named_entity(doc_coref, q_start, q_end)
            if ne_text is not None:
                q['named_entity'] = ne_text
                q['named_entity_type'] = ne_type
            else:
                q['named_entity'] = ''
                q['named_entity_type'] = 'UNKNOWN'

            quote_no_nes.append(q)

    all_quotes = []
    for ne, q in zip(quote_nes.keys(), quote_nes.values()):
        all_quotes = all_quotes + q

    all_quotes = all_quotes + quote_no_nes

    return quote_nes, quote_no_nes, all_quotes

def test_eq(idx):
    doc = traindf.iloc[idx]
    q1, ner1 = parse_doc_new(doc, pub, True)
    q2, ner2 = parse_doc_new(doc, pub, False)
    if q1 != q2:
        print("SOMETHING WRONG quotes", doc.Art_id)
        return None
    if ner1 != ner2:
        print("SOMETHING WRONG ner", doc.Art_id)
        return None
    print("all good ", idx)

# for i in range(0, -20, -1):
#     print(i)
#     test_eq(idx=i)


def parse_doc_new(doc, pub, test=True):
    """THEO DEFINIED FN. 
    
    """
    # quotes = quote_extractor.parse_doc()
    doc_id = str(doc['Art_id'])
    text = doc['Body']
    text_preprocessed = utils.preprocess_text(text)
    # doc_coref = ut.timeit(nlp, text_preprocessed)
    # Going with these disabled after testing found no difference with full pipeline
    if test:
        doc_coref = nlp(text_preprocessed, disable=[ "tok2vec",  "attribute_ruler", "lemmatizer"]) #"parser","tagger",
    else:
        doc_coref = nlp(text_preprocessed)
    ents = doc_coref.ents
    dct_ls = []
    for ent in ents:
        if ent.label_ not in ner_filter:
            dct = {

                "entity" : ent.text,
                "label_" : ent.label_,
                # "label" : ent.label,
                "start" : ent.start,
                "end" : ent.end,
                "start_char" : ent.start_char,
                "end_char" : ent.end_char,
                "Art_id": doc["Art_id"],
                "publication": pub.name,
                # "year": year,
            }
            dct_ls.append(dct)
    # return dct_ls
    unified_nes = merge_nes(doc_coref)
    final_nes = remove_invalid_nes(unified_nes)

    quotes = quote_extractor.extract_quotes(doc_id=doc_id, doc=doc_coref, write_tree=False, pubname=pub.name)
    # all_quotes is list of dictionaries of quotes with named_entity and named_entity_type.
    _, _, all_quotes = quote_assign(final_nes, quotes, doc_coref)
    return all_quotes, dct_ls

def gen_ner_quotes(pub, df):
    quotes, ners = [], []
    df.Body = df.Body.astype(str).str[:10000]
    with tqdm(total=df.shape[0]) as pbar: 
        for i, row in df.iterrows():
            pbar.update(1)
            quote, ner = parse_doc_new(row[['Art_id', 'Body', "Publication"]], pub)
            quotes.append(quote)
            ners.append(ner)
    ners_unnest = chain(*ners)
    nerdf = pd.DataFrame(ners_unnest)
    quotes_unnest = chain(*quotes)
    quotedf = pd.DataFrame(quotes_unnest)
    quotedf.index = quotedf.index.set_names(["quid"])
    quotedf = quotedf.reset_index()
    nerdf.index = nerdf.index.set_names(["ner_index"])
    nerdf = nerdf.reset_index()
    return nerdf, quotedf
# NERS = []
# QUOTES = []
# def parse_doc_global(doc, pub):
#     """THEO DEFINIED FN. 
    
#     """
#     global NERS, QUOTES
#     # quotes = quote_extractor.parse_doc()
#     doc_id = str(doc['Art_id'])
#     text = doc['Body']
#     text_preprocessed = utils.preprocess_text(text)
#     # doc_coref = ut.timeit(nlp, text_preprocessed)
#     doc_coref = nlp(text_preprocessed)
#     ents = doc_coref.ents
#     dct_ls = []
#     for ent in ents:
#         if ent.label_ not in ner_filter:
#             dct = {

#                 "entity" : ent.text,
#                 "label_" : ent.label_,
#                 # "label" : ent.label,
#                 "start" : ent.start,
#                 "end" : ent.end,
#                 "start_char" : ent.start_char,
#                 "end_char" : ent.end_char,
#                 pub.uidcol: doc["Art_id"],
#                 "publication": pub.name,
#                 # "year": year,
#             }
#             NERS.append(dct)
#     # return dct_ls
#     unified_nes = merge_nes(doc_coref)
#     final_nes = remove_invalid_nes(unified_nes)

#     quotes = quote_extractor.extract_quotes(doc_id=doc_id, doc=doc_coref, write_tree=False, pubname=pub.name)
#     # all_quotes is list of dictionaries of quotes with named_entity and named_entity_type.
#     nes_quotes, quotes_no_nes, all_quotes = quote_assign(final_nes, quotes, doc_coref)
#     QUOTES += all_quotes
#     # return all_quotes, dct_ls


if __name__ == '__main__':
    pub = ut.publications['nyt']
    # need some flexibility bc might need to read in yearly dfs
    # take a shortcut and cut really long articles
    df = ut.standardize(ut.read_df_s3(None, pubdefault=pub) , pub)
    tts = "train"
    train = ut.read_df_s3(f"{pub}/tts_mask/{tts}_main1.csv")
    train = ut.standardize(train, pub)
    trainmask = df.Art_id.isin(train.Art_id)
    traindf = ut.drop_report(df,trainmask)
    tts = "test"
    test = ut.read_df_s3(f"{pub}/tts_mask/{tts}_main1.csv")
    test = ut.standardize(test, pub)
    testmask = df.Art_id.isin(test.Art_id)
    testdf = ut.drop_report(df,testmask)

    tts = "train"
    # just remember to multi index later on. 
    nerdf, quotedf = gen_ner_quotes(pub, traindf)
    ut.df_to_s3(nerdf, f"{pub.name}/ner/ner_{tts}2.csv")
    ut.df_to_s3(quotedf, f"{pub.name}/quotes/quotes_{tts}2.csv")

    tts = "test"
    # just remember to multi index later on. 
    nerdf, quotedf = gen_ner_quotes(pub, testdf)
    ut.df_to_s3(nerdf, f"{pub.name}/ner/ner_{tts}2.csv")
    ut.df_to_s3(quotedf, f"{pub.name}/quotes/quotes_{tts}2.csv")




# def globaldict_way():
#     NERS = []
#     QUOTES = []
#     for i, row in df.iterrows():
#         print(i)
#         parse_doc_global(row[['Art_id', 'Body', "Publication"]], pub)
#         if i > 4: 
#             break
#     y = pd.DataFrame(NERS)
#     z = pd.DataFrame(QUOTES)
#     return y, z
# def globaldict_apply():
#     NERS = []
#     QUOTES = []
#     df[['Art_id', 'Body', "Publication"]].head().apply(parse_doc_global, pub=pub, axis=1)
#     y = pd.DataFrame(NERS)
#     z = pd.DataFrame(QUOTES)
#     return y, z

# y, z = globaldict_apply()
# y, z = globaldict_way()
# %timeit  -r 5 globaldict_apply()
# %timeit  -r 5 globaldict_way()
# %timeit -r 5 iterrow_way()