"""
This file processes the raw excel sheet and extract data
"""

import time
import csv
from collections import defaultdict
from Bio import Entrez
from pathlib import Path

import unicodedata

def _is_whitespace(char):
  """Checks whether `chars` is a whitespace character."""
  # \t, \n, and \r are technically contorl characters but we treat them
  # as whitespace since they are generally considered as such.
  if char == " " or char == "\t" or char == "\n" or char == "\r":
    return True
  cat = unicodedata.category(char)
  if cat == "Zs":
    return True
  return False


def _is_control(char):
  """Checks whether `chars` is a control character."""
  # These are technically control characters but we count them as whitespace
  # characters.
  if char == "\t" or char == "\n" or char == "\r":
    return False
  cat = unicodedata.category(char)
  if cat.startswith("C"):
    return True
  return False

# clean text does not tokenize anything!
def clean_text(text):
    """Performs invalid character removal and whitespace cleanup on text."""
    output = []
    for char in text:
      cp = ord(char)
      if cp == 0 or cp == 0xfffd or _is_control(char):
        continue
      if _is_whitespace(char):
        output.append(" ")
      else:
        output.append(char)
    return "".join(output)

def reduce_whitespace(text):
    return ' '.join(text.split())

major_5_panels = {'experimental-studies', 'allele-data', 'segregation-data', 'specificity-of-phenotype', 'case-control'}
label_vocab = ['experimental-studies', 'allele-data', 'segregation-data', 'specificity-of-phenotype', 'case-control']

class DatasetExtractor(object):
    def __init__(self, path=None):
        self.major_5_pmid_to_panel = defaultdict(set)
        header = None

        if path is not None:
            with open(path, encoding='utf-8', errors='ignore') as f:
                reader = csv.reader(f)
                for i, line in enumerate(reader):
                    if i == 0:
                        header = line[:-2]
                    elif line[4] != '':  # ClinVar ID cannot be null
                        if line[1] in major_5_panels:
                            self.major_5_pmid_to_panel[line[2]].add(line[1])

    def fetch_title_abstract_keywords(self, one_id):
        ids = one_id
        Entrez.email = 'leo.niecn@gmail.com'
        handle = Entrez.efetch(db='pubmed',
                               retmode='xml',
                               id=ids)
        results = Entrez.read(handle)

        # retrieving for only 1 result
        for i, paper in enumerate(results['PubmedArticle']):
            abstract = []
            if 'Abstract' in paper['MedlineCitation']['Article']:
                for section in paper['MedlineCitation']['Article']['Abstract']['AbstractText']:
                    abstract.append(section)
            else:
                continue
            abstract = " ".join(abstract)
            title = paper['MedlineCitation']['Article']['ArticleTitle']
            keywords = []
            for elem in paper['MedlineCitation']['KeywordList']:
                for e in elem:
                    keywords.append(e)

            keywords = ' '.join(keywords)

            return title, abstract, keywords

        return None

    def merge_text(self, title, abstract, keywords, entrez=False):
        # a standard function to map
        text = ''
        if not entrez:
            text = title + " || " + " ".join(keywords.split('/')) + " || " + reduce_whitespace(clean_text(abstract))
        else:
            text = title + " || " + keywords + " || " + reduce_whitespace(clean_text(abstract))
        return text

    def generate_pmid_panel_set(self, log=False, tqdm=False, notebook=False):
        # will call Entrez BioPython to grab abstracts
        data = []
        pmid_to_data = {}

        start = time.time()
        cnt = 0
        for k, v in self.major_5_pmid_to_panel.items():
            cnt += 1
            res = self.fetch_title_abstract_keywords(k)
            if res is None:
                continue  # 24940364 is not found...
            text = self.merge_text(*res)
            # label = ['0'] * len(label_vocab)
            label = []
            for v_i in v:
                label.append(str(label_vocab.index(v_i)))
            data.append('\t'.join([text, ' '.join(label)]))
            pmid_to_data[k] = '\t'.join([text, ' '.join(label)])
            if log:
                if cnt % 100 == 0:
                    print(cnt, time.time() - start, 'secs')

        return data, pmid_to_data

    def write_data_to_csv(self, data, csv_file_path):
        # expect `data` directly from `generate_pmid_panel_set`
        with open(csv_file_path, encoding='utf-8', errors='ignore', mode='w') as f:
            for line in data:
                f.write(line + '\n')

    def write_pmid_to_list(self, path):
        # it will directly save as "pmids.txt", which is what PubMunch expects
        # call this function to generate a list of pmid
        # so you can use PubMunch to download
        p = Path(path)
        p.mkdir(exist_ok=True)
        with open('{}/pmids.txt'.format(path),  'w') as f:
            for pmid in self.major_5_pmid_to_panel.keys():
                f.write(pmid + '\n')

    def __sub__(self, other):
        assert type(other) == type(self)
        new_pmids = set(list(self.major_5_pmid_to_panel.keys())) - set(list(other.major_5_pmid_to_panel))
        de = DatasetExtractor()
        for pmid in new_pmids:
            panel = self.major_5_pmid_to_panel[pmid]
            de.major_5_pmid_to_panel[pmid] = panel

        return de


if __name__ == '__main__':
    # testing
    de = DatasetExtractor("../corpus/ML Data (as of 3_17_19).csv")
    print(de.merge_text(*de.fetch_title_abstract_keywords("10206684")))
