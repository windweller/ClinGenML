"""
We write this to comb through the AlleleRegistry pipeline
"""

import time
import json
import requests
from tqdm import tqdm

class ACMGDownloader(object):
    def __init__(self):
        self.acmg_cancer_gens = {'APC', 'BRCA1', 'BRCA2', 'TP53', 'MLH1', 'MSH2', 'RET', 'PTEN', 'RB1'}
        # cardiovascular diseases
        self.acmg_cvd_gens = {'DSP', 'PKP2', 'FBN1', 'LMNA', 'MYH7', 'MYBPC3', 'KCNQ1', 'SCN5A', 'LDLR'}

        self.gene_variants_map = {}

        self.variant_lookup_url = "https://reg.genome.network/alleles?gene={}&fields=none+@id+externalRecords.dbSNP.rs"

    def query_variants(self, gene_name):
        url = self.variant_lookup_url.format(gene_name)
        res = requests.get(url)
        res = json.loads(res.text)

        if len(res) == 0:
            return None

        total_registry_record_count = len(res)

        allele_id_to_rs_ids = {}
        for row in res:
            if 'externalRecords' in row:
                rs_id = row['externalRecords']['dbSNP'][0]['rs']
                allele_id_to_rs_ids[row['@id']] = rs_id

        return allele_id_to_rs_ids, total_registry_record_count

    def query_all_genes(self):
        for gene_name in tqdm(self.acmg_cancer_gens | self.acmg_cvd_gens, total=len(self.acmg_cancer_gens)+len(self.acmg_cvd_gens)):
            allele_id_to_rs_id_map, total_rec_count = self.query_variants(gene_name)
            self.gene_variants_map[gene_name] = allele_id_to_rs_id_map

    def save_to_disk(self, path, indent=None):
        json.dump(self.gene_variants_map, open(path, 'w'), indent=indent)

    def query_rsid_from_litvar(self, rsid, loop=0):
        if loop > 10:
            return None

        url = "https://www.ncbi.nlm.nih.gov/research/bionlp/litvar/api/v1/public/rsids2pmids?rsids=rs{}".format(rsid)
        res = requests.get(url)

        if res.status_code != 200:
            # then the website has some problem
            time.sleep(60)  # we wait 1 minute
            return self.query_rsid_from_litvar(rsid, loop+1)

        res = json.loads(res.text)

        if len(res) == 0:
            return None

        if 'pmids' in res[0]:
            pmids = res[0]['pmids']
            return pmids
        else:
            return None

    def query_from_litvar(self, pmid_save_path=None, rsid_pmid_map_save_path=None):
        rsids = set()
        pmids = set()

        rsid_pmids = {}  # this map is important later on

        not_found = 0

        for allele_id_to_rsid_dic in self.gene_variants_map.values():
            rsids = rsids | set(list(allele_id_to_rsid_dic.values()))

        print("we have ", len(rsids), "unique rsid in total")
        for rsid in tqdm(rsids, total=len(rsids)):
            new_pmids = self.query_rsid_from_litvar(rsid)
            rsid_pmids[rsid] = new_pmids

            if new_pmids is not None:
                pmids = pmids | set(new_pmids)
            else:
                not_found += 1

        print("we get {} unique papers from PubMed".format(len(pmids)))
        print("{} rsid are not found".format(not_found))

        if pmid_save_path is not None:
            with open(pmid_save_path, 'w') as f:
                for pmid in pmids:
                    f.write(str(pmid) + '\n')

        if rsid_pmid_map_save_path is not None:
            json.dump(rsid_pmids, open(rsid_pmid_map_save_path, 'w'))

        return pmids, rsid_pmids


if __name__ == '__main__':

    downloader = ACMGDownloader()
    downloader.query_all_genes()
    downloader.query_from_litvar(pmid_save_path="../corpus/acmg_pmids.txt",
                                 rsid_pmid_map_save_path="../corpus/acmg_rsid_pmid_map.json")
    # 10821 --> before we fixed the error
