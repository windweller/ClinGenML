"""
The goal is to find the roughly 300 new PMIDs that we can use as a disjoint
test set.
"""
# import csv
#
# old_pmid = set()
# with open("../corpus/ML Data (as of 3_17_19).csv", encoding='utf-8', errors='ignore') as f:
#     csv_reader = csv.reader(f)
#     for i, line in enumerate(csv_reader):
#         if i == 0:
#             continue
#         old_pmid.add(line[2])
#
# new_pmid = set()
# with open("../corpus/ML Data (as of 5_1_19).csv", encoding='utf-8', errors='ignore') as f:
#     csv_reader = csv.reader(f)
#     for i, line in enumerate(csv_reader):
#         if i == 0:
#             continue
#         new_pmid.add(line[2])
#
# print(len(new_pmid - old_pmid))

from data.clingen_raw_to_training import DatasetExtractor

old = DatasetExtractor("../corpus/ML Data (as of 3_17_19).csv")
new = DatasetExtractor("../corpus/ML Data (as of 5_1_19).csv")
de = new - old

print(len(de.major_5_pmid_to_panel))

data, _ = de.generate_pmid_panel_set(log=True)

de.write_data_to_csv(data, "../models/data/vci_358_abs_tit_key_may_7_2019_true_test.csv")