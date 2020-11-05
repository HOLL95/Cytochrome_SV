import re
sequence = "GTGCTCAATGGATAATACTGAGCTCGAGGTGGACTTCTATAGTTGCGTACACTCGATGAC"
def transcribe(matchobj):
    nucleotide_dict={"C":"G", "G":"C", "A":"U", "T":"A"}
    return nucleotide_dict[matchobj.group(0)]

def reverse_comp(text):
    return re.sub(r'[ACGT]', transcribe, text)


print(sequence)
print(reverse_comp(sequence))
