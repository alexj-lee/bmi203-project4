# Import NeedlemanWunsch class and read_fasta function
from align import read_fasta, NeedlemanWunsch


def main():
    """
    This function should
    (1) Align all species to humans and print species in order of most similar to human BRD
    (2) Print all alignment scores between each species BRD2 and human BRD2
    """
    hs_seq, hs_header = read_fasta("./data/Homo_sapiens_BRD2.fa")
    gg_seq, gg_header = read_fasta("./data/Gallus_gallus_BRD2.fa")
    mm_seq, mm_header = read_fasta("./data/Mus_musculus_BRD2.fa")
    br_seq, br_header = read_fasta("./data/Balaeniceps_rex_BRD2.fa")
    tt_seq, tt_header = read_fasta("./data/Tursiops_truncatus_BRD2.fa")

    seqs = (
        (gg_seq, gg_header),
        (mm_seq, mm_header),
        (br_seq, br_header),
        (tt_seq, tt_header),
    )

    species_names = (
        "Chicken (gallus)",
        "Mouse (mus)",
        "Stork (baeleniceps)",
        "Dolphin (tursiops)",
    )

    sub_matrix_file = "./substitution_matrices/BLOSUM62.mat"
    gap_open = -10
    gap_extend = -1
    nw = NeedlemanWunsch(sub_matrix_file, gap_open, gap_extend)

    alignments = []
    for seq, species in zip(seqs, species_names):
        score, hs_aligned, species_aligned = nw.align((hs_seq, hs_header), seq)
        alignments.append((species, score))

    alignments.sort(key=lambda align: align[1], reverse=True)

    print("Species from most to least similar:")
    for species, score in alignments:
        print(f"\t{species}")

    print("\nSpecies scores:")
    for species, score in alignments:
        print(f"\t{species.ljust(25)}| {score}")

    # TODO Align all species to humans and print species in order of most similar to human BRD
    # using gap opening penalty of -10 and a gap extension penalty of -1 and BLOSUM62 matrix
    pass

    # TODO print all of the alignment score between each species BRD2 and human BRD2
    # using gap opening penalty of -10 and a gap extension penalty of -1 and BLOSUM62 matrix
    pass


if __name__ == "__main__":
    main()
