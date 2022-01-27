# Importing Dependencies
import pytest
from align import NeedlemanWunsch, read_fasta
import numpy as np
import pathlib


@pytest.fixture
def substitution_matrix_path():
    return (
        pathlib.Path(__file__).resolve().parent.parent
        / "substitution_matrices"
        / "BLOSUM62.mat"
    )


def test_nw_alignment(substitution_matrix_path):
    """
    TODO: Write your unit test for NW alignment
    using test_seq1.fa and test_seq2.fa by
    asserting that you have correctly filled out
    the your 3 alignment matrices.
    Use the BLOSUM62 matrix and a gap open penalty
    of -10 and a gap extension penalty of -1.
    """
    seq1, _ = read_fasta("./data/test_seq1.fa")
    seq2, _ = read_fasta("./data/test_seq2.fa")
    nw = NeedlemanWunsch(substitution_matrix_path, -10, -1)
    out = nw.align((seq1, _), (seq2, _))

    len_seq1 = len(seq1)
    len_seq2 = len(seq2)

    assert nw._align_matrix[0, 0] == 0, "Initial position (0,0) of matrix must be 0."

    # fmt: off
    assert (np.isinf(nw._align_matrix[0, 1:]).sum() == len_seq2) and (np.isinf(nw._gapB_matrix[0, 1:]).sum() == len_seq2), "First row of align and gapA matrix must be infs with same length as seq2"
    assert (np.isinf(nw._align_matrix[1:, 0]).sum() == len_seq1) and (np.isinf(nw._gapA_matrix[1:, 0]).sum() == len_seq1), "First row of align and gapA matrix must be infs with same length as seq2"
    # fmt: on

    # for this part of the test we will randomly sample half the matrix values and then test whether the assignment was correct following the rules for M, Ix, and Iy
    num_positions_to_sample = (len_seq1 * len_seq2) // 2
    random_j_inds = np.random.randint(1, len(seq2), size=num_positions_to_sample)
    random_i_inds = np.random.randint(1, len(seq1), size=num_positions_to_sample)

    for i, j in zip(random_i_inds, random_j_inds):
        assert (
            nw._align_matrix[i, j]
            == max(
                nw._align_matrix[i - 1, j - 1],
                nw._gapA_matrix[i - 1, j - 1],
                nw._gapB_matrix[i - 1, j - 1],
            )
            + nw.sub_dict[(seq1[i - 1], seq2[j - 1])]
        ), f"Score matrix m was not properly assigned to position ({i}, {j})"

        assert nw._gapA_matrix[i, j] == max(
            nw._align_matrix[i, j - 1] - 11, nw._gapA_matrix[i, j - 1] - 1
        ), f"GapA matrix was not properly assigned at position ({i}, {j})"

        assert nw._gapB_matrix[i, j] == max(
            nw._align_matrix[i - 1, j] - 11, nw._gapB_matrix[i - 1, j] - 1
        ), f"GapB matrix was not properly assigned at position ({i}, {j})"


def test_nw_backtrace(substitution_matrix_path):
    """
    TODO: Write your unit test for NW backtracing
    using test_seq3.fa and test_seq4.fa by
    asserting that the backtrace is correct.
    Use the BLOSUM62 matrix. Use a gap open
    penalty of -10 and a gap extension penalty of -1.
    """
    seq3, _ = read_fasta("./data/test_seq3.fa")
    seq4, _ = read_fasta("./data/test_seq4.fa")

    nw = NeedlemanWunsch(substitution_matrix_path, -10, -1)
    out = nw.align((seq3, _), (seq4, _))

    len_seq3 = len(seq3)
    len_seq4 = len(seq4)

    num_positions_to_sample = (len_seq3 * len_seq4) // 2

    random_j_inds = np.random.randint(1, len(seq4), size=num_positions_to_sample)
    random_i_inds = np.random.randint(1, len(seq3), size=num_positions_to_sample)

    for i, j in zip(random_i_inds, random_j_inds):
        assert nw._back[i, j] == np.argmax(
            (
                nw._align_matrix[i - 1, j - 1],
                nw._gapB_matrix[i - 1, j - 1],
                nw._gapA_matrix[i - 1, j - 1],
            )
        )

    assert np.unique(nw._back).size <= 4  # only inf,, 0, 1, 2 values
    assert np.isinf(nw._back[0, :]).sum() == len_seq4 + 1
    assert np.isinf(nw._back[:, 0]).sum() == len_seq3 + 1
