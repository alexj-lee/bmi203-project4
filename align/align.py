# Importing Dependencies
from re import A
import numpy as np
from typing import Tuple

# Defining class for Needleman-Wunsch Algorithm for Global pairwise alignment
class NeedlemanWunsch:
    """Class for NeedlemanWunsch Alignment

    Parameters:
        sub_matrix_file: str
            Path/filename of substitution matrix
        gap_open: float
            Gap opening penalty
        gap_extend: float
            Gap extension penalty

    Attributes:
        seqA_align: str
            seqA alignment
        seqB_align: str
            seqB alignment
        alignment_score: float
            Score of alignment from algorithm
        gap_open: float
            Gap opening penalty
        gap_extend: float
            Gap extension penalty
    """

    def __init__(self, sub_matrix_file: str, gap_open: float, gap_extend: float):
        # Init alignment and gap matrices
        self._align_matrix = None
        self._gapA_matrix = None
        self._gapB_matrix = None

        # Init matrices for backtrace procedure; I found it easier to use one matrix than three for whatever reason
        self._back = None

        # Init alignment_score
        self.alignment_score = 0

        # Init empty alignment attributes
        self.seqA_align = ""
        self.seqB_align = ""

        # Init empty sequences
        self._seqA = ""
        self._seqB = ""

        # Setting gap open and gap extension penalties
        self.gap_open = gap_open
        assert gap_open < 0, "Gap opening penalty must be negative."
        self.gap_extend = gap_extend
        assert gap_extend < 0, "Gap extension penalty must be negative."

        # Generating substitution matrix
        self.sub_dict = self._read_sub_matrix(
            sub_matrix_file
        )  # substitution dictionary

    def _read_sub_matrix(self, sub_matrix_file):
        """
        DO NOT MODIFY THIS METHOD! IT IS ALREADY COMPLETE!

        This function reads in a scoring matrix from any matrix like file.
        Where there is a line of the residues followed by substitution matrix.
        This file also saves the alphabet list attribute.

        Parameters:
            sub_matrix_file: str
                Name (and associated path if not in current working directory)
                of the matrix file that contains the scoring matrix.

        Returns:
            dict_sub: dict
                Substitution matrix dictionary with tuple of the two residues as
                the key and score as value e.g. {('A', 'A'): 4} or {('A', 'D'): -8}
        """
        with open(sub_matrix_file, "r") as f:
            dict_sub = {}  # Dictionary for storing scores from sub matrix
            residue_list = []  # For storing residue list
            start = False  # trigger for reading in score values
            res_2 = 0  # used for generating substitution matrix
            # reading file line by line
            for line_num, line in enumerate(f):
                # Reading in residue list
                if "#" not in line.strip() and start is False:
                    residue_list = [
                        k for k in line.strip().upper().split(" ") if k != ""
                    ]
                    start = True
                # Generating substitution scoring dictionary
                elif start is True and res_2 < len(residue_list):
                    line = [k for k in line.strip().split(" ") if k != ""]
                    # reading in line by line to create substitution dictionary
                    assert len(residue_list) == len(
                        line
                    ), "Score line should be same length as residue list"
                    for res_1 in range(len(line)):
                        dict_sub[(residue_list[res_1], residue_list[res_2])] = float(
                            line[res_1]
                        )
                    res_2 += 1
                elif start is True and res_2 == len(residue_list):
                    break

        return dict_sub

    def align(self, seqA: str, seqB: str) -> Tuple[float, str, str]:
        """
        Uses Needleman-Wunsch algorithm to produce a globally optimal alignment for the two sequences.
        The gaps and scores for the first sequence will be encoded in the rows of align_matrix, gapA_matrix, and gapB_matrix.
        Likewise the columns of the same matrices will encode the information for the second sequence (B).

        Args:
            seqA (str): First sequence that will be aligned.
            seqB (str): Second sequence that will be aligned.

        Returns:
            Tuple[score: float, seqA_aligned: str, seqB_aligned: str]: score, alignment of sequence A, and alignment of sequence B
        """

        # Initialize 6 matrix private attributes for use in alignment
        # create matrices for alignment scores and gaps

        seqA = seqA[0]
        seqB = seqB[0]

        self._align_matrix = np.ones((len(seqA) + 1, len(seqB) + 1)) * -np.inf  # match
        self._gapA_matrix = np.ones((len(seqA) + 1, len(seqB) + 1)) * -np.inf  # ix
        self._gapB_matrix = np.ones((len(seqA) + 1, len(seqB) + 1)) * -np.inf  # iy

        # create matrices for pointers used in backtrace procedure
        self._back = np.ones((len(seqA) + 1, len(seqB) + 1)) * -np.inf
        #        self._back_A = np.ones((len(seqA) + 1, len(seqB) + 1)) * -np.inf
        #        self._back_B = np.ones((len(seqA) + 1, len(seqB) + 1)) * -np.inf

        # Resetting alignment in case method is called more than once
        self.seqA_align = ""
        self.seqB_align = ""

        # Resetting alignment score in case method is called more than once
        self.alignment_score = 0

        # Initializing sequences for use in backtrace method
        self._seqA = seqA
        self._seqB = seqB

        len_seqA = len(seqA)
        len_seqB = len(seqB)

        # initialize the align, gapA, and gapB matrices with the correct column/row values
        iscore = np.arange(len_seqB + 1) * self.gap_extend + self.gap_open
        jscore = np.arange(len_seqA + 1) * self.gap_extend + self.gap_open

        self._align_matrix[0, 0] = 0

        self._gapA_matrix[0, :] = iscore
        self._gapB_matrix[:, 0] = jscore

        #        self._back_B[0, :] = np.inf
        #        self._back_A[:, 0] = np.inf
        self._back[0, :] = np.inf
        self._back[:, 0] = np.inf

        for i in range(1, len_seqA + 1):
            for j in range(1, len_seqB + 1):
                seq_substitution = (seqA[i - 1], seqB[j - 1])
                score = self.sub_dict[seq_substitution]

                self._update_gap("a", i, j)
                self._update_gap("b", i, j)
                self._update_align_matrix(i, j, score)

        return self._backtrace()

    def _update_gap(self, which: str, i: int, j: int):
        """
        Update the given gap matrix (gapA or gapB)

        Args:
            which (str): string flag to update A or B gap matrix
            i (int): row index
            j (int): column index
        """

        if which == "b":  # if we're updating the B matrix, we use i-1 instead of i
            matrix = self._gapB_matrix
            gap = matrix[i - 1, j]
            align = self._align_matrix[i - 1, j]
        else:  # if we're updating the A matrix, we use j-1 instead of j
            matrix = self._gapA_matrix
            gap = matrix[i, j - 1]
            align = self._align_matrix[i, j - 1]

        matrix[i, j] = max(
            gap + self.gap_extend, align + self.gap_open + self.gap_extend
        )

    def _update_align_matrix(self, i: int, j: int, score: float):
        """
        Given the current i, j indices as type int, update the alignment matrix with the i-1, j-1 positions of the score, gapA, and gapB matrices.

        Args:
            i (int): row index
            j (int): column index
            score (float): replacement score from substitution matrix for i-1, j-1 entries in sequences.

        """

        # grab the i-1, j-1 vales in the three scoring/gap matrices
        m = self._align_matrix[i - 1, j - 1]
        iy = self._gapA_matrix[i - 1, j - 1]
        ix = self._gapB_matrix[i - 1, j - 1]
        neighbor_values = (m, ix, iy)

        argmax = np.argmax(neighbor_values)

        self._align_matrix[i, j] = neighbor_values[argmax] + score

        self._back[i, j] = argmax

    def _backtrace(self) -> Tuple[float, str, str]:
        """
        Use the traceback matrix self._back to quickly recover the optimal score path.
        For this we start at the last position (-1, -1) and read off the value at that site; we also initialize a variable to keep track of the current and previous positions.

        From the current position value, we read off whether to move diagonally (value == 0), left (value == 1), or up (value == 2).
        From the previous position value, we read off whether to  add the i-1, j-1 sequence position to both sequences (value == 0), gap in B (value == 1), or gap in A (value == 2)
        """

        seqA = self._seqA
        seqB = self._seqB
        j, i = len(seqB), len(seqA)

        alignment_score = self._align_matrix[-1, -1]
        # variable initialization
        seqA_align = ""
        seqB_align = ""

        # initialize state variables
        prev_trace = self._back[i, j]
        current_trace = self._back[i, j]

        while i > 0 or j > 0:  # keep going until we hit the end (given by 0, 0)
            if prev_trace == 0:  # diagonal
                seqA_align += seqA[i - 1]
                seqB_align += seqB[j - 1]
            elif prev_trace == 1:  # gap in B
                seqA_align += seqA[i - 1]
                seqB_align += "-"
            elif prev_trace == 2:  # gap in A
                seqA_align += "-"
                seqB_align += seqB[j - 1]

            if current_trace == 0:  # update next position diagonally
                i -= 1
                j -= 1
            elif current_trace == 1:  # update next position left
                i -= 1
            elif current_trace == 2:  # update next position up
                j -= 1

            # update state variables
            prev_trace = current_trace
            current_trace = self._back[i, j]

        seqA_align = seqA_align[::-1]
        seqB_align = seqB_align[::-1]
        return (alignment_score, seqA_align, seqB_align)


def read_fasta(fasta_file: str) -> Tuple[str, str]:
    """
    DO NOT MODIFY THIS FUNCTION! IT IS ALREADY COMPLETE!

    This function reads in a FASTA file and returns the associated
    string of characters (residues or nucleotides) and the header.
    This function assumes a single protein or nucleotide sequence
    per fasta file and will only read in the first sequence in the
    file if multiple are provided.

    Parameters:
        fasta_file: str
            name (and associated path if not in current working directory)
            of the Fasta file.

    Returns:
        seq: str
            String of characters from FASTA file
        header: str
            Fasta header
    """
    assert fasta_file.endswith(
        ".fa"
    ), "Fasta file must be a fasta file with the suffix .fa"
    with open(fasta_file) as f:
        seq = ""  # initializing sequence
        first_header = True
        for line in f:
            is_header = line.strip().startswith(">")
            # Reading in the first header
            if is_header and first_header:
                header = line.strip()  # reading in fasta header
                first_header = False
            # Reading in the sequence line by line
            elif not is_header:
                seq += line.strip().upper()  # generating full sequence
            # Breaking if more than one header is provided in the fasta file
            elif is_header and not first_header:
                break
    return seq, header
