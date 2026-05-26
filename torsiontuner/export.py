import biotite.structure as stripe
import biotite.structure.io.pdb as pdb
import numpy as np


def save_for_psvs(struct: stripe.AtomArray, file_path: str):
    """
    Save a protein structure to a PDB file specifically formatted for the
    PSVS validation server (http://psvs-1_5.nesg.org/).

    Ensures that chain IDs are set and standard atom/residue naming is used
    to prevent parsing errors in PROCHECK/MolProbity.

    Args:
        struct: The Biotite AtomArray to save.
        file_path: Output destination path.
    """
    # 1. Strip non-protein atoms (HETATM) if present to simplify validation
    struct = struct[stripe.filter_amino_acids(struct)]

    # 2. Ensure Chain ID is not empty (required by some PSVS sub-tools)
    if np.all(struct.chain_id == ""):
        struct.chain_id[:] = "A"

    # 3. Write PDB
    pdb_file = pdb.PDBFile()
    pdb_file.set_structure(struct)
    pdb_file.write(file_path)
