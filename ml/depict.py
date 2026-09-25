"""
Step 7 support: compute a molecule's 2D layout (atom positions, bonds) once
in the backend, so the frontend can draw it with plain SVG and never needs
its own chemistry library (e.g. RDKit.js/WASM) just to put a molecule on
screen.

This is deliberately NOT part of featurize.py: it has nothing to do with
model input, it's purely for rendering, and only the API (not training)
ever needs it.
"""

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import AllChem


@dataclass
class AtomLayout:
    index: int  # matches the keys in Explanation.atom_contributions
    element: str
    x: float
    y: float


@dataclass
class BondLayout:
    begin_atom: int
    end_atom: int
    order: float  # 1.0, 1.5 (aromatic), 2.0, 3.0


@dataclass
class MoleculeLayout:
    atoms: list[AtomLayout]
    bonds: list[BondLayout]


def compute_layout(mol: Chem.Mol) -> MoleculeLayout:
    """
    2D coordinates + bonds for one molecule. Operates on a COPY of `mol`
    (Compute2DCoords adds a conformer in place) so this never mutates the
    molecule object shared with featurize.py/explain.py -- atom indices are
    preserved exactly, only a copy gets the added conformer.
    """
    mol = Chem.Mol(mol)
    AllChem.Compute2DCoords(mol)
    conformer = mol.GetConformer()

    atoms = [
        AtomLayout(
            index=atom.GetIdx(),
            element=atom.GetSymbol(),
            x=conformer.GetAtomPosition(atom.GetIdx()).x,
            y=conformer.GetAtomPosition(atom.GetIdx()).y,
        )
        for atom in mol.GetAtoms()
    ]
    bonds = [
        BondLayout(
            begin_atom=bond.GetBeginAtomIdx(),
            end_atom=bond.GetEndAtomIdx(),
            order=bond.GetBondTypeAsDouble(),
        )
        for bond in mol.GetBonds()
    ]
    return MoleculeLayout(atoms=atoms, bonds=bonds)
