import pubchempy as pcp
import pandas as pd
from rdkit import Chem
from mol2chemfigPy3 import mol2chemfig




def convert_smiles(smiles):
    try:
        compounds = pcp.get_compounds(smiles, namespace='smiles')
        c = compounds[0]
        input_smiles = smiles
        sum_formula = c.molecular_formula
        iupac_name = c.iupac_name
        isomeric_smiles = c.isomeric_smiles
        #canonical_smiles = match.canonical_smiles #Canonical SMILES, with no stereochemistry information.
        inchi = c.inchi
        inchikey = c.inchikey
        #fingerprint = match.fingerprint
        synonyms = c.synonyms
        #Because PubChem only support isomeric SMILES, So we generate canonical SMILES from RDKit.
        mol_inchi = Chem.inchi.MolFromInchi(inchi)
        smiles_cano = Chem.MolToSmiles(mol_inchi, canonical=True, isomericSmiles=False)

        h_bond_acceptor_count = c.h_bond_acceptor_count
        #TODO
        #not here, but in UI,only show 5 items, and click "more" to show all items.

        print('CID in PubChem:',compounds)
        print('Input SMILES:',smiles)
        print('IUPAC Name:',iupac_name)
        print('Sum Formula:',sum_formula)
        print('Isomeric SMILES:',isomeric_smiles)
        print('Canonical SMILES:',smiles_cano)
        print('InChI:', inchi)
        print('InChIKeys', inchikey)
        print('Chemfig:')
        print(mol2chemfig(inchi))
        print('elements', c.elements)
        print("This chemical structure contains {0} ements {1} atom ".format(h_bond_acceptor_count,h_bond_acceptor_count))
        #rint('molecular_weight', match.molecula_weight)
        print('h_bond_donor_count', c.h_bond_donor_count)
        print('atoms:', c.atoms)
        print('bonds:', c.bonds)
        print('h_bond_acceptor_count', match.h_bond_acceptor_count)
        #print('Fingerprint:', fingerprint)
        print('Synonyms:',synonyms)
        print('='*200)

    except:
        print("Invalid SMILES: This SMILES string doesn't exist in PubChem library.")


# def chemfig(mol):
#     c1 = mol2chemfig(mol)  # search the PubChem database
#      # transfer CID/InChI/SMILES to chemfig


s = 'C1=CC=CC=C1'
convert_smiles(s)

def elements_num(elements):
    ls = {}
    n = 0
    count = 1
    for idx in range(len(elements)-1):
        #if idx < len(elements)-1:
        ele = elements[idx]
        print(ele)

        if elements[idx+1] != elements[idx] or idx == len(elements) - 2:
            ls.update({ele: count})
            count = 1
        else:
            count += 1


    #         else:
    #             pass


    print(ls)
