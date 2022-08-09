import argparse

from utils import str2bool, convert_smiles

def parse_args():
    parser = argparse.ArgumentParser()

    # generate more informatin for one sample test
    parser.add_argument('--others', default=True, type=str2bool,
                        help='Should we predict other chemical representations from SMILES?')

    config = parser.parse_args()

    return config


if __name__ == '__main__':
    # Get configuration
    config = vars(parse_args())
    # why we need this command?

    if config['others']==True:
        print('Carrying out generating other chemical representations')
        convert_smiles(smiles='C1=CC(=C(C(=C1)O)N)C(=O)O')
