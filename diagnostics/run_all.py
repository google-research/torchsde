import torch

from . import ito_additive, ito_diagonal, ito_general, ito_scalar
from . import stratonovich_additive, stratonovich_diagonal, stratonovich_general, stratonovich_scalar
from . import utils

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float64)
    utils.manual_seed()

    for module in (ito_additive, ito_diagonal, ito_general, ito_scalar, stratonovich_additive, stratonovich_diagonal,
                   stratonovich_general, stratonovich_scalar):
        module.main(device)
