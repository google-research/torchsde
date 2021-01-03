from . import ito_additive, ito_diagonal, ito_general, ito_scalar
from . import stratonovich_additive, stratonovich_diagonal, stratonovich_general, stratonovich_scalar

if __name__ == '__main__':
    for module in (ito_additive, ito_diagonal, ito_general, ito_scalar, stratonovich_additive, stratonovich_diagonal,
                   stratonovich_general, stratonovich_scalar):
        module.main()
