class GenxBaseError(Exception):
    pass

class GenxModelLoadError(GenxBaseError):
    pass

class GenxModelNotFoundError(GenxBaseError):
    pass