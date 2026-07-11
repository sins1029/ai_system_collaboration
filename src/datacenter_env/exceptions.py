class DataCenterError(Exception):
    """Base exception for the public package API."""


class ConfigurationError(DataCenterError):
    pass


class InputValidationError(DataCenterError):
    pass


class TimeAlignmentError(InputValidationError):
    pass


class ControllerError(DataCenterError):
    pass


class OptimizerError(ControllerError):
    pass


class StorageError(DataCenterError):
    pass


class RunStateError(DataCenterError):
    pass
