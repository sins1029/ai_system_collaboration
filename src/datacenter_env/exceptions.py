class DataCenterError(Exception):
    """公共包 API 的基础异常。"""


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


class SchedulingError(DataCenterError):
    pass


class UnschedulableTaskError(SchedulingError):
    pass
