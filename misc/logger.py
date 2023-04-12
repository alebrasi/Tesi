from enum import Enum


# class Bcolors:
#     HEADER = '\033[95m'
#     OKBLUE = '\033[94m'
#     OKCYAN = '\033[96m'
#     OKGREEN = '\033[92m'
#     WARNING = '\033[93m'
#     FAIL = '\033[91m'
#     ENDC = '\033[0m'
#     BOLD = '\033[1m'
#     UNDERLINE = '\033[4m'


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class Logger(metaclass=SingletonMeta):
    class LogLevel(Enum):
        SEED_LINE = 0,
        POST_PROCESS = 1,
        REGION_BELOW = 2,
        LOCATE_SEEDS = 3,
        BASIC = 4,
        NONE = 5,
        ALL = 6

        def __str__(self):
            return self.name

    def __init__(self, log_level):
        self._logLevel = log_level

    def __call__(self, *args, **kwargs):
        print(args[0])

    def log(self, val, log_level):
        if self._logLevel == log_level or self._logLevel == self.LogLevel.ALL:
            print(f'[{log_level}] {val}')
