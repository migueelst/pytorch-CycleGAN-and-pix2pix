from .test_options import TestOptions


class EvalOptions(TestOptions):
    """This class includes evaluation options.

    It also includes shared options defined in BaseOptions & TestOptions.
    """

    def initialize(self, parser):
        parser = TestOptions.initialize(self, parser)  # define shared options
        # TODO: add some config options here, otherwise delete this class and just keep test_options
        return parser
