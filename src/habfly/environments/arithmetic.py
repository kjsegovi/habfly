from .curriculum import CurriculumEnvironment


class ArithmeticEnvironment(CurriculumEnvironment):
    def __init__(self, **kwargs):
        super().__init__(stage="arithmetic", **kwargs)
