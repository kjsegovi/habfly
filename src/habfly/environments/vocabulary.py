from .curriculum import CurriculumEnvironment


class VocabularyEnvironment(CurriculumEnvironment):
    def __init__(self, **kwargs):
        super().__init__(stage="recognize", **kwargs)
