class Learner(object):
    def __init__(self, initial_subjects, initial_fitness):
        super().__init__()
        self.subjects = initial_subjects
        self.fitness = initial_fitness


    def __repr__(self):
        return f'Learner (sub: {self.subjects} fit: {self.fitness})'