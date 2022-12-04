from pathlib import Path
import argparse
from gats.algorithm import gatsby, genops
from gats.data import util
from gats.model import scheduler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('track', type=str, nargs='?', default='AI')
    args = parser.parse_args()

    valid_tracks = {"AI", "HCI", "Systems", "Theory", "Unspecialized"}

    if args.track not in valid_tracks:
        raise RuntimeError(f"Invalid Track! Choose From {valid_tracks}")

    print("Chosen Track:", args.track)

    bulletin = util.CourseBulletin(str(Path('./data/sample_courses.json').resolve()))

    cop = scheduler.Scheduler(bulletin, args.track)
    cop.add_default_constraints()

    ga = gatsby.Gatsby(120, 700, cop, genops.selection_rank, genops.crossover_uniform,
                       genops.mutate_shuffle, Pc=0.8, Pm=0.08, max_fitness=0, tabu=True, verbose=1, max_repeat=100)

    ga.run()
    ga.plot_fitness()
