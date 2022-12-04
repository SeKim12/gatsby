from pathlib import Path
from gats.algorithm import genops
from gats.data import util
from gats.model import scheduler
from gats.telemetry import wrapper


if __name__ == "__main__":
    bulletin = util.CourseBulletin(str(Path('./data/sample_courses.json').resolve()))
    cop = scheduler.Scheduler(bulletin, 'AI')
    cop.add_default_constraints()

    telem = wrapper.GATSTelemetry(cop)

    telem.load_config(str(Path('./config.yaml').resolve()))
    telem.dump_config()

    telem.tune_params(
        target_params=["crossover_fn", "mutate_fn"],
        ranges=[[genops.crossover_single_point, genops.crossover_two_point, genops.crossover_uniform],
                [genops.mutate_single_swap, genops.mutate_drop, genops.mutate_shuffle]],
        iterations=5
    )