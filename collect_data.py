from pathlib import Path
from gats.data import util
from gats.model import scheduler
from gats.telemetry import wrapper


if __name__ == "__main__":
    bulletin = util.CourseBulletin(str(Path("./data/sample_courses.json").resolve()))
    cop = scheduler.Scheduler(bulletin, "AI")
    cop.add_default_constraints()

    telem = wrapper.GATSTelemetry(cop)

    telem.load_config(str(Path("./config.yaml").resolve()))
    telem.dump_config()

    telem.tune_params(target_params=["Pm", "Pc"], ranges=[[0.1], [0.6]], iterations=5)
