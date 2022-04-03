from seutil import IOUtils, BashUtils
from pts.main import Macros
from pts.models.rank_model.TestSelectionModel import get_selection_time_per_sha

def test_run_time_model(project: str):
    institution = project.split('_')[0]
    project_name = project.split('_')[1]
    eval_data_json = IOUtils.load(f'{Macros.eval_data_dir}/mutated-eval-data/{institution}_{project_name}-ag.json',
                                  IOUtils.Format.json)
    for eval_data_item in eval_data_json:
        data_type = "Fail-Basic"
        model_saving_dir = Macros.model_data_dir / "rank-model" / project_name / data_type / "saved_models" / "best_model"
        model_time = get_selection_time_per_sha(eval_data_item, model_saving_dir)
        print(model_time)
