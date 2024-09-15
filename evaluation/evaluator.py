import os
import json
from actions.action import Action
from response_parsers.webarena_som_response_parser import WebArenaSoMResponseParser
from pprint import pprint


class TrajectoryEvaluator:
    def __init__(self) -> None:
        self.response_parser = WebArenaSoMResponseParser(None)
        # Create a dictionary of numbers 1 to 200 with coordinates that are the number itself twice
        self.id2center = {str(i): (i, i) for i in range(0, 200)}

    def trajectory_file_to_trajectories(self, trajectory_file):
        # open json
        with open(trajectory_file, "r") as f:
            trajectories_json = json.load(f)
            trajectories = []
            for trajectory in trajectories_json:
                actions = []
                for response in trajectory:
                    actions.append(
                        self.response_parser.response_to_action(
                            response, id2center=self.id2center, no_filter=True
                        )
                    )

                trajectories.append(actions)

        return trajectories

    def data_folder_to_trajectory(self, data_folder):
        # go through folders in data_folder
        # for each folder, get the trajectory file
        step_folders = [
            d
            for d in os.listdir(data_folder)
            if os.path.isdir(os.path.join(data_folder, d))
        ]
        actions = []
        for i in range(len(step_folders)):
            info_file = os.path.join(data_folder, "step_{}".format(i), "info.json")
            with open(info_file, "r") as f:
                info = json.load(f)
                actions.append(
                    self.response_parser.response_to_action(
                        info["response"], id2center=self.id2center, no_filter=False
                    )
                )

        return actions

    def evaluate_trajectory(self, gt_trajectory, pred_trajectory):
        # TODO What is the best way to evaluate trajectories?
        # Compare just action types as a fuzzy matching
        # Compare action types and the parameters for more exact matching
        # Slide window to match in case the model loops?
        # Should I penalize actions that do nothing?
        score = 0
        gt_index = 0
        for pred_index in range(len(pred_trajectory)):
            if gt_index >= len(gt_trajectory):
                break
            print("Comparing {}".format(gt_trajectory[gt_index]))
            print("With {}".format(pred_trajectory[pred_index]))
            if (
                gt_trajectory[gt_index].action_type
                == pred_trajectory[pred_index].action_type
                and gt_trajectory[gt_index].action_params
                == pred_trajectory[pred_index].action_params
            ):
                print("Successful comparison \n")
                gt_index += 1
                score += 1

        return score

    def evaluate_data_folder(self, category, experiment_number, data_folder):
        print("\nStarting experiment {}\n".format(experiment_number))
        trajectory_file = "valid_trajectories/test_{}/{}.json".format(
            category, experiment_number
        )
        gt_trajectories = self.trajectory_file_to_trajectories(trajectory_file)
        pred_trajectory = self.data_folder_to_trajectory(data_folder)

        # pprint(gt_trajectories)
        # pprint(pred_trajectory)

        scores = [
            self.evaluate_trajectory(gt, pred_trajectory) for gt in gt_trajectories
        ]

        return scores


if __name__ == "__main__":
    evaluator = TrajectoryEvaluator()
    category = "reddit"
    valid_trajectory_folder = "valid_trajectories/test_{}/".format(category)

    experiment_numbers = [
        file.split(".json")[0] for file in os.listdir(valid_trajectory_folder)
    ]
    # experiment_numbers = [89]
    data_folder = "/Users/waynechi/dev/gui-agent/data/vwa_reddit_easy_som_20240221_205026/vwa_config_files/test_reddit/{}.json_run_0"
    scores_dict = {}
    for experiment_number in experiment_numbers:
        try:
            scores = evaluator.evaluate_data_folder(
                category, experiment_number, data_folder.format(experiment_number)
            )
            scores_dict[experiment_number] = max(
                scores
            )  # TODO Do I want to take the max here?
        except:
            pass

    print(scores_dict)
