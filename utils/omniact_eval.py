import json
from math import sqrt
import re
from nltk.translate.bleu_score import sentence_bleu


def get_bounds(box: dict, cx, cy):
    for i in box:
        tl = box[i]["top_left"]
        br = box[i]["bottom_right"]
        if (tl[0] + br[0]) / 2 == cx and (tl[1] + br[1]) / 2 == cy:
            return (tl, br)

    assert False


def dynamic_dirichlet_l2_penalty(tl, br, px, py):
    len_x = br[0] - tl[0]
    len_y = br[1] - tl[1]

    cx = (br[0] + tl[0]) / 2
    cy = (br[1] + tl[1]) / 2

    dx = abs(cx - px) - (len_x * 0.5)
    dy = abs(cy - py) - (len_y * 0.5)
    dist = sqrt((dx * (dx > 0)) ** 2 + (dy * (dy > 0)) ** 2)

    mu = sqrt(len_x**2 + len_y**2)

    score = mu / (dist + mu)
    penalty = 1 - score
    return penalty


def get_scores(gold, pred):
    sequence_match = 0
    action_score = 0
    total_click_penalty = 0
    total_press_penalty = 0
    total_write_penalty = 0
    ideal_score = 0
    max_click_penalty = 0
    max_press_penalty = 0
    max_write_penalty = 0

    gpt4v_envs = [
        "4",
        "58",
        "115",
        "147",
        "156",
        "162",
        "165",
        "178",
        "179",
        "194",
        "204",
        "218",
        "235",
        "240",
        "248",
        "297",
        "353",
        "374",
        "391",
        "392",
        "395",
        "404",
        "409",
        "419",
        "434",
        "462",
        "487",
        "492",
        "517",
        "533",
        "556",
        "573",
        "598",
        "658",
        "667",
        "673",
        "678",
        "719",
        "795",
        "827",
        "896",
        "910",
        "944",
        "961",
        "975",
        "1018",
        "1025",
        "1038",
        "1084",
        "1093",
        "1101",
        "1103",
        "1128",
        "1130",
        "1138",
        "1142",
        "1147",
        "1181",
        "1192",
        "1219",
        "1252",
        "1284",
        "1291",
        "1353",
        "1427",
        "1442",
        "1448",
        "1514",
        "1521",
        "1538",
        "1580",
        "1590",
        "1594",
        "1600",
        "1606",
        "1622",
        "1636",
        "1641",
        "1665",
        "1684",
        "1694",
        "1696",
        "1710",
        "1711",
        "1719",
        "1726",
        "1731",
        "1740",
        "1743",
        "1845",
        "1877",
        "1883",
        "1918",
        "1924",
        "1951",
        "1960",
        "1993",
        "1994",
        "1997",
        "2011",
    ]

    for idx in range(len(gold)):
        # if str(idx) not in gpt4v_envs:
        #     continue
        gold_script = open(gold[idx]["task"]).read().strip().split("\n")[1:]
        gold_script = [
            x.lower() for x in gold_script if x.lower().strip().startswith("pyautogui")
        ]
        llm_script = pred[idx].strip().split("\n")
        llm_script = [
            x.lower() for x in llm_script if x.lower().strip().startswith("pyautogui")
        ]
        # find extreme case values
        sample_weight = len(gold_script) - 0.9
        # sample_weight = 1

        ideal_score += sample_weight
        for gold_line in gold_script:
            action_type = gold_line.split("pyautogui.")[1].split("(")[0]
            if (
                action_type == "click"
                or action_type == "rightClick"
                or action_type == "moveTo"
                or action_type == "dragTo"
            ):
                max_click_penalty += sample_weight / len(gold_script)
            if action_type == "press" or action_type == "hotkey":
                max_press_penalty += sample_weight / len(gold_script)
            if action_type == "write":
                max_write_penalty += sample_weight / len(gold_script)

        seq_match_flag = 1
        click_penalty = 0
        press_penalty = 0
        write_penalty = 0

        # if length doesn't seq match is 0
        # llm_script = llm_script[:len(gold_script)]
        if len(llm_script) != len(gold_script):
            seq_match_flag = 0
        if seq_match_flag == 1:
            for i in range(len(gold_script)):
                gold_line = gold_script[i].strip()
                gold_action = gold_line.split("pyautogui.")[1].split("(")[0]
                pred_line = llm_script[i]
                if pred_line.startswith("pyautogui.") == False:
                    seq_match_flag = 0
                    break
                pred_action = pred_line.split("pyautogui.")[1].split("(")[0]
                if pred_action != gold_action:
                    seq_match_flag = 0
                    break

        # if seq_match_flag == 0:
        #     breakpoint()

        # find penalties for correct and wrong sequences
        box_path = gold[idx]["box"]
        # box_num = re.search(r"\d+", box_path.split("/")[-1])
        # box_num = box_path.split("_")[-1].split(".json")[0]
        # box_path = "_".join(box_path.split("_")[:-1]) + box_num + "_boxes.json"
        box = json.load(open(box_path))

        for i in range(len(gold_script)):
            gold_line = gold_script[i].strip()
            gold_action = gold_line.split("pyautogui.")[1].split("(")[0]
            # just add the penalties
            if seq_match_flag == 0:
                if (
                    gold_action == "click"
                    or gold_action == "rightClick"
                    or gold_action == "moveTo"
                    or gold_action == "dragTo"
                ):
                    click_penalty += 1 / len(gold_script)
                if gold_action == "press" or gold_action == "hotkey":
                    press_penalty += 1 / len(gold_script)
                if gold_action == "write":
                    write_penalty += 1 / len(gold_script)
                continue
            pred_line = llm_script[i]
            pred_action = pred_line.split("pyautogui.")[1].split("(")[0]

            # l2 penalty for click

            if gold_action == "click" or gold == "rightClick":
                # get original box bounds
                gold_cx = gold_line.split("pyautogui.")[1].split("(")[1].split(",")[0]
                gold_cy = (
                    gold_line.split("pyautogui.")[1]
                    .split("(")[1]
                    .split(",")[1]
                    .split(")")[0]
                )
                tl, br = get_bounds(box, float(gold_cx), float(gold_cy))

                # get predicted point
                pred_cx = pred_line.split("pyautogui.")[1].split("(")[1].split(",")[0]
                pred_cy = (
                    pred_line.split("pyautogui.")[1]
                    .split("(")[1]
                    .split(",")[1]
                    .split(")")[0]
                )

                click_penalty += (
                    1.0 / len(gold_script)
                ) * dynamic_dirichlet_l2_penalty(tl, br, float(pred_cx), float(pred_cy))

            # penalty for press
            if gold_action == "press":
                gold_key = gold_line.split('"')[1]
                pred_key = (re.split("\"|'", pred_line))[1]
                if gold_key.strip() != pred_key.strip():
                    press_penalty += 1 / len(gold_script)

            # penalty for hotkey
            if gold_action == "hotkey":
                gold_keys = gold_line.split("(")[1].split(")")[0].split(",")
                pred_keys = pred_line.split("(")[1].split(")")[0].split(",")

                gold_key_set = set([x[1:-1] for x in gold_keys if len(x) > 2])
                pred_key_set = set([x[1:-1] for x in pred_keys if len(x) > 2])
                if gold_key_set != pred_key_set:
                    press_penalty += 1 / len(gold_script)

            if gold_action == "write":
                reference = [gold_line.split('"')[1]]
                candidate = re.split("\"|'", pred_line)[1]
                write_penalty += (
                    1 - sentence_bleu(reference, candidate, weights=(0.5, 0.5))
                ) / len(gold_script)

        sequence_match += (seq_match_flag) * sample_weight
        action_score += (
            max(seq_match_flag - click_penalty - press_penalty - write_penalty, 0)
        ) * sample_weight
        if seq_match_flag:
            total_click_penalty += click_penalty * sample_weight
            total_press_penalty += press_penalty * sample_weight
            total_write_penalty += write_penalty * sample_weight

    # print(ideal_score)
    total_sequence_score = sequence_match / ideal_score
    total_action_score = action_score / ideal_score
    # print(f"Sequence score: {total_sequence_score}")
    # print(f"Action score: {total_action_score}")

    # print(total_click_penalty / ideal_score)
    # print(total_press_penalty / ideal_score)
    # print(total_write_penalty / ideal_score)

    return {
        "sequence_score": total_sequence_score,
        "action_score": total_action_score,
        "click_penalty": total_click_penalty / ideal_score,
        "press_penalty": total_press_penalty / ideal_score,
        "write_penalty": total_write_penalty / ideal_score,
        "ideal_score": ideal_score,
    }


if __name__ == "__main__":
    from pprint import pprint

    file_paths = [
        # {
        #     "exp_name": "omniact_gemini_emphasize_ordering_diff_app_even",
        #     "gold": "/data/waynechi/gui-agent/data/omniact_gemini_emphasize_ordering_diff_app_even_20240505_010527/1631_run_0/gt_json.json",
        #     "pred": "/data/waynechi/gui-agent/data/omniact_gemini_emphasize_ordering_diff_app_even_20240505_010527/1631_run_0/pred_json.json",
        # },
        # {
        #     "exp_name": "omniact_gemini_emphasize_ordering_diff_app_even_no_img",
        #     "gold": "/data/waynechi/gui-agent/data/omniact_gemini_emphasize_ordering_diff_app_even_no_img_20240506_003735/1631_run_0/gt_json.json",
        #     "pred": "/data/waynechi/gui-agent/data/omniact_gemini_emphasize_ordering_diff_app_even_no_img_20240506_003735/1631_run_0/pred_json.json",
        # },
        # {
        #     "exp_name": "omniact_gemini_emphasize_ordering_diff_app_even_gt_order",
        #     "gold": "/data/waynechi/gui-agent/data/omniact_gemini_emphasize_ordering_diff_app_even_gt_order_20240505_184758/1631_run_0/gt_json.json",
        #     "pred": "/data/waynechi/gui-agent/data/omniact_gemini_emphasize_ordering_diff_app_even_gt_order_20240505_184758/1631_run_0/pred_json.json",
        # },
        # {
        #     "exp_name": "omniact_gemini_emphasize_ordering_diff_app_even_no_ocr",
        #     "gold": "/data/waynechi/gui-agent/data/omniact_gemini_emphasize_ordering_diff_app_even_no_ocr_20240505_040449/1631_run_0/gt_json.json",
        #     "pred": "/data/waynechi/gui-agent/data/omniact_gemini_emphasize_ordering_diff_app_even_no_ocr_20240505_040449/1631_run_0/pred_json.json",
        # },
        # {
        #     "exp_name": "omniact_gemini_emphasize_ordering_diff_app_even_only_ocr",
        #     "gold": "/data/waynechi/gui-agent/data/omniact_gemini_emphasize_ordering_diff_app_even_only_ocr_20240506_023649/1631_run_0/gt_json.json",
        #     "pred": "/data/waynechi/gui-agent/data/omniact_gemini_emphasize_ordering_diff_app_even_only_ocr_20240506_023649/1631_run_0/pred_json.json",
        # },
        # {
        #     "exp_name": "omniact_gemini_emphasize_ordering_diff_app_even_pred_box",
        #     "gold": "/data/waynechi/gui-agent/data/omniact_gemini_emphasize_ordering_diff_app_even_pred_box_20240505_203356/1631_run_0/gt_json.json",
        #     "pred": "/data/waynechi/gui-agent/data/omniact_gemini_emphasize_ordering_diff_app_even_pred_box_20240505_203356/1631_run_0/pred_json.json",
        # },
        # {
        #     "exp_name": "omniact_gemini_text_interaction_diff_app_even",
        #     "gold": "/data/waynechi/gui-agent/data/omniact_gemini_text_interaction_diff_app_even_20240505_223959/1631_run_0/gt_json.json",
        #     "pred": "/data/waynechi/gui-agent/data/omniact_gemini_text_interaction_diff_app_even_20240505_223959/1631_run_0/pred_json.json",
        # },
        # {"exp_name": ""},
        # {
        #     "exp_name": "omniact_gemini_emphasize_ordering",
        #     "gold": "/data/waynechi/gui-agent/data/omniact_gemini_emphasize_ordering_20240504_040438/2014_run_0/gt_json.json",
        #     "pred": "/data/waynechi/gui-agent/data/omniact_gemini_emphasize_ordering_20240504_040438/2014_run_0/pred_json.json",
        # },
        # {
        #     "exp_name": "omniact_gemini_3",
        #     "gold": "/data/waynechi/gui-agent/data/omniact_gemini_3_20240504_063022/2014_run_0/gt_json.json",
        #     "pred": "/data/waynechi/gui-agent/data/omniact_gemini_3_20240504_063022/2014_run_0/pred_json.json",
        # },
        # {"exp_name": ""},
        # {
        #     "exp_name": "omniact_gpt_type_2",
        #     "gold": "/data/waynechi/gui-agent/data/omniact_gpt_type_2_20240502_213725/2018_run_0/gt_json.json",
        #     "pred": "/data/waynechi/gui-agent/data/omniact_gpt_type_2_20240502_213725/2018_run_0/pred_json.json",
        # },
        # {
        #     "exp_name": "omniact_gemini",
        #     "gold": "/data/waynechi/gui-agent/data/omniact_gemini_20240501_015126/2018_run_0/gt_json.json",
        #     "pred": "/data/waynechi/gui-agent/data/omniact_gemini_20240501_015126/2018_run_0/pred_json.json",
        # },
        # {
        #     "exp_name": "omniact_gemini_short",
        #     "gold": "/data/waynechi/gui-agent/data/omniact_gemini_short_20240504_002111/2018_run_0/gt_json.json",
        #     "pred": "/data/waynechi/gui-agent/data/omniact_gemini_short_20240504_002111/2018_run_0/pred_json.json",
        # },
        # {
        #     "exp_name": "omniact_gemini_pyautogui",
        #     "gold": "/data/waynechi/gui-agent/data/omniact_gemini_pyautogui_20240502_035358/2018_run_0/gt_json.json",
        #     "pred": "/data/waynechi/gui-agent/data/omniact_gemini_pyautogui_20240502_035358/2018_run_0/pred_json.json",
        # },
        # {
        #     "exp_name": "omniact_gemini_text_interaction",
        #     "gold": "/data/waynechi/gui-agent/data/omniact_gemini_text_interaction_20240503_185801/2018_run_0/gt_json.json",
        #     "pred": "/data/waynechi/gui-agent/data/omniact_gemini_text_interaction_20240503_185801/2018_run_0/pred_json.json",
        # },
        # {
        #     "exp_name": "omniact_gemini_filter",
        #     "gold": "/data/waynechi/gui-agent/data/omniact_gemini_filter_20240503_054842/2018_run_0/gt_json.json",
        #     "pred": "/data/waynechi/gui-agent/data/omniact_gemini_filter_20240503_054842/2018_run_0/pred_json.json",
        # },
        # {"exp_name": ""},
        # {
        #     "exp_name": "omniact_gemini_all",
        #     "gold": "/data/waynechi/gui-agent/data/omniact_gemini_all_20240501_063315/2020_run_0/gt_json.json",
        #     "pred": "/data/waynechi/gui-agent/data/omniact_gemini_all_20240501_063315/2020_run_0/pred_json.json",
        # },
        {"exp_name": ""},
        {
            "exp_name": "omniact_gemini_gt_random_all",
            "gold": "/data/waynechi/gui-agent/data/omniact_gemini_random_gt_bbox/gt_json.json",
            "pred": "/data/waynechi/gui-agent/data/omniact_gemini_random_gt_bbox/pred_json.json",
        },
        {
            "exp_name": "omniact_gemini_gt_tsne_all",
            "gold": "/data/waynechi/gui-agent/data/omniact_gemini_gt_bbox_tsne_ordering/gt_json.json",
            "pred": "/data/waynechi/gui-agent/data/omniact_gemini_gt_bbox_tsne_ordering/pred_json.json",
        },
        {
            "exp_name": "omniact_gemini_pred_random",
            "gold": "/data/waynechi/gui-agent/data/omniact_gemini_random_order_pred_bbox/gt_json.json",
            "pred": "/data/waynechi/gui-agent/data/omniact_gemini_random_order_pred_bbox/pred_json.json",
        },
        {
            "exp_name": "omniact_gemini_pred_tsne",
            "gold": "/data/waynechi/gui-agent/data/omniact_gemini_single_best_pred_bbox_all_20240512_081150/2020_run_0/gt_json.json",
            "pred": "/data/waynechi/gui-agent/data/omniact_gemini_single_best_pred_bbox_all_20240512_081150/2020_run_0/pred_json.json",
        },
        {
            "exp_name": "omniact_gemini_pred_raster",
            "gold": "/data/waynechi/gui-agent/data/omniact_gemini_raster_pred/gt_json.json",
            "pred": "/data/waynechi/gui-agent/data/omniact_gemini_raster_pred/pred_json.json",
        },
        {
            "exp_name": "omniact_gemini_pred_tsne_all_no_img",
            "gold": "/data/waynechi/gui-agent/data/omniact_gemini_single_tsne_pred_bbox_no_img_all_20240519_091408/2020_run_0/gt_json.json",
            "pred": "/data/waynechi/gui-agent/data/omniact_gemini_single_tsne_pred_bbox_no_img_all_20240519_091408/2020_run_0/pred_json.json",
        },
        # {
        #     "exp_name": "omniact_llama_random_all",
        #     "gold": "/data/waynechi/gui-agent/data/omniact_llama8b_no_tag_short_random_order_gt_bbox_all_20240515_170457/2020_run_0/gt_json.json",
        #     "pred": "/data/waynechi/gui-agent/data/omniact_llama8b_no_tag_short_random_order_gt_bbox_all_20240515_170457/2020_run_0/pred_json.json",
        # },
        # {
        #     "exp_name": "omniact_llama_tsne_all (aka perplexity 30)",
        #     "gold": "/data/waynechi/gui-agent/data/omniact_llama8b_no_tag_short_gt_bbox_all_20240511_045443/2020_run_0/gt_json.json",
        #     "pred": "/data/waynechi/gui-agent/data/omniact_llama8b_no_tag_short_gt_bbox_all_20240511_045443/2020_run_0/pred_json.json",
        # },
        # {
        #     "exp_name": "perplexity_10_llama",
        #     "gold": "/data/waynechi/gui-agent/data/omniact_llama8b_no_tag_short_gt_bbox_perp_10_all_20240515_010232/gt_json.json",
        #     "pred": "/data/waynechi/gui-agent/data/omniact_llama8b_no_tag_short_gt_bbox_perp_10_all_20240515_010232/pred_json.json",
        # },
        # {
        #     "exp_name": "perlexity_20_llama",
        #     "gold": "/data/waynechi/gui-agent/data/omniact_llama8b_no_tag_short_gt_bbox_perp_20_all_20240515_083611/gt_json.json",
        #     "pred": "/data/waynechi/gui-agent/data/omniact_llama8b_no_tag_short_gt_bbox_perp_20_all_20240515_083611/pred_json.json",
        # },
        # {
        #     "exp_name": "perlexity_40_llama",
        #     "gold": "/data/waynechi/gui-agent/data/omniact_llama8b_no_tag_short_gt_bbox_perp_40_all_20240515_112345/gt_json.json",
        #     "pred": "/data/waynechi/gui-agent/data/omniact_llama8b_no_tag_short_gt_bbox_perp_40_all_20240515_112345/pred_json.json",
        # },
        # {
        #     "exp_name": "perlexity_50_llama",
        #     "gold": "/data/waynechi/gui-agent/data/omniact_llama8b_no_tag_short_gt_bbox_perp_50_all_20240515_141406/gt_json.json",
        #     "pred": "/data/waynechi/gui-agent/data/omniact_llama8b_no_tag_short_gt_bbox_perp_50_all_20240515_141406/pred_json.json",
        # },
        # {
        #     "exp_name": "tsne_llama_pred",
        #     "gold": "/data/waynechi/gui-agent/data/omniact_llama8b_no_tag_short_pred_bbox_all_20240514_172625/2020_run_0/gt_json.json",
        #     "pred": "/data/waynechi/gui-agent/data/omniact_llama8b_no_tag_short_pred_bbox_all_20240514_172625/2020_run_0/pred_json.json",
        # },
        # {
        #     "exp_name": "random_llama_pred",
        #     "gold": "/data/waynechi/gui-agent/data/omniact_llama8b_no_tag_short_random_order_pred_bbox_all_20240514_210746/2020_run_0/gt_json.json",
        #     "pred": "/data/waynechi/gui-agent/data/omniact_llama8b_no_tag_short_random_order_pred_bbox_all_20240514_210746/2020_run_0/pred_json.json",
        # },
        # {
        #     "exp_name": "tsne_llama_gt",
        #     "gold": "/data/waynechi/gui-agent/data/omniact_llama8b_no_tag_short_gt_bbox_all_20240511_045443/2020_run_0/gt_json.json",
        #     "pred": "/data/waynechi/gui-agent/data/omniact_llama8b_no_tag_short_gt_bbox_all_20240511_045443/2020_run_0/pred_json.json",
        # },
        # {
        #     "exp_name": "random_llama_gt",
        #     "gold": "/data/waynechi/gui-agent/data/omniact_llama8b_no_tag_short_random_order_gt_bbox_all_run_1/2020_run_0/gt_json.json",
        #     "pred": "/data/waynechi/gui-agent/data/omniact_llama8b_no_tag_short_random_order_gt_bbox_all_run_1/2020_run_0/pred_json.json",
        # },
        # {
        #     "exp_name": "random_llama_pred",
        #     "gold": "/data/waynechi/gui-agent/data/omniact_gemini_random_order_pred_bbox/gt_json.json",
        #     "pred": "/data/waynechi/gui-agent/data/omniact_gemini_random_order_pred_bbox/pred_json.json",
        # },
        {
            "exp_name": "omniact llama8b gt raster",
            "gold": "/data/waynechi/gui-agent/data/omniact_llama8b_no_tag_short_raster_order_gt_bbox_all_20240517_220944/2020_run_0/gt_json.json",
            "pred": "/data/waynechi/gui-agent/data/omniact_llama8b_no_tag_short_raster_order_gt_bbox_all_20240517_220944/2020_run_0/pred_json.json",
        },
        {
            "exp_name": "omniact llama8b pred raster",
            "gold": "/data/waynechi/gui-agent/data/omniact_llama8b_no_tag_short_raster_order_pred_bbox_all_20240517_184401/2020_run_0/gt_json.json",
            "pred": "/data/waynechi/gui-agent/data/omniact_llama8b_no_tag_short_raster_order_pred_bbox_all_20240517_184401/2020_run_0/pred_json.json",
        },
        {"exp_name": "gpt4v omniact"},
        {
            "exp_name": "omniact_gpt4v_pred_tsne FULL TEST SET",
            "gold": "/data/waynechi/gui-agent/data/omniact_gpt4v_pred_box_all/gt_json.json",
            "pred": "/data/waynechi/gui-agent/data/omniact_gpt4v_pred_box_all/pred_json.json",
        },
        {
            "exp_name": "omniact_gpt4v_random_pred",
            "gold": "/data/waynechi/gui-agent/data/omniact_gpt4v_random_pred_box_all_20240518_010533/2011_run_0/gt_json.json",
            "pred": "/data/waynechi/gui-agent/data/omniact_gpt4v_random_pred_box_all_20240518_010533/2011_run_0/pred_json.json",
        },
        {
            "exp_name": "omniact_gpt4v_raster_pred",
            "gold": "/data/waynechi/gui-agent/data/omniact_gpt4v_raster_pred_box_all_20240518_013253/2011_run_0/gt_json.json",
            "pred": "/data/waynechi/gui-agent/data/omniact_gpt4v_raster_pred_box_all_20240518_013253/2011_run_0/pred_json.json",
        },
        {
            "exp_name": "omniact_gpt4v_random_gt",
            "gold": "/data/waynechi/gui-agent/data/omniact_gpt4v_random_gt_box_all_20240518_015938/2011_run_0/gt_json.json",
            "pred": "/data/waynechi/gui-agent/data/omniact_gpt4v_random_gt_box_all_20240518_015938/2011_run_0/pred_json.json",
        },
        {
            "exp_name": "omniact_gpt4v_raster_gt",
            "gold": "/data/waynechi/gui-agent/data/omniact_gpt4v_raster_gt_box_all_20240518_025100/2011_run_0/gt_json.json",
            "pred": "/data/waynechi/gui-agent/data/omniact_gpt4v_raster_gt_box_all_20240518_025100/2011_run_0/pred_json.json",
        },
        {
            "exp_name": "omniact_gpt4v_tsne_gt",
            "gold": "/data/waynechi/gui-agent/data/omniact_gpt4v_tsne_gt_box_all_20240518_022522/2011_run_0/gt_json.json",
            "pred": "/data/waynechi/gui-agent/data/omniact_gpt4v_tsne_gt_box_all_20240518_022522/2011_run_0/pred_json.json",
        },
        {"exp_name": "gpt4v apples to apples"},
        {
            "exp_name": "gpt4v_random_pyautogui",
            "gold": "/data/waynechi/gui-agent/data/omniact_gpt4v_random_pred_box_pyautogui_no_img_all_20240520_042424/2011_run_0/gt_json.json",
            "pred": "/data/waynechi/gui-agent/data/omniact_gpt4v_random_pred_box_pyautogui_no_img_all_20240520_042424/2011_run_0/pred_json.json",
        },
        {
            "exp_name": "gpt4v_tsne_pyautogui",
            "gold": "/data/waynechi/gui-agent/data/omniact_gpt4v_tsne_pred_box_pyautogui_no_img_all_20240520_043946/2011_run_0/gt_json.json",
            "pred": "/data/waynechi/gui-agent/data/omniact_gpt4v_tsne_pred_box_pyautogui_no_img_all_20240520_043946/2011_run_0/pred_json.json",
        },
        {
            "exp_name": "gpt4v_random_actions",
            "gold": "/data/waynechi/gui-agent/data/omniact_gpt4v_random_pred_box_no_img_all_20240520_051753/2011_run_0/gt_json.json",
            "pred": "/data/waynechi/gui-agent/data/omniact_gpt4v_random_pred_box_no_img_all_20240520_051753/2011_run_0/pred_json.json",
        },
        {
            "exp_name": "gpt4v_tsne_actions",
            "gold": "/data/waynechi/gui-agent/data/omniact_gpt4v_tsne_pred_box_no_img_all_20240520_053159/2011_run_0/gt_json.json",
            "pred": "/data/waynechi/gui-agent/data/omniact_gpt4v_tsne_pred_box_no_img_all_20240520_053159/2011_run_0/pred_json.json",
        },
    ]

    for file_path in file_paths:
        exp_name = file_path["exp_name"]
        if "gold" not in file_path:
            print(f"{file_path['exp_name']} ================= SECTION")
            continue
        gold = json.load(open(file_path["gold"]))
        pred = json.load(open(file_path["pred"]))
        scores = get_scores(gold, pred)
        print(exp_name)
        pprint(scores)
        print("=====")
