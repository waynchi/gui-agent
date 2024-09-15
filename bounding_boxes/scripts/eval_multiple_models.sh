#!/bin/bash

# python eval/bounding_box_eval.py --dataset vwa_crawl --model_path outputs/100k_bs_16_classes_2/model_0379999.pth --additional_params _100k_bs_16_classes_2_test
# python eval/bounding_box_eval.py --dataset vwa_crawl --model_path outputs/20k_bs_16_classes_2/model_final.pth --additional_params _20k_bs_16_classes_2_test
# python eval/bounding_box_eval.py --dataset vwa_crawl --model_path outputs/vwa_20k_classes_2/model_final.pth --additional_params _vwa_20k_classes_2_test
# python eval/bounding_box_eval.py --dataset vwa_crawl --model_path outputs/vwa_100k_classes_2/model_final.pth --additional_params _vwa_100k_classes_2_test
# python eval/bounding_box_eval.py --dataset vwa_crawl --model_path outputs/20k_bs_16_classes_9/model_final.pth --additional_params _20k_bs_16_classes_9_test


python eval/bounding_box_eval.py --dataset vwa_crawl --ground_truth_path /home/waynechi/dev/gui-agent/bounding_boxes/datasets/vwa_2/static_vwa_crawl_classes_9_test.csv --model_path outputs/vwa_pl_100k_classes_9/model_final.pth --additional_params _vwa_pl_100k_classes_9_test