import pandas as pd
import time
import heapq
import numpy as np
from sklearn.utils import shuffle
from sklearn.manifold import TSNE


class MST:
    def __init__(self, df):
        # Use Prim's algorithm to construct MST
        start_index = self.find_closest_to_origin(df)
        num_vertices = len(df)
        in_mst = [False] * num_vertices
        # min_edge is a list of all minimum edges. Each edge is to_vertex (cost, from_vertex)
        min_edge = [(float("inf"), -1)] * num_vertices  # (cost, from_vertex)
        min_edge[start_index] = (0, start_index)
        pq = [(0, start_index)]  # (cost, vertex)

        self.mst = [(start_index, start_index, 0)]
        self.mst_order = [start_index]
        total_cost = 0

        while pq:
            current_cost, u = heapq.heappop(pq)
            if in_mst[u]:
                continue

            in_mst[u] = True
            total_cost += current_cost
            if min_edge[u][1] != u:
                self.mst.append((min_edge[u][1], u, current_cost))
                self.mst_order.append(u)

            for v in range(num_vertices):
                if not in_mst[v]:
                    weight = self.distance(df, u, v)
                    if weight < min_edge[v][0]:
                        min_edge[v] = (weight, u)
                        heapq.heappush(pq, (weight, v))

        self.total_cost = total_cost

    def distance(self, df, from_index, to_index):
        box1 = df.iloc[from_index]
        box2 = df.iloc[to_index]

        # bottom_left_corner1 = (box1["Left"], box1["Bottom"])
        # bottom_center1 = ((box1["Left"] + box1["Right"]) / 2, box1["Bottom"])
        # bottom_right_corner1 = (box1["Right"], box1["Bottom"])
        # right_center1 = (box1["Right"], (box1["Top"] + box1["Bottom"]) / 2)
        # top_right_corner1 = (box1["Right"], box1["Top"])

        # bottom_left_corner2 = (box2["Left"], box2["Bottom"])
        # left_center2 = (box2["Left"], (box2["Top"] + box2["Bottom"]) / 2)
        # top_left_corner2 = (box2["Left"], box2["Top"])
        # top_center2 = ((box2["Left"] + box2["Right"]) / 2, box2["Top"])
        # top_right_corner2 = (box2["Right"], box2["Top"])

        # distances = [
        #     # Distances between top 3 points of box1 and top 3 points of box 2
        #     # Distances between left 3 points of box1 and left 3 points of box2
        #     # Distances between bottom 3 points of box1 and top 3 points of box2
        #     self.euclidean_distance(bottom_left_corner1, top_left_corner2),
        #     self.euclidean_distance(bottom_left_corner1, top_center2),
        #     self.euclidean_distance(bottom_left_corner1, top_right_corner2),
        #     self.euclidean_distance(bottom_center1, top_left_corner2),
        #     self.euclidean_distance(bottom_center1, top_center2),
        #     self.euclidean_distance(bottom_center1, top_right_corner2),
        #     self.euclidean_distance(bottom_right_corner1, top_left_corner2),
        #     self.euclidean_distance(bottom_right_corner1, top_center2),
        #     self.euclidean_distance(bottom_right_corner1, top_right_corner2),
        #     # Distances between right 3 points of box1 and left 3 points of box2
        #     self.euclidean_distance(right_center1, left_center2),
        #     self.euclidean_distance(right_center1, bottom_left_corner2),
        #     self.euclidean_distance(right_center1, top_left_corner2),
        #     self.euclidean_distance(top_right_corner1, left_center2),
        #     self.euclidean_distance(top_right_corner1, bottom_left_corner2),
        #     self.euclidean_distance(top_right_corner1, top_left_corner2),
        #     self.euclidean_distance(bottom_right_corner1, top_left_corner2),
        #     self.euclidean_distance(bottom_right_corner1, left_center2),
        #     self.euclidean_distance(bottom_right_corner1, top_left_corner2),
        # ]  # Return the minimum of these distances

        # Define points for box1
        bottom_left_corner1 = (box1["Left"], box1["Bottom"])
        bottom_center1 = ((box1["Left"] + box1["Right"]) / 2, box1["Bottom"])
        bottom_right_corner1 = (box1["Right"], box1["Bottom"])
        right_center1 = (box1["Right"], (box1["Top"] + box1["Bottom"]) / 2)
        top_right_corner1 = (box1["Right"], box1["Top"])
        top_center1 = ((box1["Left"] + box1["Right"]) / 2, box1["Top"])
        top_left_corner1 = (box1["Left"], box1["Top"])
        left_center1 = (box1["Left"], (box1["Top"] + box1["Bottom"]) / 2)

        # Define points for box2
        bottom_left_corner2 = (box2["Left"], box2["Bottom"])
        bottom_center2 = ((box2["Left"] + box2["Right"]) / 2, box2["Bottom"])
        bottom_right_corner2 = (box2["Right"], box2["Bottom"])
        right_center2 = (box2["Right"], (box2["Top"] + box2["Bottom"]) / 2)
        top_right_corner2 = (box2["Right"], box2["Top"])
        top_center2 = ((box2["Left"] + box2["Right"]) / 2, box2["Top"])
        top_left_corner2 = (box2["Left"], box2["Top"])
        left_center2 = (box2["Left"], (box2["Top"] + box2["Bottom"]) / 2)

        distances = [
            #     # Distances between top 3 points of box1 and top 3 points of box2
            #     self.euclidean_distance(top_left_corner1, top_left_corner2),
            #     self.euclidean_distance(top_left_corner1, top_center2),
            #     self.euclidean_distance(top_left_corner1, top_right_corner2),
            #     self.euclidean_distance(top_center1, top_left_corner2),
            #     self.euclidean_distance(top_center1, top_center2),
            #     self.euclidean_distance(top_center1, top_right_corner2),
            #     self.euclidean_distance(top_right_corner1, top_left_corner2),
            #     self.euclidean_distance(top_right_corner1, top_center2),
            #     self.euclidean_distance(top_right_corner1, top_right_corner2),
            #     # Distances between left 3 points of box1 and left 3 points of box2
            #     self.euclidean_distance(top_left_corner1, top_left_corner2),
            #     self.euclidean_distance(top_left_corner1, left_center2),
            #     self.euclidean_distance(top_left_corner1, bottom_left_corner2),
            #     self.euclidean_distance(left_center1, top_left_corner2),
            #     self.euclidean_distance(left_center1, left_center2),
            #     self.euclidean_distance(left_center1, bottom_left_corner2),
            #     self.euclidean_distance(bottom_left_corner1, top_left_corner2),
            #     self.euclidean_distance(bottom_left_corner1, left_center2),
            #     self.euclidean_distance(bottom_left_corner1, bottom_left_corner2),
            # Distances between bottom 3 points of box1 and top 3 points of box2
            self.euclidean_distance(bottom_left_corner1, top_left_corner2),
            self.euclidean_distance(bottom_left_corner1, top_center2),
            self.euclidean_distance(bottom_left_corner1, top_right_corner2),
            self.euclidean_distance(bottom_center1, top_left_corner2),
            self.euclidean_distance(bottom_center1, top_center2),
            self.euclidean_distance(bottom_center1, top_right_corner2),
            self.euclidean_distance(bottom_right_corner1, top_left_corner2),
            self.euclidean_distance(bottom_right_corner1, top_center2),
            self.euclidean_distance(bottom_right_corner1, top_right_corner2),
            # Distances between right 3 points of box1 and left 3 points of box2
            self.euclidean_distance(right_center1, left_center2),
            self.euclidean_distance(right_center1, bottom_left_corner2),
            self.euclidean_distance(right_center1, top_left_corner2),
            self.euclidean_distance(top_right_corner1, left_center2),
            self.euclidean_distance(top_right_corner1, bottom_left_corner2),
            self.euclidean_distance(top_right_corner1, top_left_corner2),
            self.euclidean_distance(bottom_right_corner1, top_left_corner2),
            self.euclidean_distance(bottom_right_corner1, left_center2),
            self.euclidean_distance(bottom_right_corner1, top_left_corner2),
        ]  # Return the minimum of these distances

        return min(distances)

    def euclidean_distance(self, point1, point2):
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    # def distance(self, df, index1, index2):
    #     # Calculate Euclidean distance between two points in the dataframe
    #     dx = df.iloc[index1]["centroid_x"] - df.iloc[index2]["centroid_x"]
    #     dy = df.iloc[index1]["centroid_y"] - df.iloc[index2]["centroid_y"]
    #     return np.sqrt(dx**2 + dy**2)

    def find_closest_to_origin(self, df):
        # Compute distance to origin and return index of the closest point
        distances = np.sqrt(df["centroid_x"] ** 2 + df["centroid_y"] ** 2)
        return distances.idxmin()


class AccessibilityTreeModifier:
    def __init__(
        self,
        use_bboxes="gt",
        use_tags="gt",
        use_interact_element_text="gt",
        use_static_text="gt",
        use_ordering="default",
        tsne_perplexity=30,
    ):
        self.use_bboxes = use_bboxes
        self.use_tags = use_tags
        self.use_interact_element_text = use_interact_element_text
        self.use_static_text = use_static_text
        self.use_ordering = use_ordering
        self.tsne_perplexity = tsne_perplexity

    def modify(self, gt_df, pred_df):
        # TODO Implement others here

        if self.use_bboxes == "pred":
            main_df = pred_df.copy()
        elif self.use_bboxes == "gt":
            main_df = gt_df.copy()
        elif self.use_bboxes == "none":
            main_df = pred_df.copy()
            main_df = main_df[main_df["Element"] == "StaticText"]
        else:
            raise ValueError("Invalid value for 'use_bboxes'")

        # This just means I want to skip a bunch of stuff and is a shortcut
        if gt_df.equals(pred_df):
            if self.use_ordering == "gt" or self.use_ordering == "pred":
                main_df_to_pred_map = self.match_bounding_boxes(main_df, pred_df)
                main_df_to_gt_map = self.match_bounding_boxes(main_df, gt_df)
            else:
                main_df_to_pred_map = {}
                main_df_to_gt_map = {}

            main_df = self.reorder_dataframe(
                main_df,
                self.use_ordering,
                gt_df,
                pred_df,
                main_df_to_gt_map,
                main_df_to_pred_map,
            )
            return main_df

        start_time = time.time()
        main_df_to_pred_map = self.match_bounding_boxes(main_df, pred_df)
        main_df_to_gt_map = self.match_bounding_boxes(main_df, gt_df)

        main_df_to_pred_map_for_tags = self.match_bounding_boxes_for_tags(
            main_df, pred_df
        )
        main_df_to_gt_map_for_tags = self.match_bounding_boxes_for_tags(main_df, gt_df)
        print(
            f"Time taken to match bounding boxes for tags: {time.time() - start_time:.2f}s"
        )

        if self.use_tags == "pred":
            self.update_elements_based_on_mapping(
                main_df, pred_df, main_df_to_pred_map_for_tags
            )
        elif self.use_tags == "gt":
            self.update_elements_based_on_mapping(
                main_df, gt_df, main_df_to_gt_map_for_tags
            )
        elif self.use_tags == "none":
            pass
        else:
            raise ValueError("Invalid value for 'use_tags'")

        if self.use_interact_element_text == "pred":
            self.swap_interactable_text_content(main_df, pred_df, main_df_to_pred_map)
        elif self.use_interact_element_text == "gt":
            self.swap_interactable_text_content(main_df, gt_df, main_df_to_gt_map)
        elif self.use_interact_element_text == "none":
            main_df.loc[main_df["Interactable"], "TextContent"] = ""
        else:
            raise ValueError("Invalid value for 'use_interact_element_text'")

        if self.use_static_text == "pred":
            main_df = self.swap_static_boxes(main_df, pred_df)
        elif self.use_static_text == "gt":
            main_df = self.swap_static_boxes(main_df, gt_df)
        elif self.use_static_text == "none":
            main_df = main_df[main_df["Interactable"]]
            if main_df.empty:
                main_df = pd.DataFrame(columns=gt_df.columns)
        else:
            raise ValueError("Invalid value for 'use_static_text'")

        # Redo mapping after swapping static boxes
        main_df_to_pred_map = self.match_bounding_boxes(main_df, pred_df)
        main_df_to_gt_map = self.match_bounding_boxes(main_df, gt_df)

        start_time = time.time()
        main_df = self.reorder_dataframe(
            main_df,
            self.use_ordering,
            gt_df,
            pred_df,
            main_df_to_gt_map,
            main_df_to_pred_map,
        )
        print(f"Time taken to reorder DataFrame: {time.time() - start_time:.2f}s")

        return main_df

    def calculate_bbox_similarity(self, box1, box2):
        # Calculate the centers of each box
        center1 = (
            (box1["Left"] + box1["Right"]) / 2,
            (box1["Top"] + box1["Bottom"]) / 2,
        )
        center2 = (
            (box2["Left"] + box2["Right"]) / 2,
            (box2["Top"] + box2["Bottom"]) / 2,
        )

        # Calculate the Euclidean distance between the centers
        center_distance = np.sqrt(
            (center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2
        )

        # Combine the differences into a single similarity score
        similarity_score = center_distance
        return similarity_score

    def match_bounding_boxes(self, main_df, target_df):
        id_mapping = {}

        for _, main_row in main_df.iterrows():
            best_score = float("inf")
            best_id = None

            for _, target_row in target_df.iterrows():
                score = self.calculate_bbox_similarity(main_row, target_row)
                if score < best_score:
                    best_score = score
                    best_id = target_row["ID"]
                if score == best_score:
                    # If the scores are equal, prefer the ID that is closer to the original ID
                    if abs(target_row["ID"] - main_row["ID"]) < abs(
                        best_id - main_row["ID"]
                    ):
                        best_id = target_row["ID"]

            id_mapping[main_row["ID"]] = best_id

        return id_mapping

    def match_bounding_boxes_for_tags(self, main_df, target_df):
        id_mapping = {}

        for _, main_row in main_df.iterrows():
            best_score = float("inf")
            best_id = None

            # Determine the condition for current row in main_df
            main_is_static_text = main_row["Element"] == "StaticText"

            for _, target_row in target_df.iterrows():
                # Check if target row meets the same condition as the main row
                target_is_static_text = target_row["Element"] == "StaticText"

                # Proceed only if both are static texts or both are not static texts
                if main_is_static_text == target_is_static_text:
                    score = self.calculate_bbox_similarity(main_row, target_row)
                    if score < best_score:
                        best_score = score
                        best_id = target_row["ID"]
                    elif score == best_score:
                        # Prefer the ID that is closer to the original ID if scores are the same
                        if abs(target_row["ID"] - main_row["ID"]) < abs(
                            best_id - main_row["ID"]
                        ):
                            best_id = target_row["ID"]

            id_mapping[main_row["ID"]] = best_id

        return id_mapping

    def update_elements_based_on_mapping(self, main_df, other_df, id_mapping):
        """
        Updates the 'Element' column in 'main_df' using the 'Element' values from 'other_df'
        based on the ID mappings provided by 'id_mapping'.

        Parameters:
        - main_df: The main DataFrame whose 'Element' values are to be updated.
        - other_df: The DataFrame from which the new 'Element' values will be sourced.
        - id_mapping: A dictionary mapping IDs in 'main_df' to corresponding IDs in 'other_df'.
        """
        # Create a temporary mapping of ID to Element for the other dataframe for quick lookup
        other_id_to_element = other_df.set_index("ID")["Element"].to_dict()

        # Iterate through the main dataframe and update the Element values based on the mapping
        for main_id, other_id in id_mapping.items():
            if other_id in other_id_to_element:
                # Find the corresponding Element value in other_df and update it in main_df
                main_df.loc[main_df["ID"] == main_id, "Element"] = other_id_to_element[
                    other_id
                ]

    def swap_interactable_text_content(self, main_df, other_df, id_mapping):
        """
        Swaps 'TextContent' for interactable elements in 'main_df' based on 'id_mapping' and 'other_df'.

        Parameters:
        - main_df: The main DataFrame whose 'TextContent' values are to be updated for interactable elements.
        - other_df: The DataFrame from which the new 'TextContent' values will be sourced.
        - id_mapping: A dictionary mapping IDs in 'main_df' to corresponding IDs in 'other_df'.
        """
        other_id_to_text = other_df.set_index("ID")["TextContent"].to_dict()
        other_id_to_alt = other_df.set_index("ID")["Alt"].to_dict()

        # Update TextContent in main_df for interactable elements based on the mapping
        for main_id, other_id in id_mapping.items():
            if (
                other_id in other_id_to_text
                and main_df.loc[main_df["ID"] == main_id, "Interactable"].values[0]
            ):
                main_df.loc[main_df["ID"] == main_id, "TextContent"] = other_id_to_text[
                    other_id
                ]
            if (
                other_id in other_id_to_alt
                and main_df.loc[main_df["ID"] == main_id, "Interactable"].values[0]
            ):
                main_df["Alt"] = main_df["Alt"].astype("object")
                main_df.loc[main_df["ID"] == main_id, "Alt"] = other_id_to_alt[other_id]

    def swap_static_boxes(self, main_df, other_df):
        """
        Replaces non-interactable bounding boxes in 'main_df' with those from 'other_df',
        based on 'id_mapping', and attempts to preserve ordering.

        Parameters:
        - main_df: The main DataFrame.
        - other_df: The DataFrame from which bounding boxes will be sourced.
        - id_mapping: A dictionary mapping IDs in 'main_df' to corresponding IDs in 'other_df'.
        """
        # Filter out non-interactable boxes from main_df
        interactable_df = main_df[main_df["Interactable"]]

        # Create a dataframe of new boxes to be added from other_df
        static_df = other_df[~other_df["Interactable"]]
        # new_boxes_df = other_df[
        #     other_df["ID"].isin(id_mapping.values()) & ~other_df["Interactable"]
        # ]

        # Combine interactable boxes with new static boxes, attempting to preserve ordering
        combined_df = pd.concat([interactable_df, static_df])
        if combined_df.empty:
            combined_df = pd.DataFrame(columns=main_df.columns)
            return combined_df

        combined_df = combined_df.sort_values(by=["ID"])

        # Reassign IDs to maintain continuity and potentially original ordering
        combined_df["ID"] = range(1, len(combined_df) + 1)

        return combined_df

    def _order_by_euclidean_distance(self, df):
        # Calculate the Euclidean distance from the origin (0,0) for each bounding box
        distances = np.sqrt(df["Left"] ** 2 + df["Top"] ** 2)
        return df.loc[distances.argsort()].reset_index(drop=True)

    def reorder_dataframe(
        self,
        main_df,
        ordering_type,
        gt_df=None,
        pred_df=None,
        main_to_gt_map=None,
        main_to_pred_map=None,
    ):
        """
        Reorders the main_df based on the specified ordering_type.

        Parameters:
        - main_df: DataFrame to be reordered.
        - ordering_type: A string indicating how the DataFrame should be reordered ('default', 'gt', 'pred', 'random', 'origin').
        - gt_df: Ground truth DataFrame, required if ordering_type is 'gt'.
        - pred_df: Prediction DataFrame, required if ordering_type is 'pred'.
        - main_to_gt_map: ID mapping from main_df to gt_df, required if ordering_type is 'gt'.
        - main_to_pred_map: ID mapping from main_df to pred_df, required if ordering_type is 'pred'.
        """
        if ordering_type == "default":
            # Default ordering means no change is required
            return main_df
        elif ordering_type == "gt" or ordering_type == "pred":
            target_df = gt_df if ordering_type == "gt" else pred_df
            id_mapping = main_to_gt_map if ordering_type == "gt" else main_to_pred_map

            # Step 1: Assign a sort order in target_df based on its current order
            target_df["sort_order"] = range(len(target_df))

            # Step 2: Create a DataFrame from id_mapping for easier manipulation
            mapping_df = pd.DataFrame(
                list(id_mapping.items()), columns=["ID", "target_ID"]
            )

            # Merge mapping_df with target_df to get sort_order based on target_ID
            # This operation maps each main_df ID to its corresponding target_df sort_order
            mapping_df = mapping_df.merge(
                target_df[["ID", "sort_order"]],
                left_on="target_ID",
                right_on="ID",
                how="left",
                suffixes=("", "_target"),
            )

            # Now, incorporate a mechanism to maintain the original order within duplicates by using the original index from main_df
            main_df = main_df.reset_index().rename(columns={"index": "original_order"})

            # Merge main_df with the mapping_df to bring in the sort_order from target_df
            main_df_sorted = main_df.merge(
                mapping_df[["ID", "sort_order"]], on="ID", how="left"
            )

            # Step 3: Sort main_df by sort_order, then by original_order to correctly position duplicates
            main_df_sorted.sort_values(
                by=["sort_order", "original_order"], inplace=True
            )

            # Clean up by removing temporary columns
            main_df_sorted.drop(["sort_order", "original_order"], axis=1, inplace=True)

            # If necessary, reset the index to reflect the new order without adding an index column
            main_df_sorted.reset_index(drop=True, inplace=True)

            return main_df_sorted
        elif ordering_type == "random":
            # Randomly shuffle the DataFrame
            shuffled_main_df = shuffle(main_df).reset_index(drop=True)
            shuffled_main_df["ID"] = range(1, len(shuffled_main_df) + 1)
            return shuffled_main_df
        elif ordering_type == "origin":
            # Order by Euclidean distance from the origin (0,0)
            return self._order_by_euclidean_distance(main_df)
        elif ordering_type == "tsne":
            main_df = self.apply_tsne_sorting(main_df)
            return main_df
        elif ordering_type == "mst":
            main_df = self.apply_mst_sorting(main_df)
            return main_df
        elif ordering_type == "raster":
            main_df = self.apply_raster_sorting(main_df)
            return main_df
        else:
            raise NotImplementedError(
                f"Ordering type '{ordering_type}' not implemented."
            )

    def apply_raster_sorting(self, df):
        df["centroid_x"] = (df["Left"] + df["Right"]) / 2
        df["centroid_y"] = (df["Top"] + df["Bottom"]) / 2

        # Discretize centroid_y
        df["y_discrete"] = np.round(df["centroid_y"] / 8) * 8

        # Sort the DataFrame in raster order: first by y_discrete, then by centroid_x
        sorted_df = df.sort_values(
            by=["y_discrete", "centroid_x"], ascending=[True, True]
        ).reset_index(drop=True)
        sorted_df["ID"] = range(1, len(sorted_df) + 1)
        sorted_df = sorted_df.drop(columns=["centroid_x", "centroid_y", "y_discrete"])

        return sorted_df

    def apply_mst_sorting(self, df):
        df["centroid_x"] = (df["Left"] + df["Right"]) / 2
        df["centroid_y"] = (df["Top"] + df["Bottom"]) / 2
        mst = MST(df)
        sorted_df = df.iloc[mst.mst_order].reset_index(drop=True)
        sorted_df["ID"] = range(1, len(sorted_df) + 1)
        sorted_df = sorted_df.drop(columns=["centroid_x", "centroid_y"])
        return sorted_df

    def apply_tsne_sorting(self, df):
        # Step 1: Prepare the data
        # Extract centroids of bounding boxes as the features for t-SNE
        df["centroid_x"] = (df["Left"] + df["Right"]) / 2
        df["centroid_y"] = (df["Top"] + df["Bottom"]) / 2
        features = df[["centroid_x", "centroid_y"]]

        # Step 2: Apply t-SNE
        perplexity = self.tsne_perplexity
        if len(df) <= perplexity:
            perplexity = len(df) - 1
        if len(df) < 2:
            return df

        tsne = TSNE(
            n_components=1, random_state=42, perplexity=perplexity
        )  # n_components=1 for 1D output
        tsne_results = tsne.fit_transform(features)

        # Step 3: Sort Data
        # Add the t-SNE results to the dataframe for sorting
        df["tsne_1d"] = tsne_results.ravel()
        sorted_df = df.sort_values(by="tsne_1d", ascending=False)

        sorted_df["ID"] = range(1, len(sorted_df) + 1)
        # Optionally, you can remove the auxiliary columns if not needed
        sorted_df = sorted_df.drop(columns=["centroid_x", "centroid_y", "tsne_1d"])

        return sorted_df
