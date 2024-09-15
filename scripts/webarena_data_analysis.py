import os
from bs4 import BeautifulSoup


def load_and_search_html_files(folder_path, search_terms, difficulty_terms):
    search_results = {}
    difficulty_term_count = 0

    for file in os.listdir(folder_path):
        if file.endswith(".html"):
            file_path = os.path.join(folder_path, file)
            with open(file_path, "r", encoding="utf-8") as html_file:
                contents = html_file.read()
                soup = BeautifulSoup(contents, "html.parser")

                # Check for difficulty terms
                if all(
                    soup.find_all(string=lambda text: text and term in text)
                    for term in difficulty_terms
                ):
                    difficulty_term_count += 1
                    # Search for terms in this file
                    for term in search_terms:
                        if soup.find_all(string=lambda text: text and term in text):
                            search_results.setdefault(file, []).append(term)

    return search_results, difficulty_term_count


# Example usage:
folders = ["reddit", "shopping", "classifieds"]
for folder in folders:
    folder_path = "webarena_trajectories/{}".format(
        folder
    )  # Replace with your folder path
    search_terms = [
        "new_tab",
        "tab_focus",
        "close_tab",
        "goto [",
        "go_back",
        "go_forward",
        "hover",
        "press [",
    ]
    difficulty_terms = [
        "reasoning_difficulty: easy",
        "visual_difficulty: easy",
        "overall_difficulty: easy",
    ]
    found_terms, file_count = load_and_search_html_files(
        folder_path, search_terms, difficulty_terms
    )
    print(found_terms)
    print("Number of files meeting difficulty criteria: {}".format(file_count))
