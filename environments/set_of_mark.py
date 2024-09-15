class AccTreeElement:
    def __init__(self, identifier: int, label, description) -> None:
        self.identifier = identifier
        self.label = label
        self.description = description


class SoMState:
    """
    TODO: Migrate to SoMCaption rather than just using string
    """

    def __init__(self, som_image, acc_tree: str, id2center, observation, url: str = ""):
        self.som_image = som_image
        self.acc_tree = acc_tree
        self.id2center = id2center
        self.observation = observation
        self.url = url
