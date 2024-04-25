class job:
    def __init__(
        self,
        matchers: dict = None,
        multiclass_strategy: list = None,
        binary_strategy: list = None,
        event_map: dict = None,
        label: str = None,
    ):
        self.matchers = matchers or {
            "maximum-overlap": {},
            "iou": {},
            "iou/05": {"iou_threshold": 0.5},
            "sample": {},
        }
        self.multiclass_strategy = multiclass_strategy or [
            "all",
            "ignore_matched_undef",
            "ignore_unmatched_undef",
            "ignore_undef",
        ]
        self.binary_strategy = binary_strategy or ["tn", "error", "ignore"]
        self.event_map = event_map or {
            "1": 1,
            "2": 2,
            "4": 4,
            "0": 100,
        }  # TODO: This is wrong, for the lookAtPoint Dataset. Change to correct values
        self.label = label or "default_label"
