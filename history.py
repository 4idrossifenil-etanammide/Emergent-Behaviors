class History:
    def __init__(self, agents, landmarks):
        self.num_agents = agents.shape[1]

        self.history = {
            "initial_agents": {
                "positions": agents[:, :, :2].clone(),
                "colors": agents[:, :, 6:9].clone(),
                "shapes": agents[:, :, 9].clone(),
                "gaze": agents[:, :, 4:6].clone()
            },
            "initial_landmarks": {
                "positions": landmarks[:, :, :2].clone(),
                "colors": landmarks[:, :, 6:9].clone(),
                "shapes": landmarks[:, :, 9].clone()
            },
            "agents": [
                {
                    "positions": [],
                    "gaze": [],
                    "utterances": []
                } for _ in range(self.num_agents)
            ]
        }

    def update(self, agent_idx, updated_agents, gaze, utterance):
        self.history["agents"][agent_idx]["positions"].append(updated_agents[:, agent_idx, :2].clone())
        self.history["agents"][agent_idx]["gaze"].append(gaze.clone())
        self.history["agents"][agent_idx]["utterances"].append(utterance.clone())

    def get_history(self):
        return self.history