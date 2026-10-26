'''
Object that keeps track of the different evaluation scenarios.
'''



import json
import os
from datetime import datetime

class ScenarioScheduler:

    def __init__(self, path, max_retries):

        with open(path) as f:
            self.scenarios = json.load(f)["scenarios"]
        self.max_retries = max_retries
        self.index = 0
        self.attempt = 0
        self.results = []
        self.stats = dict(car_collisions=0, wall_collisions=0, other_failures=0, overtakes_total=0, laps_total=0)
        self.scenario_file = path


    def reset_stats(self):
        self.stats = dict(car_collisions=0, wall_collisions=0, other_failures=0, overtakes_total=0, laps_total=0)

    def finished(self):
        return self.index >= len(self.scenarios)

    def current_scenario(self):
        return self.scenarios[self.index]

    def end(self, outcome, lap_times, overtakes):
        self.stats["laps_total"] += len(lap_times)
        self.stats["overtakes_total"] += overtakes
        if outcome == "car_collision":
            self.stats["car_collisions"] += 1
        elif outcome == "wall_collision":
            self.stats["wall_collisions"] += 1
        elif outcome != "completed":
            self.stats["other_failures"] += 1

        if outcome == "completed" or self.attempt >= self.max_retries:
            self.record(outcome, lap_times, overtakes)
            self.index += 1
            self.attempt = 0
            self.reset_stats()
        else:
            self.attempt += 1

    def record(self, outcome, lap_times, overtakes):
        self.results.append(dict(
            scenario=self.index,
            seed=self.current_scenario().get("seed"),
            completed=(outcome == "completed"),
            final_try=self.attempt + 1,
            tries_allowed=self.max_retries + 1,
            final_outcome=outcome,
            final_lap_times=[float(t) for t in lap_times],
            final_overtakes=int(overtakes),
            **self.stats,
        ))

    def save(self, folder, meta):
        os.makedirs(folder, exist_ok=True)
        name =  f"scenario_results_{meta.get('controller')}_{datetime.now():%Y-%m-%d_%H-%M-%S}.json"
        path = os.path.join(folder, name)
        with open(path, "w") as f:
            json.dump({"meta": meta, "scenarios": self.results}, f, indent=2)
        print(f"Scenario results saved to {path}")
        return path

    