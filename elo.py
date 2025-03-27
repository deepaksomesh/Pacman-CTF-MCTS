import os
import subprocess
import math
import re


# Elo rating system implementation
class EloRating:
    def __init__(self, k=32, initial_rating=1500):
        self.k = k
        self.ratings = {"myTeam": initial_rating, "RaveTeam": initial_rating}

    def expected_score(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update_ratings(self, winner, loser):
        expected_win = self.expected_score(self.ratings[winner], self.ratings[loser])
        expected_lose = self.expected_score(self.ratings[loser], self.ratings[winner])

        self.ratings[winner] += self.k * (1 - expected_win)
        self.ratings[loser] += self.k * (0 - expected_lose)

    def get_ratings(self):
        return self.ratings


# Function to run a match and return scores
def run_match():
    cmd = [
        "python", "capture.py",
        "-r", "raveteam",
        "-b", "myteam"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout
        print(output)

        red_score, blue_score = 0, 0  # Default scores

        blue_match = re.search(r"The Blue team wins by (\d+) points", output)
        red_match = re.search(r"The Red team wins by (\d+) points", output)
        red_all = re.search(r"The Red team has returned at least (\d+) of the opponents' dots", output)
        blue_all = re.search(r"The Blue team has returned at least (\d+) of the opponents' dots", output)

        if blue_match or blue_all:
            blue_score = int(blue_match.group(1))
        elif blue_all:
            blue_score = int(blue_match.group(1))
        elif red_match:
            red_score = int(red_match.group(1))
        elif red_all:
            red_score = int(red_all.group(1))

        return red_score, blue_score
    except subprocess.CalledProcessError as e:
        print("Error running the game:", e)
        return None, None


# Elo rating calculation loop
elo = EloRating()
num_matches = 3  # Number of matches to run

for i in range(num_matches):
    print(f"Running match {i + 1}...")
    red_score, blue_score = run_match()

    if red_score is None or blue_score is None:
        continue

    if red_score > blue_score:
        elo.update_ratings("myTeam", "RaveTeam")
    elif blue_score > red_score:
        elo.update_ratings("RaveTeam", "myTeam")

    print(f"Match {i + 1} result: Red {red_score} - Blue {blue_score}")
    print("Updated Elo Ratings:", elo.get_ratings())

# Final Elo ratings
print("\nFinal Elo Ratings:")
print("myTeam:", elo.get_ratings()["myTeam"])
print("RaveTeam:", elo.get_ratings()["RaveTeam"])
