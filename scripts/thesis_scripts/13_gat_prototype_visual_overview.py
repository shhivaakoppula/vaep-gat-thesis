# ==============================================================================
# 13_gat_prototype_visual_overview.py - GAT PROTOTYPE, COMPLETE VISUAL OVERVIEW
# ==============================================================================
# Builds every picture needed to show how the 34-match prototype data looks: the categorical data
# (named groups such as action type), the numerical data (numbers such as pitch position) and one real
# example graph. It does NOT train or evaluate any model. It loads the already verified data and draws it.
# All 34 prototype matches (train, validation and test together) are shown, because looking at the shape of
# the raw data is different from evaluating a model.
# Speed and memory: freeze frames are indexed by event uuid once per match (the original filtered a table of
# every player row for every action), and node rows are collected as arrays instead of one Python dict per
# player. The numbers and the pictures are the same.
# ==============================================================================

import torch  # imported first, before numpy and pandas: the fix for the Windows DLL conflict

import json  # reading the 360 files
import time  # progress timing

import numpy as np  # arrays
import pandas as pd  # tables
import matplotlib  # plotting library
matplotlib.use("Agg")  # draw to files only
import matplotlib.pyplot as plt  # pyplot interface
import matplotlib.patches as mpatches  # legend patches
from sklearn.neighbors import NearestNeighbors  # nearest neighbours for the example graph
from socceraction.data.statsbomb import StatsBombLoader  # reads the local StatsBomb files
from socceraction.spadl import statsbomb as statsbomb_spadl  # events -> SPADL actions
from socceraction import spadl  # SPADL helpers and lookup tables

import common  # shared helpers
from common import DATA_DIR, OUTPUT_DIR, print_header, require_file  # shared paths and helpers
import common_spadl as cs  # loader and SPADL actions

THREE_SIXTY_DIR = DATA_DIR / "three-sixty"  # 360 freeze-frame files
VISUALS_DIR = OUTPUT_DIR / "gat_prototype_visuals"  # where the pictures go
TARGET_FILE = OUTPUT_DIR / "rq1_literal_next10_target_34_matches.csv"  # prototype target
SPLIT_FILE = OUTPUT_DIR / "final_common_match_split.csv"  # prototype split
K_NEAREST_NEIGHBOURS = 5  # k used for the example graph
GOAL_X_NORM, GOAL_Y_NORM = 1.0, 0.5  # goal centre on the normalised pitch
EVENT_ID_CANDIDATES = ["original_event_id", "event_id"]  # SPADL columns that link back to the StatsBomb event uuid
TARGET_COLUMN = "target_score_next10_literal"  # label column


def safe_to_bool(value):
    """Convert a value to True/False by checking known spellings, because bool("False") would be True."""
    text = str(value).strip().lower()  # normalise
    if text in ("true", "1", "1.0"):  # known true spellings
        return True
    if text in ("false", "0", "0.0"):  # known false spellings
        return False
    raise ValueError(f"I do not recognise '{value}' as True or False. I need to extend safe_to_bool() to handle this value.")  # stop


def find_event_id_column(actions):  # name of the SPADL column that links an action to its event uuid
    for candidate in EVENT_ID_CANDIDATES:  # in order of preference
        if candidate in actions.columns:
            return candidate  # hand back candidate
    raise KeyError(f"I could not find my event-link column. Looked for {EVENT_ID_CANDIDATES}. Actual columns: {list(actions.columns)}")  # stop


def load_360_frames_for_match(game_id):
    """{event_uuid: (x, y, teammate, actor, keeper) arrays}, rows in file order. Empty dict when the file is missing."""
    frame_file = THREE_SIXTY_DIR / f"{int(game_id)}.json"  # path
    if not frame_file.exists():  # no 360 data for this match
        return {}
    with open(frame_file, "r", encoding="utf-8") as f:  # open
        raw_frames = json.load(f)  # parse
    collected = {}  # uuid -> five lists
    for entry in raw_frames:  # one entry per event
        event_uuid = entry.get("event_uuid")  # which event
        for player_entry in entry.get("freeze_frame", []):  # every visible player
            location = player_entry.get("location", [None, None])  # [x, y]
            rec = collected.setdefault(event_uuid, ([], [], [], [], []))  # lists for this event (repeated uuids keep appending, as before)
            rec[0].append(location[0]); rec[1].append(location[1])  # position
            rec[2].append(bool(player_entry.get("teammate", False))); rec[3].append(bool(player_entry.get("actor", False))); rec[4].append(bool(player_entry.get("keeper", False)))  # flags
    return {u: (np.array(r[0], dtype=float), np.array(r[1], dtype=float), np.array(r[2], dtype=float),
                np.array(r[3], dtype=float), np.array(r[4], dtype=float)) for u, r in collected.items() if u is not None}  # None never matched before


def load_everything():  # reads the target, the split and the match metadata
    print_header("LOAD MY TARGET, SPLIT, AND ACTION TYPE NAMES")  # title
    require_file(TARGET_FILE, "Prototype target")  # input 1
    require_file(SPLIT_FILE, "Prototype split")  # input 2
    target = pd.read_csv(TARGET_FILE)  # target table
    split = pd.read_csv(SPLIT_FILE)  # split table
    # The real columns are printed before assuming anything about the schema of this older file.
    print("Actual columns in my prototype target file:")
    print(list(target.columns))  # print the result of list(target.columns)
    print()  # print a blank line
    if "type_id" in target.columns and "result_id" in target.columns:  # the names can be built from the ids
        actiontypes, results = spadl.actiontypes_df(), spadl.results_df()  # lookup tables
        target["type_name"] = target["type_id"].map(dict(zip(actiontypes["type_id"], actiontypes["type_name"])))  # readable action type
        target["result_name"] = target["result_id"].map(dict(zip(results["result_id"], results["result_name"])))  # readable result
        print("[OK] type_id and result_id found - I built type_name and result_name columns for my action-type chart.")
    else:  # older schema
        print("[NOTE] My prototype target file does not have type_id / result_id columns. I cannot build my action-type "
              "distribution chart (Chart 1) from this file alone. I will rebuild action types directly from the raw match "
              "data instead, in the same step where I already reconstruct actions for my other charts, rather than "
              "assuming this file has them.")
        target["type_name"] = None  # no action types in this file, filled in later
        target["result_name"] = None  # no result names in this file, filled in later
    target = target.merge(split, on="game_id", how="left")  # attach the split columns
    print()
    print(f"Target rows loaded: {len(target):,}")  # tell the user: Target rows loaded (plus its value)
    print(f"Matches in split: {len(split):,}")  # tell the user: Matches in split (plus its value)
    return target, split  # hand back target, split


def chart_action_type_distribution(type_name_counter):  # draws chart 1: how often each action type occurs
    print_header("CHART 1 - ACTION TYPE DISTRIBUTION (CATEGORICAL)")  # title
    if not type_name_counter:  # nothing counted
        print("[SKIP] I collected no action type names while reconstructing my matches - I cannot draw this chart without them.")
        return  # leave the function
    counts = pd.Series(type_name_counter).sort_values(ascending=True)  # smallest first so the biggest bar is on top
    print("What this chart shows: how many of each kind of action")
    print("(pass, carry, shot, and so on) appear across all my")  # tell the user: (pass, carry, shot, and so on) appear across all my
    print("34 prototype matches. This is categorical data - each")  # tell the user: 34 prototype matches. This is categorical data - each
    print("action falls into one named group, not a number. I")  # tell the user: action falls into one named group, not a number. I
    print("built this directly from my raw reconstructed match")  # tell the user: built this directly from my raw reconstructed match
    print("actions, not from my target CSV, since my target CSV")  # tell the user: actions, not from my target CSV, since my target CSV
    print("did not contain action type names.")  # tell the user: did not contain action type names
    print()  # print a blank line
    print(counts.to_string())  # print the result of counts.to_string()
    fig, ax = plt.subplots(figsize=(9, 6))  # canvas
    ax.barh(counts.index, counts.values, color="#3b6ea5")  # horizontal bars
    ax.set_xlabel("Number of actions")
    ax.set_title("Action Type Distribution Across My 34 Prototype Matches")  # chart title
    for i, v in enumerate(counts.values):  # value label at the end of each bar
        ax.text(v, i, f" {v:,}", va="center", fontsize=8)
    plt.tight_layout()  # trim the margins
    plt.savefig(VISUALS_DIR / "01_action_type_distribution.png", dpi=150)  # save the picture to the figures folder
    plt.close()  # close the figure to free memory
    print(f"Saved: {VISUALS_DIR / '01_action_type_distribution.png'}")  # tell the user: Saved (plus its value)


def chart_target_distribution(target):  # draws chart 2: how unbalanced the target label is
    print_header("CHART 2 - TARGET LABEL DISTRIBUTION (CATEGORICAL)")  # title
    raw_column = target[TARGET_COLUMN]  # the label column, whatever its stored type
    print(f"Raw dtype of my target column: {raw_column.dtype}")
    print(f"Raw unique values found: {raw_column.unique().tolist()[:5]} (showing up to 5)")  # tell the user: Raw unique values found: (showing up to 5) (plus its value)
    target_bool = raw_column.astype(str).str.strip().str.lower().map(
        {"true": True, "1": True, "1.0": True, "false": False, "0": False, "0.0": False})  # explicit conversion
    n_unmapped = target_bool.isna().sum()  # rows with an unknown spelling
    if n_unmapped > 0:
        print(f"[WARNING] {n_unmapped} rows did not match any of my expected True/False representations. "  # tell the user: [WARNING] rows did not match any of my expected True/False... (plus its value)
              f"I am treating these as missing, not silently as False.")
    counts = target_bool.value_counts()  # count of each label value
    positive = int(counts.get(True, 0))  # number of positive labels
    negative = int(counts.get(False, 0))  # number of negative labels
    print()  # print a blank line
    print("What this chart shows: how many of my eligible actions")  # tell the user: What this chart shows: how many of my eligible actions
    print("are labelled positive (a goal followed within 10 actions)")  # tell the user: are labelled positive (a goal followed within 10 actions)
    print("versus negative (no goal followed). This shows how rare")  # tell the user: versus negative (no goal followed). This shows how rare
    print("the positive class is - a fact I must handle carefully")  # tell the user: the positive class is - a fact I must handle carefully
    print("when I train my model.")  # tell the user: when I train my model
    print()  # print a blank line
    print(f"Positive: {positive:,}")  # tell the user: Positive (plus its value)
    print(f"Negative: {negative:,}")  # tell the user: Negative (plus its value)
    total = positive + negative
    if total > 0:  # only compute the rate when there is data
        print(f"Positive rate: {positive / total * 100:.4f}%")  # tell the user: Positive rate: % (plus its value)
    else:
        print("[SKIP] I found 0 positive and 0 negative rows after conversion - my mapping above did not match this "  # tell the user: [SKIP] I found 0 positive and 0 negative rows after conversion -...
              "column's actual values. I need to look at the raw unique values printed above and extend my mapping.")
        return  # leave the function
    fig, ax = plt.subplots(figsize=(6, 6))
    labels = [f"Negative\n{negative:,}", f"Positive\n{positive:,}"]  # text for the two bars
    values = [negative, positive]  # heights of the two bars
    ax.bar(labels, values, color=["#8c8c8c", "#c0392b"])  # draw the two bars
    ax.set_ylabel("Number of actions")  # y axis label
    ax.set_title("My Target Label — Extremely Imbalanced")  # chart title
    for i, v in enumerate(values):  # loop over the bars
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=10)  # write the count above each bar
    plt.tight_layout()  # trim the margins
    plt.savefig(VISUALS_DIR / "02_target_label_distribution.png", dpi=150)  # save the picture to the figures folder
    plt.close()  # close the figure to free memory
    print(f"Saved: {VISUALS_DIR / '02_target_label_distribution.png'}")  # tell the user: Saved (plus its value)


def chart_split_distribution(split):  # draws chart 3: how many matches are in each split
    print_header("CHART 3 - MATCH SPLIT DISTRIBUTION (CATEGORICAL)")  # title
    counts = split["split"].value_counts()
    print(counts.to_string())  # print the result of counts.to_string()
    fig, ax = plt.subplots(figsize=(6, 6))
    colours = {"train": "#3b6ea5", "validation": "#e0a029", "test": "#c0392b"}  # colour of each split
    labels = counts.index.tolist()  # split names for the x axis
    values = counts.values.tolist()  # number of matches per split
    ax.bar(labels, values, color=[colours.get(lbl, "#666666") for lbl in labels])  # draw one bar per split
    ax.set_ylabel("Number of matches")  # y axis label
    ax.set_title("My 34 Prototype Matches — Split Into Three Groups")  # chart title
    for i, v in enumerate(values):  # loop over the bars
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=11)  # write the count above each bar
    plt.tight_layout()  # trim the margins
    plt.savefig(VISUALS_DIR / "03_match_split_distribution.png", dpi=150)  # save the picture to the figures folder
    plt.close()  # close the figure to free memory
    print(f"Saved: {VISUALS_DIR / '03_match_split_distribution.png'}")  # tell the user: Saved (plus its value)


def chart_actions_per_match(target):  # draws chart 4: how many eligible actions each match has
    print_header("CHART 4 - ACTIONS PER MATCH (NUMERICAL)")  # title
    counts = target.groupby("game_id").size()  # eligible actions per match
    print("What this chart shows: how many eligible actions each")
    print("of my 34 matches contains. This is numerical data - a")  # tell the user: of my 34 matches contains. This is numerical data - a
    print("count that can take any value, not a named category.")  # tell the user: count that can take any value, not a named category
    print()  # print a blank line
    print(f"Minimum actions in one match: {counts.min():,}")  # tell the user: Minimum actions in one match (plus its value)
    print(f"Maximum actions in one match: {counts.max():,}")  # tell the user: Maximum actions in one match (plus its value)
    print(f"Mean actions per match: {counts.mean():.1f}")  # tell the user: Mean actions per match (plus its value)
    print(f"Median actions per match: {counts.median():.1f}")  # tell the user: Median actions per match (plus its value)
    print(f"Standard deviation: {counts.std():.1f}")  # tell the user: Standard deviation (plus its value)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(counts.values, bins=15, color="#3b6ea5", edgecolor="black")  # draw the histogram of actions per match
    ax.set_xlabel("Eligible actions in one match")  # x axis label
    ax.set_ylabel("Number of matches")  # y axis label
    ax.set_title("How Many Eligible Actions Each of My 34 Matches Has")  # chart title
    plt.tight_layout()  # trim the margins
    plt.savefig(VISUALS_DIR / "04_actions_per_match.png", dpi=150)  # save the picture to the figures folder
    plt.close()  # close the figure to free memory
    print(f"Saved: {VISUALS_DIR / '04_actions_per_match.png'}")  # tell the user: Saved (plus its value)


def build_player_level_dataset(loader, metadata, target, match_ids):
    """Collect player rows of every usable graph for the pitch and feature charts (same graph logic as the training scripts)."""
    print_header("BUILD PLAYER-LEVEL DATA FOR MY PITCH AND FEATURE CHARTS")  # title
    home = dict(zip(metadata["game_id"], metadata["home_team_id"]))  # game_id -> home team id
    node_blocks = []  # one (n_players x 7) array per usable graph
    target_blocks = []  # one label per usable graph
    node_counts_per_graph = []  # players per graph (counted before the actor check, as before)
    example_graph_data = None  # first graph with at least 8 players
    type_name_counter = {}  # action type name -> count over all actions
    start_time = time.time()
    for number, game_id in enumerate(sorted(match_ids), start=1):  # one match at a time
        if game_id not in home:  # no metadata for this match
            continue
        try:  # a broken match is skipped
            home_team_id = int(home[game_id])
            events = loader.events(game_id=int(game_id))  # raw events
            if events is None or events.empty:
                continue  # skip to the next item
            actions = statsbomb_spadl.convert_to_actions(events, home_team_id).reset_index(drop=True)  # SPADL actions in order
            actions["action_index"] = range(len(actions))  # position inside the match
            actions = spadl.add_names(actions)  # readable type_name and result_name
            for tname, tcount in actions["type_name"].value_counts().items():  # count every action type of the full match
                type_name_counter[tname] = type_name_counter.get(tname, 0) + tcount
            event_id_column = find_event_id_column(actions)  # link column
            frames = load_360_frames_for_match(game_id)  # freeze frames by event uuid
            if not frames:  # nothing usable
                continue
            target_for_match = target[target["game_id"] == game_id]  # rows of this match
            eligible_actions = actions.merge(target_for_match, on="action_index", how="inner")  # eligible actions in action order
            for event_uuid, label_value in zip(eligible_actions[event_id_column].to_numpy(), eligible_actions[TARGET_COLUMN].to_numpy()):  # arrays, no iterrows
                frame = frames.get(event_uuid)  # freeze frame of this action
                if frame is None or len(frame[0]) < 2:  # fewer than two players
                    continue
                x_raw, y_raw, teammate, actor, keeper = frame  # unpack
                n_players = len(x_raw)
                node_counts_per_graph.append(n_players)  # counted for chart 6
                x_norm = x_raw / 120.0  # pitch length
                y_norm = y_raw / 80.0  # pitch width
                if actor.sum() != 1:  # exactly one actor is required
                    continue
                dx = GOAL_X_NORM - x_norm  # horizontal distance to the goal
                dy = GOAL_Y_NORM - y_norm  # vertical distance to the goal
                distance_goal = np.sqrt(dx ** 2 + dy ** 2)  # distance to goal
                angle_goal = np.arctan2(np.abs(dy), np.abs(dx)) / np.pi  # angle to goal
                node_blocks.append(np.stack([x_norm, y_norm, teammate, keeper, actor, distance_goal, angle_goal], axis=1))  # 7 columns per player
                target_blocks.append((n_players, safe_to_bool(label_value)))  # label (converted strictly, as before)
                if example_graph_data is None and n_players >= 8:  # keep the first graph with at least 8 players
                    example_graph_data = {"x_norm": x_norm, "y_norm": y_norm, "teammate": teammate, "keeper": keeper, "actor": actor, "game_id": int(game_id)}
        except Exception as ex:  # report and go on
            print(f"  [SKIP - match {game_id}] {type(ex).__name__}: {ex}")
            continue  # skip to the next item
        if number % 10 == 0 or number == len(match_ids):  # progress line
            print(f"  [{number}/{len(match_ids)}] matches processed. Elapsed: {(time.time() - start_time)/60:.1f} min.")

    if node_blocks:  # build the player table in one go
        nodes = np.concatenate(node_blocks, axis=0)  # all players of all graphs
        node_df = pd.DataFrame(nodes, columns=["x_norm", "y_norm", "teammate", "keeper", "actor", "distance_goal", "angle_goal"])  # float64 columns
        node_df["target"] = np.repeat(np.array([t for _, t in target_blocks], dtype=bool), [n for n, _ in target_blocks])  # label repeated for every player of the graph
    else:  # nothing collected
        node_df = pd.DataFrame()
    print()  # print a blank line
    print(f"Total player-position rows collected: {len(node_df):,}")  # tell the user: Total player-position rows collected (plus its value)
    print(f"Total graphs (eligible actions with 360 data) seen: {len(node_counts_per_graph):,}")  # tell the user: Total graphs (eligible actions with 360 data) seen (plus its value)
    return node_df, node_counts_per_graph, example_graph_data, type_name_counter  # hand back node_df, node_counts_per_graph, example_graph_data, type_...


def chart_pitch_heatmap(node_df):  # draws chart 5: where the players stand on the pitch
    print_header("CHART 5 - PITCH POSITION HEATMAP (NUMERICAL)")  # title
    print("What this chart shows: where on the pitch my player")
    print("positions are concentrated, across every graph I built")  # tell the user: positions are concentrated, across every graph I built
    print("for my 34 prototype matches.")  # tell the user: for my 34 prototype matches
    fig, ax = plt.subplots(figsize=(10, 7))
    h = ax.hist2d(node_df["x_norm"] * 120, node_df["y_norm"] * 80, bins=[60, 40], cmap="viridis")  # 2-D histogram of positions
    plt.colorbar(h[3], ax=ax, label="Number of player positions")
    ax.set_xlabel("Pitch length (metres, 0 = own goal, 120 = opponent goal)")  # x axis label
    ax.set_ylabel("Pitch width (metres)")  # y axis label
    ax.set_title("Where My Player Positions Fall Across the Pitch")  # chart title
    plt.tight_layout()  # trim the margins
    plt.savefig(VISUALS_DIR / "05_pitch_position_heatmap.png", dpi=150)  # save the picture to the figures folder
    plt.close()  # close the figure to free memory
    print(f"Saved: {VISUALS_DIR / '05_pitch_position_heatmap.png'}")  # tell the user: Saved (plus its value)


def chart_players_per_graph(node_counts_per_graph):  # draws chart 6: how many players each freeze frame shows
    print_header("CHART 6 - PLAYERS PER GRAPH (NUMERICAL)")  # title
    counts = np.array(node_counts_per_graph)
    print(f"Minimum players in a graph: {counts.min()}")  # tell the user: Minimum players in a graph (plus its value)
    print(f"Maximum players in a graph: {counts.max()}")  # tell the user: Maximum players in a graph (plus its value)
    print(f"Mean players per graph: {counts.mean():.2f} (out of 22 players who are actually on the pitch)")  # tell the user: Mean players per graph: (out of 22 players who are actually on the... (plus its value)
    print(f"Median players per graph: {np.median(counts):.1f}")  # tell the user: Median players per graph (plus its value)
    print(f"Standard deviation: {counts.std():.2f}")  # tell the user: Standard deviation (plus its value)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(counts, bins=range(0, 23), color="#3b6ea5", edgecolor="black")  # draw the histogram of players per graph
    ax.axvline(counts.mean(), color="#c0392b", linestyle="--", label=f"Mean = {counts.mean():.1f}")  # mark the average with a dashed line
    ax.set_xlabel("Visible players in one graph (out of 22 on the pitch)")  # x axis label
    ax.set_ylabel("Number of graphs")  # y axis label
    ax.set_title("How Many Players My 360 Data Actually Shows Me Per Graph")  # chart title
    ax.legend()  # show the legend
    plt.tight_layout()  # trim the margins
    plt.savefig(VISUALS_DIR / "06_players_per_graph.png", dpi=150)  # save the picture to the figures folder
    plt.close()  # close the figure to free memory
    print(f"Saved: {VISUALS_DIR / '06_players_per_graph.png'}")  # tell the user: Saved (plus its value)


def chart_node_feature_distributions(node_df):  # draws chart 7: the distribution of every node feature
    print_header("CHART 7 - MY 7 NODE FEATURES (NUMERICAL AND CATEGORICAL)")  # title
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()  # flatten the grid of panels into a list
    numeric_features = [("x_norm", "x position (0 to 1)"), ("y_norm", "y position (0 to 1)"),  # the four numeric features and their titles (continues on the next line)
                        ("distance_goal", "distance to goal (normalised)"), ("angle_goal", "angle to goal (normalised)")]
    for ax, (col, label) in zip(axes[:4], numeric_features):  # four histograms
        ax.hist(node_df[col], bins=40, color="#3b6ea5", edgecolor="none")
        ax.set_title(label, fontsize=10)  # chart title for this panel
        ax.set_xlabel(col)  # x axis label for this panel
    categorical_features = [("teammate", "Teammate flag (1) vs Opponent (0)"), ("keeper", "Goalkeeper flag"),  # the three flag features and their titles (continues on the next line)
                            ("actor", "Actor flag (the player performing the action)")]
    for ax, (col, label) in zip(axes[4:7], categorical_features):  # three bar charts
        counts = node_df[col].value_counts().sort_index()
        ax.bar(["0", "1"], [counts.get(0.0, 0), counts.get(1.0, 0)], color=["#8c8c8c", "#3b6ea5"])  # draw the 0/1 counts as two bars
        ax.set_title(label, fontsize=10)  # chart title for this panel
    axes[7].axis("off")  # unused panel
    plt.suptitle("My 7 Node Features — What Every Player in Every Graph Carries", fontsize=13)
    plt.tight_layout()  # trim the margins
    plt.savefig(VISUALS_DIR / "07_node_feature_distributions.png", dpi=150)  # save the picture to the figures folder
    plt.close()  # close the figure to free memory
    print(f"Saved: {VISUALS_DIR / '07_node_feature_distributions.png'}")  # tell the user: Saved (plus its value)


def chart_distance_by_target(node_df):  # draws chart 8: distance to goal for positive versus negative targets
    print_header("CHART 8 - DISTANCE TO GOAL, POSITIVE VS NEGATIVE TARGET")  # title
    print("What this chart shows: whether actions that lead to a")
    print("goal happen closer to the opponent's goal than actions")  # tell the user: goal happen closer to the opponent's goal than actions
    print("that do not. This is my sanity check that my distance")  # tell the user: that do not. This is my sanity check that my distance
    print("feature carries real, useful information.")  # tell the user: feature carries real, useful information
    positive = node_df[node_df["target"] == True]["distance_goal"]  # players in graphs followed by a goal
    negative = node_df[node_df["target"] == False]["distance_goal"]  # players in graphs not followed by a goal
    print(f"Mean distance to goal, POSITIVE actions: {positive.mean():.4f}")
    print(f"Mean distance to goal, NEGATIVE actions: {negative.mean():.4f}")  # tell the user: Mean distance to goal, NEGATIVE actions (plus its value)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot([negative.dropna(), positive.dropna()], tick_labels=["Negative\n(no goal followed)", "Positive\n(goal followed)"])  # new matplotlib argument name
    ax.set_ylabel("Distance to opponent's goal (normalised)")
    ax.set_title("Distance to Goal — Positive vs Negative Target")  # chart title
    plt.tight_layout()  # trim the margins
    plt.savefig(VISUALS_DIR / "08_distance_by_target.png", dpi=150)  # save the picture to the figures folder
    plt.close()  # close the figure to free memory
    print(f"Saved: {VISUALS_DIR / '08_distance_by_target.png'}")  # tell the user: Saved (plus its value)


def chart_example_graph(example_graph_data):
    """The single picture to show the supervisor: one graph on a pitch. Cyan = teammates, red = opponents,
    white = the actor, gold = the goalkeeper, and a line from each player to their 5 nearest neighbours."""
    print_header("CHART 9 - ONE EXAMPLE GRAPH ON A PITCH")  # title
    if example_graph_data is None:
        print("[SKIP] I did not find a suitable example graph with at least 8 players to draw.")  # tell the user: [SKIP] I did not find a suitable example graph with at least 8...
        return  # leave the function
    x_norm, y_norm = example_graph_data["x_norm"], example_graph_data["y_norm"]  # positions
    teammate, keeper, actor = example_graph_data["teammate"], example_graph_data["keeper"], example_graph_data["actor"]  # flags
    game_id = example_graph_data["game_id"]
    n_players = len(x_norm)  # number of players in the example graph
    k = min(K_NEAREST_NEIGHBOURS, n_players - 1)  # neighbours per player
    positions = np.stack([x_norm, y_norm], axis=1)
    _, indices = NearestNeighbors(n_neighbors=k + 1).fit(positions).kneighbors(positions)  # neighbour lists
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor("#2d7a27")  # grass green
    ax.plot([0, 1], [0, 0], "white", linewidth=1, alpha=0.5)  # touchlines and halfway line
    ax.plot([0, 1], [1, 1], "white", linewidth=1, alpha=0.5)
    ax.plot([0, 0], [0, 1], "white", linewidth=1, alpha=0.5)  # draw the left goal line
    ax.plot([1, 1], [0, 1], "white", linewidth=1, alpha=0.5)  # draw the right goal line
    ax.plot([0.5, 0.5], [0, 1], "white", linewidth=1, alpha=0.3)  # draw the halfway line
    for i in range(n_players):  # edges: yellow for same team, red across teams
        for j_idx in range(1, k + 1):
            j = indices[i, j_idx]  # the neighbour of node i
            colour = "yellow" if teammate[i] == teammate[j] else "red"  # edges between teammates are yellow, others red
            ax.plot([positions[i, 0], positions[j, 0]], [positions[i, 1], positions[j, 1]], color=colour, alpha=0.3, linewidth=1)  # draw one edge
    for i in range(n_players):  # players on top of the edges
        if actor[i] == 1:
            colour, size, zorder = "white", 260, 6  # colour, size and layer for the actor (large white dot)
        elif keeper[i] == 1:  # otherwise, if the player is the goalkeeper
            colour, size, zorder = "gold", 190, 5  # colour for the goalkeeper (gold)
        elif teammate[i] == 1:  # otherwise, if the player is a teammate
            colour, size, zorder = "cyan", 150, 4  # colour for a teammate (cyan)
        else:
            colour, size, zorder = "red", 150, 4  # colour for an opponent (red)
        ax.scatter(positions[i, 0], positions[i, 1], c=colour, s=size, zorder=zorder, edgecolors="black", linewidth=0.8)  # draw one player
    ax.legend(handles=[mpatches.Patch(color="white", label="Actor (performing the action)"), mpatches.Patch(color="cyan", label="Teammate"),  # add the legend (continues on the next lines)
                       mpatches.Patch(color="red", label="Opponent"), mpatches.Patch(color="gold", label="Goalkeeper")],
              loc="upper left", fontsize=9, facecolor="white")
    ax.set_xlim(-0.05, 1.05)  # keep the x range slightly wider than the pitch
    ax.set_ylim(-0.05, 1.05)  # keep the y range slightly wider than the pitch
    ax.set_xlabel("Pitch length, normalised (0 = own goal, 1 = opponent goal)", color="white")  # x axis label
    ax.set_ylabel("Pitch width, normalised", color="white")  # y axis label
    ax.set_title(f"One Example Graph From My Prototype Data\n{n_players} visible players, each connected to their {k} nearest neighbours", color="white")  # chart title
    ax.tick_params(colors="white")  # white tick marks
    fig.patch.set_facecolor("#1a1a2e")  # dark background for the whole figure
    plt.tight_layout()  # trim the margins
    plt.savefig(VISUALS_DIR / "09_example_graph.png", dpi=150, bbox_inches="tight")  # save the picture, with a tight bounding box
    plt.close()  # close the figure to free memory
    print(f"Saved: {VISUALS_DIR / '09_example_graph.png'}")  # tell the user: Saved (plus its value)
    print(f"This example is from match {game_id}, chosen because it has {n_players} visible players - enough to show my design clearly.")  # tell the user: This example is from match , chosen because it has visible players... (plus its value)


def main():  # runs the whole step from start to finish
    common.verify_environment()  # optional interpreter check
    print_header("GAT PROTOTYPE - COMPLETE VISUAL OVERVIEW")  # title
    VISUALS_DIR.mkdir(parents=True, exist_ok=True)  # output folder
    target, split = load_everything()  # inputs
    chart_target_distribution(target)  # chart 2
    chart_split_distribution(split)  # chart 3
    chart_actions_per_match(target)  # chart 4
    loader = StatsBombLoader(getter="local", root=str(DATA_DIR))  # StatsBomb reader (silent, like the original)
    match_ids = split["game_id"].tolist()
    metadata = cs.find_match_metadata_lenient(loader, match_ids)  # home team ids
    node_df, node_counts_per_graph, example_graph_data, type_name_counter = build_player_level_dataset(loader, metadata, target, match_ids)  # player rows
    chart_action_type_distribution(type_name_counter)  # chart 1
    if not node_df.empty:  # charts 5 to 8 need player data
        chart_pitch_heatmap(node_df)
        chart_players_per_graph(node_counts_per_graph)  # draw chart 6
        chart_node_feature_distributions(node_df)  # draw chart 7
        chart_distance_by_target(node_df)  # draw chart 8
    else:
        print("[WARNING] No player-level data was collected - skipping charts 5 through 8.")  # tell the user: [WARNING] No player-level data was collected - skipping charts 5...
    chart_example_graph(example_graph_data)  # chart 9
    print_header("ALL CHARTS COMPLETE")
    print(f"Every chart is saved in:\n{VISUALS_DIR}")  # tell the user: Every chart is saved in (plus its value)
    print()  # print a blank line
    print("What I have now, for my document:")  # tell the user: What I have now, for my document
    print("  01 - Action type distribution (categorical)")  # tell the user: 01 - Action type distribution (categorical)
    print("  02 - Target label distribution (categorical)")  # tell the user: 02 - Target label distribution (categorical)
    print("  03 - Train/validation/test split (categorical)")  # tell the user: 03 - Train/validation/test split (categorical)
    print("  04 - Actions per match (numerical)")  # tell the user: 04 - Actions per match (numerical)
    print("  05 - Pitch position heatmap (numerical)")  # tell the user: 05 - Pitch position heatmap (numerical)
    print("  06 - Players per graph distribution (numerical)")  # tell the user: 06 - Players per graph distribution (numerical)
    print("  07 - My 7 node features, all together")  # tell the user: 07 - My 7 node features, all together
    print("  08 - Distance to goal, positive vs negative target")  # tell the user: 08 - Distance to goal, positive vs negative target
    print("  09 - One example graph on a pitch, showing my k=5 design")  # tell the user: 09 - One example graph on a pitch, showing my k=5 design
    return 0  # success


if __name__ == "__main__":  # only when run directly
    common.run_main(main, "GAT VISUAL OVERVIEW")  # shared error handling
