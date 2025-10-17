import os
import sys
import argparse
import random
import json
import copy
import time
import math

class TreeNode:
    def __init__(self, file_id, orig_name=None):
        self.file_id = file_id
        self.orig_name = orig_name
        self.children = []

def build_tree(file_names):
    file_map = {}  # Map file IDs to their respective TreeNode objects
    root_candidates = set()  # Store potential root candidates

    # First pass: create tree nodes for each file
    for file_name in file_names:
        parts = file_name.split(',')
        file_id = parts[0].split(':')[1]
        orig_name = None
        for part in parts[1:]:
            if 'orig:' in part:
                orig_name = part.split(':')[1].split('_')[0]  # Extract substring between "orig:" and "_"
                break
        if 'src:' not in file_name or '+' not in file_name.split('src:')[1]:
            root_candidates.add(file_id)  # Add files without two groups of numbers in src to root candidates
        file_map[file_id] = TreeNode(file_id, orig_name)

    # Second pass: build the tree structure
    for file_name in file_names:
        parts = file_name.split(',')
        file_id = parts[0].split(':')[1]
        src_id = None
        for part in parts[1:]:
            key_value = part.split(':')
            if len(key_value) == 2:
                key, value = key_value
                if key == 'src' and '+' not in value:
                    src_id = value
                    break
        if src_id:
            file_map[src_id].children.append(file_map[file_id])
            root_candidates.discard(file_id)  # Remove src files from root candidates

    # Find the root nodes
    roots = [file_map[root_id] for root_id in root_candidates]

    return roots

def print_tree(root, depth=0):
    if root is None:
        return 0
    # print('  ' * depth + '- ' + root.file_id)
    count = 1
    for child in root.children:
        count += print_tree(child, depth + 1)
    return count

def list_files(path):
    file_list = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    return file_list


# -------------------- Bandit selector (replaces random generator pick) --------------------
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

@dataclass
class _GenStats:
    pulls: int = 0
    total_reward: float = 0.0
    last_used_ts: float = 0.0

class BanditGeneratorSelector:
    """
    UCB1 + ε-greedy with a small freshness bonus; state persisted to JSON.
    """
    def __init__(self, generator_ids: List[str], state_path: Optional[str], c: float = 1.4, epsilon: float = 0.08, freshness_gamma: float = 0.15):
        self.c = c
        self.epsilon = epsilon
        self.freshness_gamma = freshness_gamma
        self.state_path = state_path
        self.start_ts = time.time()
        self.rng = random.Random()
        self.allowed: set[str] = set(generator_ids)
        self.stats: Dict[str, _GenStats] = {}
        self._load_state()
        # Keep only current IDs; drop stale ones, and add missing ones.
        loaded = self.stats
        self.stats = {gid: loaded.get(gid, _GenStats()) for gid in generator_ids}

    def _save_state(self):
        if not self.state_path:
            return
        blob = {gid: asdict(s) for gid, s in self.stats.items()}
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(blob, f, indent=2)

    def _load_state(self):
        if not self.state_path:
            return
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for gid, d in raw.items():
                self.stats[gid] = _GenStats(**d)
        except (FileNotFoundError, json.JSONDecodeError):
            self.stats = {}

    def select(self) -> str:
        ids = list(self.stats.keys())
        unexplored = [gid for gid, s in self.stats.items() if s.pulls == 0]
        if unexplored:
            return self.rng.choice(unexplored)
        if self.rng.random() < self.epsilon:
            return self.rng.choice(ids)
        total_pulls = sum(s.pulls for s in self.stats.values()) or 1
        now = time.time()
        best_gid, best_score = None, float("-inf")
        for gid, s in self.stats.items():
            mean = (s.total_reward / s.pulls) if s.pulls else 0.0
            ucb = mean + self.c * math.sqrt(math.log(total_pulls) / max(1, s.pulls))
            freshness = self.freshness_gamma * ((now - s.last_used_ts) / (now - self.start_ts + 1.0))
            score = ucb + freshness
            if score > best_score:
                best_gid, best_score = gid, score
        return best_gid

    def update(self, gid: str, reward: float):
        s = self.stats.setdefault(gid, _GenStats())
        s.pulls += 1
        s.total_reward += float(reward)
        s.last_used_ts = time.time()
        self._save_state()

def _scheduler_state_path(mutation_log: str) -> str:
    return os.path.join(mutation_log, "gen_scheduler_state.json")

def _success_reward(ok: bool) -> float:
    # Simple proxy: reward success at 1.0, failure at 0.0
    return 1.0 if ok else 0.0

# -------------------- Improved prompts and system message --------------------

generator_mutation_feature_prompt_init = '''
```
<TARGET_GENERATOR>
```

Goal
----
Produce a **more advanced generator** (same target language / file format as `<FROMAT>`) that extends the given generator with **richer, realistic file features**. Focus on producing code that is runnable, well-documented, and testable.

Requirements
- Maintain the same file format and relevant conventions as the original `<TARGET_GENERATOR>`.
- Add specific, realistic file features (describe them in the short header comment of the generated code).
- Include inline comments, input/output examples, and a small self-test or `if __name__ == "__main__":` demo that creates sample files.
- Avoid external network calls and non-standard system dependencies. If a package is necessary, list it at the top in a comment and keep it minimal.
- The assistant's output MUST be a single Markdown code block containing only the final Python code (no extra prose), following the template below.

Template
--------
Here's an extended version of the code that generates a `<FROMAT>` file with additional, more complex file features such as `<specific file features>`:
```
<Generated Code>
```
'''

generator_mutation_feature_prompt_incre = '''
Goal
----
Incrementally enhance the provided generator for `<FROMAT>` files by adding **additional, progressively complex file features**.

Requirements
- Start from the given `<TARGET_GENERATOR>` behavior and extend it; preserve backwards compatibility where feasible.
- Provide a clear brief comment at the top describing the new features added.
- Include small automated checks or examples demonstrating the new features.
- Do not call external services or require heavy external dependencies. If a third-party library is used, include a fallback pure-Python implementation or an explanation.
- Output format: a single Markdown code block containing only the updated generator code.

Template
--------
Here's an extended version of the code that generates a `<FROMAT>` file with additional more complex file features such as `<specific file features>`:
```
<Generated Code>
```
'''

generator_mutation_structure_prompt_init = '''
```
<TARGET_GENERATOR>
```

Goal
----
Create a stronger generator variant that produces `<FROMAT>` files with **more complex internal structure** (nested sections, cross-references, metadata blocks, checksums, multiple segments, etc.) while keeping the same format family.

Requirements
- Preserve the original file format's compatibility where reasonable (comment this explicitly in the header).
- Add modular code structure: helper functions, clear data model, and at least one unit-test-like demo.
- Include short examples of generated file contents (as docstring or comments).
- Provide error handling for malformed inputs and a CLI example for running the generator.
- Return **only** a single Markdown code block with the complete generator code.

Template
--------
Here's an extended version of the code that generates a `<FROMAT>` file with more complex file structures such as `<specific file structures>`:
```
<Generated Code>
```
'''

generator_mutation_structure_prompt_incre = '''
Goal
----
Incrementally transform an existing `<FROMAT>` generator to produce **richer file structures** (e.g., multiple logical sections, hierarchical segments, embedded metadata, and validation routines).

Requirements
- State clearly what structural changes are being made in the header comment.
- Add helper utilities for building and validating the new structure.
- Include example outputs and a short self-check routine.
- Keep the code self-contained (no hidden network calls) and provide a minimal dependency list if required.
- Output must be one Markdown code block containing only the generated code.

Template
--------
Here's an extended version of the code that generates a `<FROMAT>` file with more complex file structures such as `<specific file structures>`:
```
<Generated Code>
```
'''

pattern_based_mutation_prompt = '''
The original code:
```
<ORI>
```

The mutated code:
```
<MUT>
```

Task
----
Imitate the **mutation transformation** demonstrated above (how `<ORI>` became `<MUT>`) and **apply the same mutation approach** to the following target code:
```
<TARGET_CODE>
```

Deliverable Requirements
- Begin your output with a one-line summary describing the primary change (e.g., "The mutated code differs from the original mainly in ...").
- Then produce the mutated version of `<TARGET_CODE>` that applies the same transformation pattern (preserve language and style consistency).
- Add brief inline comments showing which parts correspond to the mutation pattern where helpful.
- Include a short example or quick test demonstrating the mutated generator's new behavior.
- Output only: a single Markdown code block containing the mutated target code (do **not** include long explanations outside the code block).

Template
--------
"The mutated code" differs from "The original code" mainly in <changing/adding/... specific file features/structures>. We can apply the same mutation approach to the target code to obtain:
```
<The mutated code of the target code>
```
'''

# Improved system message used in your llm log
improved_system_message = (
    "You are an expert assistant that writes, debugs, and tests generator scripts. "
    "When asked to emit code, produce a single Markdown code block containing only the final code. "
    "Write clear top-of-file comments explaining: purpose, required dependencies, new features added, and a minimal usage demo. "
    "Prefer self-contained, dependency-light code and include small self-tests or `__main__` examples. "
    "Avoid network calls and interactive prompts. If a dependency is unavoidable, list it and provide a pure-Python fallback."
)

# -------------------- Mutation functions (use external py_utils for helpers) --------------------

def mutation_based_on_pattern(model, tmp_path, seeds_path, generators, output_path, mutation_log, relationship, mutation_pattern):
    cur_mutation_pattern = random.choice(mutation_pattern)
    ori_code_path = os.path.join(generators, cur_mutation_pattern[0]) 
    mutated_code_path = os.path.join(generators, cur_mutation_pattern[1]) 

    file_format = cur_mutation_pattern[0].split('-')[0]

    with open(ori_code_path, 'r') as file:
        ori_code = file.read()
    
    with open(mutated_code_path, 'r') as file:
        mutated_code = file.read()


    # get a generator (use bandit instead of pure random within the same format)
    generator_list = [f for f in os.listdir(generators) if os.path.isfile(os.path.join(generators, f))]
    filtered_list = [s for s in generator_list if s.startswith(file_format)]
    # Fallbacks so we don't crash when the set is empty
    if not filtered_list:
        filtered_list = generator_list
    if not filtered_list:
        print("[bandit] No generators found to mutate.")
        return False
    selector = BanditGeneratorSelector(filtered_list, state_path=_scheduler_state_path(mutation_log))
    target_generator = selector.select() or random.choice(filtered_list)
    # target_generator = 'tiff-2.py'
    target_generator_path = os.path.join(generators, target_generator)
    print(target_generator_path)
    with open(target_generator_path, 'r') as file:
        target_generator_code = file.read()
    
    target_generator_log = [
        {"role": "system", "content": improved_system_message},
    ]
    prompt = copy.deepcopy(pattern_based_mutation_prompt)
    prompt = prompt.replace("<ORI>", ori_code)
    prompt = prompt.replace("<MUT>", mutated_code)
    prompt = prompt.replace("<TARGET_CODE>", target_generator_code)
    target_generator_log.append({"role": "user", "content": prompt})


    # get raw llm
    mutated_generator, raw_llm = gen_code_debug(target_generator_log, 3, model, 0.7)

    # debug the code
    mutated_generator_debuged = self_debug(mutated_generator, 6, model, temperature = 0.2, output_path=os.path.dirname(tmp_path))
    ok = bool(mutated_generator_debuged)
    if ok:
        generator_cnt = count_files_in_directory(generators) + 1

        mv_files(tmp_path, seeds_path, file_format + "-" + str(generator_cnt))

        cur_generator_path = os.path.join(generators, file_format + "-" + str(generator_cnt) + ".py")
        with open(cur_generator_path, 'w') as file:
            file.write(mutated_generator_debuged) 

        # bandit feedback
        selector.update(target_generator, _success_reward(ok))
        return True
    else:
        # bandit feedback
        selector.update(target_generator, _success_reward(ok))
        return False


def mutation_based_on_predefined_mutators(model, tmp_path, seeds_path, generators, output_path, mutation_log, cur_mutator):

    if cur_mutator == "feature": # file feature mutation
        prompt_init = copy.deepcopy(generator_mutation_feature_prompt_init)
        prompt_incre = copy.deepcopy(generator_mutation_feature_prompt_incre)
    else: # file structure mutation
        prompt_init = copy.deepcopy(generator_mutation_structure_prompt_init)
        prompt_incre = copy.deepcopy(generator_mutation_structure_prompt_incre)

    # read the relationship
    relationship_path = os.path.join(mutation_log, "relationship.json")
    if os.path.exists(relationship_path):
        with open(relationship_path, 'r') as file:
            relationship = json.load(file)
    else:
        relationship = {}

    # step I: get a generator (bandit selection)
    generator_list = [f for f in os.listdir(generators) if os.path.isfile(os.path.join(generators, f))]
    if not generator_list:
        print("[bandit] No generators available in:", generators)
        return False
    selector = BanditGeneratorSelector(generator_list, state_path=_scheduler_state_path(mutation_log))
    target_generator = selector.select() or random.choice(generator_list)
    # target_generator = 'tiff-2.py'
    file_format = target_generator.split('-')[0]
    target_generator_path = os.path.join(generators, target_generator)
    print(target_generator_path)
    with open(target_generator_path, 'r') as file:
        target_generator_code = file.read()
    
    target_generator_log = [
        {"role": "system", "content": improved_system_message},
    ]
    init = copy.deepcopy(prompt_init)
    init = init.replace("<FROMAT>", file_format)
    init = init.replace("<TARGET_GENERATOR>", target_generator_code)
    target_generator_log.append({"role": "user", "content": init})
    # print(target_generator_log[-1]["content"])
    

    # get raw llm
    mutated_generator, raw_llm = gen_code_debug(target_generator_log, 3, model, 0.7)

    # debug the code
    mutated_generator_debuged = self_debug(mutated_generator, 6, model, temperature = 0.2, output_path=os.path.dirname(tmp_path))
    ok = bool(mutated_generator_debuged)
    if ok:
        # if debug successfully, replace the wrong code with the right code
        raw_llm.replace(mutated_generator, mutated_generator_debuged)

        generator_cnt = count_files_in_directory(generators) + 1

        mv_files(tmp_path, seeds_path, file_format + "-" + str(generator_cnt))

        cur_generator_path = os.path.join(generators, file_format + "-" + str(generator_cnt) + ".py")
        with open(cur_generator_path, 'w') as file:
            file.write(mutated_generator_debuged) 

        # # get seeds
        # mv_files(tmp_path, seeds_path, os.path.splitext(target_generator)[0])

        # save the relationship
        if target_generator not in relationship.keys():
            relationship[target_generator] = [file_format + "-" + str(generator_cnt) + ".py"]
        else:
            relationship[target_generator].append(file_format + "-" + str(generator_cnt) + ".py")
        with open(relationship_path, 'w') as file:
            json.dump(relationship, file)
        
        selector.update(target_generator, _success_reward(ok))
        return True
    else:
        selector.update(target_generator, _success_reward(ok))
        return False


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Description of your script.')
    parser.add_argument('--output', type=str, help='The path to store the output')
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)
    from py_utils.func import *
    from py_utils.llm_analysis import *
    from py_utils.llm_utils import *

    output_path = args.output

    with open('model_setting.json', 'r') as file:
        model_setting = json.load(file)

    model = model_setting["model"][0]
    print("model:", model)

    tmp_path = os.path.join(args.output, "tmp") # can not be changed
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)

    seeds_path = os.path.join(args.output, "gen_seeds")
    generators = os.path.join(args.output, "generators")
    mutation_log = os.path.join(args.output, "mutation_log")
    if not os.path.exists(seeds_path):
        os.makedirs(seeds_path)
    if not os.path.exists(generators):
        os.makedirs(generators)
    if not os.path.exists(mutation_log):
        os.makedirs(mutation_log)

    mutators = ["feature", "structure"]

    # read the relationship
    relationship_path = os.path.join(mutation_log, "relationship.json")
    if os.path.exists(relationship_path):
        with open(relationship_path, 'r') as file:
            relationship = json.load(file)
    else:
        relationship = None
    
    if relationship:
        # get successfully mutation
        mutation_pattern = []
        file_names = list_files(os.path.join(output_path, "queue"))
        roots = build_tree(file_names)
        for root in roots:
            num_nodes = print_tree(root)
            if num_nodes - 1 > 0:
                cur = root.orig_name + ".py"
                print("ID:", root.file_id)
                print("Original generator:", cur)
                print("Number of sub-nodes:", num_nodes - 1)  # Subtract 1 for the root node
                print("\n")

                for ori, mutated in relationship.items():
                    if cur in mutated:
                        mutation_pattern.append([ori, cur])

        print("mutation_pattern:", mutation_pattern)

        if len(mutation_pattern) != 0:
            mutators.append("pattern")

    print("mutators:", mutators)
    cur_mutator = random.choice(mutators)
    print("cur_mutator:", cur_mutator)

    if cur_mutator == "pattern":
        mutation_based_on_pattern(model, tmp_path, seeds_path, generators, output_path, mutation_log, relationship, mutation_pattern)
    else:
        mutation_based_on_predefined_mutators(model, tmp_path, seeds_path, generators, output_path, mutation_log, cur_mutator)