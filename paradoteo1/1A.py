from collections import deque

class DFA:
    def __init__(self, states, alphabet, transitions, start_state, accept_states):
        self.states        = set(states)
        self.alphabet      = set(alphabet)
        self.transitions   = dict(transitions)       # mapping (state, symbol) → next_state
        self.start_state   = start_state
        self.accept_states = set(accept_states)

        # sanity checks
        assert start_state in self.states, "start_state must be a valid state"
        assert self.accept_states.issubset(self.states), "accept_states must be subset of states"
        for (s, sym), t in self.transitions.items():
            assert s in self.states and t in self.states, "transition uses unknown state"
            assert sym in self.alphabet, f"symbol {sym} not in alphabet"

    def accepts(self, input_sequence):
        """Standard DFA acceptance (with explicit end-marker)."""
        current = self.start_state
        for sym in input_sequence:
            if (current, sym) not in self.transitions:
                return False
            current = self.transitions[(current, sym)]
        # consume <END>
        if (current, "<END>") in self.transitions:
            current = self.transitions[(current, "<END>")]
        return current in self.accept_states

    def generate(self, max_sentences=10):
        """
        Reconstruct up to `max_sentences` by BFS from start_state:
        returns list of lists of symbols (without the <END> token).
        """
        results = []
        # queue of (current_state, path_so_far)
        queue = deque([(self.start_state, [])])

        while queue and len(results) < max_sentences:
            state, path = queue.popleft()

            # explore all outgoing transitions
            for (s, sym), nxt in self.transitions.items():
                if s != state:
                    continue

                # if it's the <END> symbol and leads to an accept, record path
                if sym == "<END>" and nxt in self.accept_states:
                    results.append(path.copy())
                    if len(results) >= max_sentences:
                        break
                # otherwise extend the path
                elif sym != "<END>":
                    queue.append((nxt, path + [sym]))

        return results


# ─────────────────────────────────────────────────────────────────────────────
# 1) Define your two sentences
# ─────────────────────────────────────────────────────────────────────────────
sent1 = "Today is our dragon boat festival"
sent2 = "We should be grateful"

# split into words
words1 = sent1.split()
words2 = sent2.split()

# ─────────────────────────────────────────────────────────────────────────────
# 2) Build the DFA
# ─────────────────────────────────────────────────────────────────────────────
states = {
    "q0",            # start
    # for sentence 1
    *{f"q{i+1}" for i in range(len(words1))},
    "q_f1",
    # for sentence 2
    *{f"r{i+1}" for i in range(len(words2))},
    "q_f2"
}

alphabet = set(words1 + words2) | {"<END>"}
transitions = {}

# auto-generate transitions for sent1
cur = "q0"
for i, w in enumerate(words1, start=1):
    transitions[(cur, w)] = f"q{i}"
    cur = f"q{i}"
transitions[(cur, "<END>")] = "q_f1"

# auto-generate transitions for sent2
cur = "q0"
for i, w in enumerate(words2, start=1):
    transitions[(cur, w)] = f"r{i}"
    cur = f"r{i}"
transitions[(cur, "<END>")] = "q_f2"

start_state   = "q0"
accept_states = {"q_f1", "q_f2"}

dfa = DFA(states, alphabet, transitions, start_state, accept_states)


# ─────────────────────────────────────────────────────────────────────────────
# 3) Recognition tests
# ─────────────────────────────────────────────────────────────────────────────
def check(sentence: str):
    toks = sentence.split()
    ok = dfa.accepts(toks)
    print(f"{'✔' if ok else '✘'} {' '.join(toks)}")

print("Recognition:")
check(sent1)                     # ✔
check(sent2)                     # ✔
check("Today is our boat")       # ✘
check("We are grateful")         # ✘
print()


# ─────────────────────────────────────────────────────────────────────────────
# 4) Reconstruction
# ─────────────────────────────────────────────────────────────────────────────
print("Reconstructed sentences:")
for seq in dfa.generate(max_sentences=2):
    print(" •", " ".join(seq))
