# Roadmap

## Fix co-author labeling in commit and PR messages to use chosen model not decision model

The current commit and PR messages label the decision model as "jev" or "Jev" in co-author fields. This is incorrect — the decision model is a pluggable component, and the chosen model is the one actually serving the request. Co-author labels should reference the chosen model (e.g. `z-ai/glm-5.3-flash`), not the decision model.

### Scope
- Commit messages
- PR titles and descriptions
- Any automation-generated attribution

### Rationale
The decision model (`typesafe/jev-1.13` today) is a routing tool, not the model doing the work. Attributing to it misrepresents which model the user actually received.
