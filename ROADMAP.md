# Roadmap

## Fix co-author labeling in commit and PR messages to use chosen model not decision model

The current commit and PR messages label the decision model as "jev" or "Jev" in co-author fields. This is incorrect — the decision model is a pluggable component, and the chosen model is the one actually serving the request. Co-author labels should reference the chosen model (e.g. `z-ai/glm-5.3-flash`), not the decision model.

### Scope
- Commit messages
- PR titles and descriptions
- Any automation-generated attribution

### Rationale
The decision model (`typesafe/jev-1.13` today) is a routing tool, not the model doing the work. Attributing to it misrepresents which model the user actually received.

## Support Requesty

Add Requesty as a supported provider alongside OpenRouter, Fireworks, and others.

### Treat multiple providers as one

Treat multiple providers (Requesty, Fireworks, OpenRouter, ...) as one unified pool and choose models from any that the user has a valid key and credits for.

- The inventory sweep should check each provider's endpoint for discounts and serving health independently.
- The router should select from the combined pool, not restrict to a single provider.
- A model available on both Requesty and OpenRouter should be eligible from either, whichever offers the better discount or latency.
- Provider-specific API keys and base URLs should be configured per-provider, not hardcoded to OpenRouter.

### Scope
- `src/inventory.py` — sweep multiple providers
- `src/config.py` — provider config (keys, base URLs, enabled providers)
- `src/router.py` — candidate filter spans all providers
- `.env.example` — provider config template
- `README.md` — multi-provider setup docs

### Rationale
OpenRouter is the default today, but Requesty and Fireworks offer competitive pricing and models not available on OpenRouter. Treating them as one pool lets the decision model pick the best model regardless of provider, as long as the user has valid credentials and credits.
